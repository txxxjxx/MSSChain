"""H2PPO training implementation. Use h2ppo.py train or retrain for release defaults.

The public entry point initializes a new model, trains in original CSV order,
and runs placement evaluation separately on the original CSV prefix.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

import blockchain.params as p


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',nargs='+',default=[p.FileInput1,p.FileInput2,p.FileInput3])
    parser.add_argument('--output',type=Path,default=Path('results/h2ppo-5m-arrival8000-seed0'))
    parser.add_argument('--train-transactions',type=int,default=5_000_000)
    parser.add_argument('--evaluation-transactions',type=int,default=2_000_000)
    parser.add_argument('--arrival-rate',type=int,default=8_000,
                        help='Mean total-system transactions per one-second decision slot')
    parser.add_argument('--arrival-process',choices=('fixed','poisson'),default='poisson',
                        help='Slot arrival process used by this experiment')
    parser.add_argument('--queue-service-mode',choices=('protocol','paper_queue'),
                        default='protocol',
                        help=('protocol runs cross-shard request/response messages; paper_queue '
                              'uses the transaction-count service B_i(t) in Eqs. (10)-(11)'))
    parser.add_argument('--critical-capacity-tps',type=float,default=None,
                        help=('For paper_queue, calibrate the declared full-budget Eq. (10) '
                              'reference allocation to this total-system service rate'))
    parser.add_argument('--peak-queue-drift-weight',type=float,default=0.,
                        help=('Weight of max_i Q_i(t)^2 in the Lyapunov potential. '
                              'This trains worst-shard recovery without inference guards'))
    parser.add_argument('--training-arrival-schedule',nargs='+',default=None,
                        metavar='SECOND:RATE',
                        help=('Optional training-only total-system arrival schedule, e.g. '
                              '0:2000 500:8000 600:2000. One decision slot is one second.'))
    parser.add_argument('--report-interval',type=int,default=50_000)
    parser.add_argument('--shards',type=int,default=32)
    parser.add_argument('--control-mode',choices=('hierarchical','flat'),default='hierarchical',
                        help='Hierarchical H2PPO or a single shared flat PPO action')
    parser.add_argument('--control-architecture',choices=('dense','equivariant'),default='dense',
                        help=('Low-level controller architecture. equivariant shares one '
                              'per-shard controller across shard labels'))
    parser.add_argument('--window-size',type=int,default=100_000)
    parser.add_argument('--block-min',type=int,default=2_000,
                        help='Minimum H2PPO block-size action in transaction equivalents')
    parser.add_argument('--block-max',type=int,default=6_000,
                        help='Maximum H2PPO block-size action in transaction equivalents')
    parser.add_argument('--placement-chunk-size',type=int,default=256)
    parser.add_argument('--ppo-repeat',type=int,default=10)
    parser.add_argument('--placement-update-size',type=int,default=2048,
                        help='Random placement decisions used by each PPO optimizer pass')
    parser.add_argument('--rollout-steps',type=int,default=8,
                        help='Arrival slots collected before each PPO update')
    parser.add_argument('--placement-temporal-weight',type=float,default=.25,
                        help='Weight of discounted rollout return in placement advantages')
    parser.add_argument('--actor-lr',type=float,default=1e-5)
    parser.add_argument('--critic-lr',type=float,default=1e-4)
    parser.add_argument('--target-kl',type=float,default=.02)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--torch-threads',type=int,default=min(8,os.cpu_count() or 1))
    parser.add_argument('--resource-budget',type=float,default=250.)
    parser.add_argument('--cpu-budget',type=float,default=None,
                        help='Total network CPU budget; defaults to --resource-budget')
    parser.add_argument('--bandwidth-budget',type=float,default=None,
                        help='Total network bandwidth budget; defaults to --resource-budget')
    parser.add_argument('--bandwidth-unit',choices=('kb/s','Mb/s'),default='kb/s',
                        help='Declared bandwidth unit for the selected resource model')
    parser.add_argument('--cpu-weight',type=float,default=5.,
                        help='CPU coefficient in paper Eq. (10)')
    parser.add_argument('--bandwidth-weight',type=float,default=3.,
                        help='Bandwidth coefficient in paper Eq. (10)')
    parser.add_argument('--resource-model',
                        choices=('huang_eq2','paper_eq10','physical_bottleneck'),
                        default='huang_eq2')
    parser.add_argument('--valid-transaction-ratio',type=float,default=.8)
    parser.add_argument('--cpu-tps-per-core',type=float,default=312.5,
                        help='Candidate 500-byte transaction equivalents/s per CPU core')
    parser.add_argument('--locality-weight',type=float,default=.65)
    parser.add_argument('--reward-locality-weight',type=float,default=None,
                        help='Environment reward alpha; defaults to locality-weight')
    parser.add_argument('--load-weight',type=float,default=4.)
    parser.add_argument('--prior-temperature',type=float,default=16.,
                        help=('Softmax temperature for the paper-state placement warm start; '
                              'larger values produce a softer prior'))
    parser.add_argument('--decay-seconds',type=float,default=86_400.)
    parser.add_argument('--preserve-dependency-magnitude',action='store_true',
                        help=('Expose the raw paper TD(u) vector to the policy instead of L1 '
                              'normalizing away its absolute time-decay magnitude'))
    parser.add_argument('--cross-penalty',type=float,default=0.,
                        help='Per-transaction training penalty for a cross-shard placement')
    parser.add_argument('--balance-sensitivity',type=float,default=8.,
                        help='Exponential reward sensitivity to relative shard-load deviation')
    parser.add_argument('--load-feature-overload-power',type=float,default=None,
                        help='Use max(relative_load-1,0)^power in the policy observation')
    parser.add_argument('--queue-prior-weight',type=float,default=0.,
                        help=('Trainable warm-start penalty on normalized per-shard Q(t); '
                              'zero preserves the checkpoint placement scorer'))
    parser.add_argument('--queue-readiness-state',action='store_true',
                        help='Add per-shard immediately-serviceable and parent-blocked queues')
    parser.add_argument('--address-affinity-weight',type=float,default=0.,
                        help='Online seen-address affinity weight in the locality prior')
    parser.add_argument('--dependency-weight',type=float,default=None,
                        help='Explicit TD(u) weight when resetting the placement prior')
    parser.add_argument('--neighbor-count-weight',type=float,default=None,
                        help='Explicit Nbr(u) weight when resetting the placement prior')
    parser.add_argument('--neighbor-location-weight',type=float,default=None,
                        help='Explicit ANbr(u) weight when resetting the placement prior')
    parser.add_argument('--reset-placement-prior',action='store_true',
                        help=('After loading a checkpoint, reset only the placement scorer '
                              'from paper-state weights; retain the learned stability controller'))
    parser.add_argument('--scale-placement-prior-alpha',type=float,default=None,
                        help=('After loading a checkpoint, retain its learned placement network '
                              'and rescale TD/Nbr/ANbr by alpha/0.5 and LB by (1-alpha)/0.5'))
    parser.add_argument('--placement-locality-prior-scale',type=float,default=1.,
                        help=('Training-initialization multiplier for the retained TD/Nbr/ANbr '
                              'placement-prior coefficients; PPO remains free to update them'))
    parser.add_argument('--freeze-placement-policy',action='store_true',
                        help=('Train only hierarchical block/resource control and the critic; '
                              'preserve the checkpoint transaction-placement policy'))
    parser.add_argument('--freeze-control-policy',action='store_true',
                        help=('Train only hierarchical transaction placement and the critic; '
                              'preserve the checkpoint block/resource controller'))
    parser.add_argument('--evaluation-load-guard-cv',type=float,default=None,
                        help='Per-window evaluation CV trade-off limit')
    parser.add_argument('--evaluation-load-guard-window',type=int,default=None,
                        help='Defaults to report-interval')
    parser.add_argument('--evaluation-cross-target',type=float,default=None,
                        help='Soft per-window crossing target; e.g. 0.25')
    parser.add_argument('--training-load-guard-cv',type=float,default=None,
                        help='Optional load CV cap applied to the training trajectory')
    parser.add_argument('--training-cross-target',type=float,default=None,
                        help='Optional crossing target applied to the training trajectory')
    parser.add_argument('--training-guard-window',type=int,default=None,
                        help='Defaults to the complete training range')
    parser.add_argument('--evaluation-from-start',action='store_true',
                        help='Evaluate the saved policy again on transactions [0,N)')
    parser.add_argument('--resume-checkpoint',type=Path,default=None,
                        help='Warm-start actor and critic from a compatible H2PPO checkpoint')
    parser.add_argument('--resume-placement-only',action='store_true',
                        help=('Load and optionally freeze only placement parameters from the '
                              'checkpoint while initializing a new low-level controller'))
    parser.add_argument('--train-only',action='store_true')
    return parser.parse_args()


def parse_arrival_schedule(values):
    if not values:
        return None
    schedule=[]
    for value in values:
        try:
            second_text,rate_text=value.split(':',1)
            second,rate=int(second_text),int(rate_text)
        except (AttributeError,TypeError,ValueError) as exc:
            raise ValueError(
                f'Invalid training arrival entry {value!r}; expected SECOND:RATE'
            ) from exc
        schedule.append((second,rate))
    if schedule[0][0] != 0:
        raise ValueError('Training arrival schedule must start at second 0')
    if any(second < 0 or rate < 1 for second,rate in schedule):
        raise ValueError('Scheduled seconds must be nonnegative and rates must be positive')
    if any(right[0] <= left[0] for left,right in zip(schedule,schedule[1:])):
        raise ValueError('Training arrival schedule seconds must be strictly increasing')
    return schedule


def validate(args):
    args.training_arrival_schedule=parse_arrival_schedule(args.training_arrival_schedule)
    positive = ('train_transactions','evaluation_transactions','arrival_rate','report_interval',
                'shards','window_size','placement_chunk_size','ppo_repeat','placement_update_size',
                'rollout_steps','torch_threads')
    if any(getattr(args,name) < 1 for name in positive):
        raise ValueError('All count, rate, shard, chunk, and thread arguments must be positive')
    if args.evaluation_transactions % args.report_interval:
        raise ValueError('Evaluation count must be divisible by report-interval')
    if not 0 < args.block_min < args.block_max:
        raise ValueError('require 0 < block-min < block-max')
    if args.evaluation_load_guard_cv is not None and args.evaluation_load_guard_cv <= 0:
        raise ValueError('evaluation-load-guard-cv must be positive')
    if args.evaluation_load_guard_window is not None and args.evaluation_load_guard_window < 1:
        raise ValueError('evaluation-load-guard-window must be positive')
    if args.evaluation_cross_target is not None and not 0 <= args.evaluation_cross_target <= 1:
        raise ValueError('evaluation-cross-target must be between zero and one')
    if not np.isfinite(args.cross_penalty) or args.cross_penalty < 0:
        raise ValueError('cross-penalty must be nonnegative')
    if args.training_load_guard_cv is not None and args.training_load_guard_cv <= 0:
        raise ValueError('training-load-guard-cv must be positive')
    if args.training_cross_target is not None and not 0 <= args.training_cross_target <= 1:
        raise ValueError('training-cross-target must be between zero and one')
    if args.training_guard_window is not None and args.training_guard_window < 1:
        raise ValueError('training-guard-window must be positive')
    if args.reward_locality_weight is not None and not 0 <= args.reward_locality_weight <= 1:
        raise ValueError('reward-locality-weight must be between zero and one')
    if not np.isfinite(args.decay_seconds) or args.decay_seconds <= 0:
        raise ValueError('decay-seconds must be positive')
    if not np.isfinite(args.balance_sensitivity) or args.balance_sensitivity <= 0:
        raise ValueError('balance-sensitivity must be positive')
    if (args.load_feature_overload_power is not None and
            (not np.isfinite(args.load_feature_overload_power) or args.load_feature_overload_power <= 0)):
        raise ValueError('load-feature-overload-power must be positive')
    if not np.isfinite(args.queue_prior_weight) or args.queue_prior_weight < 0:
        raise ValueError('queue-prior-weight must be nonnegative and finite')
    if not np.isfinite(args.address_affinity_weight) or args.address_affinity_weight < 0:
        raise ValueError('address-affinity-weight must be nonnegative')
    paper_prior=(args.dependency_weight,args.neighbor_count_weight,
                 args.neighbor_location_weight)
    if any(value is not None and (not np.isfinite(value) or value < 0)
           for value in paper_prior):
        raise ValueError('Paper-state prior weights must be nonnegative finite values')
    if not np.isfinite(args.prior_temperature) or args.prior_temperature <= 0:
        raise ValueError('prior-temperature must be positive and finite')
    if args.reset_placement_prior and args.resume_checkpoint is None:
        raise ValueError('--reset-placement-prior requires --resume-checkpoint')
    if args.scale_placement_prior_alpha is not None:
        if args.resume_checkpoint is None:
            raise ValueError('--scale-placement-prior-alpha requires --resume-checkpoint')
        if not 0 <= args.scale_placement_prior_alpha <= 1:
            raise ValueError('--scale-placement-prior-alpha must be between zero and one')
        if args.reset_placement_prior:
            raise ValueError('reset and scale placement-prior modes are mutually exclusive')
    if (not np.isfinite(args.placement_locality_prior_scale)
            or args.placement_locality_prior_scale <= 0):
        raise ValueError('--placement-locality-prior-scale must be positive and finite')
    if args.placement_locality_prior_scale != 1:
        if args.resume_checkpoint is None:
            raise ValueError('--placement-locality-prior-scale requires --resume-checkpoint')
        if args.reset_placement_prior:
            raise ValueError('reset and locality-scale placement-prior modes are mutually exclusive')
    if args.freeze_placement_policy and args.control_mode != 'hierarchical':
        raise ValueError('--freeze-placement-policy requires hierarchical control')
    if args.freeze_control_policy and args.control_mode != 'hierarchical':
        raise ValueError('--freeze-control-policy requires hierarchical control')
    if args.freeze_placement_policy and args.freeze_control_policy:
        raise ValueError('placement and control policies cannot both be frozen')
    if args.control_architecture == 'equivariant' and args.control_mode != 'hierarchical':
        raise ValueError('--control-architecture equivariant requires hierarchical control')
    if args.resume_placement_only and args.resume_checkpoint is None:
        raise ValueError('--resume-placement-only requires --resume-checkpoint')
    if not np.isfinite(args.placement_temporal_weight) or args.placement_temporal_weight < 0:
        raise ValueError('placement-temporal-weight must be nonnegative')
    if (args.critical_capacity_tps is not None and
            (not np.isfinite(args.critical_capacity_tps) or args.critical_capacity_tps <= 0)):
        raise ValueError('critical-capacity-tps must be positive and finite')
    if (not np.isfinite(args.peak_queue_drift_weight)
            or args.peak_queue_drift_weight < 0):
        raise ValueError('peak-queue-drift-weight must be nonnegative and finite')
    if args.queue_service_mode == 'protocol' and args.critical_capacity_tps is not None:
        raise ValueError('critical-capacity-tps is only defined for paper_queue service')
    resources = (args.resource_budget, args.cpu_budget, args.bandwidth_budget,
                 args.cpu_weight, args.bandwidth_weight, args.cpu_tps_per_core)
    if any(value is not None and (not np.isfinite(value) or value <= 0) for value in resources):
        raise ValueError('Resource budgets and weights must be positive finite values')
    if (not np.isfinite(args.valid_transaction_ratio) or
            not 0 < args.valid_transaction_ratio <= 1):
        raise ValueError('valid-transaction-ratio must lie in (0,1]')
    if (args.training_arrival_schedule and not args.train_only
            and not args.evaluation_from_start):
        raise ValueError(
            'Scheduled training followed by evaluation requires --evaluation-from-start '
            'so evaluation has a declared constant arrival rate'
        )
    if args.resume_checkpoint is not None and not args.resume_checkpoint.is_file():
        raise FileNotFoundError(f'Missing resume checkpoint: {args.resume_checkpoint}')
    missing = [path for path in args.input if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f'Missing input files: {missing}')


def dump_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def write_evaluation_csv(path,rows):
    fields = list(rows[0]) if rows else []
    with path.open('w',newline='',encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for row in rows:
            cooked = {key:(json.dumps(value,separators=(',',':')) if isinstance(value,list) else value)
                      for key,value in row.items()}
            writer.writerow(cooked)


def main():
    args = parse_args()
    validate(args)
    args.output.mkdir(parents=True,exist_ok=True)
    p.ShardNum = args.shards
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.torch_threads)

    # Imports occur after ShardNum is fixed because legacy modules read it at runtime.
    from msschain_env import Supervisor
    from algo.h2ppo import HPPO

    started = time.time()
    log_path = args.output/'run.log'
    log_stream = log_path.open('a',encoding='utf-8',buffering=1)

    def log(message):
        elapsed = time.time()-started
        line = f'[{elapsed:10.1f}s] {message}'
        print(line,flush=True)
        log_stream.write(line+'\n')

    total = args.train_transactions+args.evaluation_transactions
    guard_window = (args.report_interval if args.evaluation_load_guard_window is None
                    else args.evaluation_load_guard_window)
    training_guard_window = (args.train_transactions if args.training_guard_window is None
                             else args.training_guard_window)
    reward_locality_weight = (args.locality_weight if args.reward_locality_weight is None
                              else args.reward_locality_weight)
    cpu_budget = args.resource_budget if args.cpu_budget is None else args.cpu_budget
    bandwidth_budget = (args.resource_budget if args.bandwidth_budget is None
                        else args.bandwidth_budget)
    budget = (cpu_budget,bandwidth_budget)
    initial = (cpu_budget/args.shards,bandwidth_budget/args.shards)
    resource_weights = (args.cpu_weight,args.bandwidth_weight)
    service_capacity_scale=1.0
    reference_capacity_tps=None
    if args.queue_service_mode == 'paper_queue' and args.critical_capacity_tps is not None:
        # Eq. (10) uses normalized simulation resource units.  Calibrate their
        # conversion once at the declared full-budget, equal-share, maximum-
        # block reference point; runtime actions remain continuous and uncapped.
        per_shard_factor=sum(np.sqrt(weight*amount/args.shards)
                             for weight,amount in zip(resource_weights,budget))
        per_shard_reference=min(
            args.block_max,
            args.block_max*args.valid_transaction_ratio/8.0*per_shard_factor,
        )
        reference_capacity_tps=args.shards*per_shard_reference
        service_capacity_scale=args.critical_capacity_tps/reference_capacity_tps
    protocol = {
        'method':('Flat PPO' if args.control_mode == 'flat' else 'H2PPO'),
        'control_mode':args.control_mode,
        'dataset_order':'CSV order across input files',
        'training_range':[0,args.train_transactions],
        'evaluation_range':([0,args.evaluation_transactions] if args.evaluation_from_start
                            else [args.train_transactions,total]),
        'evaluation_replays_training_prefix':bool(args.evaluation_from_start),
        'training_transactions':args.train_transactions,
        'evaluation_transactions':args.evaluation_transactions,
        'arrival_rate_transactions_per_second':args.arrival_rate,
        'arrival_process':args.arrival_process,
        'queue_service_mode':args.queue_service_mode,
        'service_capacity_calibration':{
            'critical_capacity_tps':args.critical_capacity_tps,
            'raw_equal_share_full_budget_reference_tps':reference_capacity_tps,
            'multiplicative_scale':service_capacity_scale,
            'runtime_capacity_clamped_to_critical_rate':False,
        },
        'lyapunov_queue_potential':{
            'definition':'sum_i Q_i(t)^2 + weight * max_i Q_i(t)^2',
            'peak_queue_drift_weight':args.peak_queue_drift_weight,
            'enabled_in_training_reward':True,
            'inference_action_override':False,
        },
        'arrival_scope':'total system across all shards',
        'training_arrival_schedule_seconds':(
            None if args.training_arrival_schedule is None else
            [{'start_second':second,'arrival_rate_tps':rate}
             for second,rate in args.training_arrival_schedule]
        ),
        'training_schedule_event_order':(
            'set total-system arrival rate, generate/admit arrivals, H2PPO placement and '
            'post-placement stability control, immediate consensus/service'
        ),
        'evaluation_report_interval':args.report_interval,
        'num_shards':args.shards,'window_size':args.window_size,
        'block_size_action_range_transactions':[args.block_min,args.block_max],
        'placement_chunk_size':args.placement_chunk_size,
        'placement_update_size':args.placement_update_size,
        'rollout_steps':args.rollout_steps,
        'placement_temporal_weight':args.placement_temporal_weight,
        'primary_cross_shard_rate_definition':(
            'cross_shard_transactions_in_each_report_window / '
            'all_transactions_in_that_window_including_coinbase'),
        'primary_load_cv_definition':(
            'std(protocol_workload_per_shard_in_each_report_window) / '
            'mean(protocol_workload_per_shard_in_each_report_window), where protocol workload '
            'counts internal originals, cross-shard originals, and generated subtransactions'),
        'primary_load_balance_degree_definition':'1 / (1 + window_load_cv)',
        'cumulative_metrics':'retained as secondary fields only',
        'architecture':(
            'shared Flat PPO with simultaneous placement/block/resource action'
            if args.control_mode == 'flat'
            else 'shared H2PPO with hierarchical post-placement resource control'
        ),
        'control_architecture':args.control_architecture,
        'state_space':{
            'paper_macro_state':['CTx(t)','A(t)','Q(t)','S^B(t)'],
            'paper_transaction_features':['TD(u)','Nbr(u)','ANbr(u)'],
            'implementation_extensions':['current CPU allocation','current bandwidth allocation',
                                         'active-window relative shard load',
                                         'immediately-serviceable queue','parent-blocked queue'],
            'placement_actor_input':'paper macro + paper transaction features + extensions',
            'stability_actor_input':'paper macro + current resource/load extensions',
        },
        'paper_state_space_fully_included':True,
        'queue_readiness_state_enabled':bool(args.queue_readiness_state),
        'resume_checkpoint':(
            None if args.resume_checkpoint is None else str(args.resume_checkpoint.resolve())
        ),
        'resume_placement_only':bool(args.resume_placement_only),
        'ppo_repeat_max':args.ppo_repeat,'target_kl':args.target_kl,
        'actor_learning_rate':args.actor_lr,'critic_learning_rate':args.critic_lr,
        'locality_prior':{'locality_weight':args.locality_weight,'load_weight':args.load_weight,
                          'temperature':args.prior_temperature},
        'paper_state_placement_prior':{
            'reset_after_checkpoint_load':bool(args.reset_placement_prior),
            'scaled_after_checkpoint_load_alpha':args.scale_placement_prior_alpha,
            'load_weight':args.load_weight,
            'TD_weight':(1-args.locality_weight if args.dependency_weight is None
                         else args.dependency_weight),
            'Nbr_weight':(args.locality_weight if args.neighbor_count_weight is None
                          else args.neighbor_count_weight),
            'ANbr_weight':(args.address_affinity_weight if args.neighbor_location_weight is None
                           else args.neighbor_location_weight),
            'temperature':args.prior_temperature,
            'retained_locality_coefficient_scale':args.placement_locality_prior_scale,
        },
        'placement_policy_frozen_during_training':bool(args.freeze_placement_policy),
        'control_policy_frozen_during_training':bool(args.freeze_control_policy),
        'placement_objective_locality_weight':args.locality_weight,
        'reward_locality_weight':reward_locality_weight,
        'dependency_decay_seconds':args.decay_seconds,
        'balance_reward_relative_deviation_sensitivity':args.balance_sensitivity,
        'load_feature_overload_power':args.load_feature_overload_power,
        'queue_prior_weight':args.queue_prior_weight,
        'address_affinity_weight':args.address_affinity_weight,
        'placement_cross_penalty':args.cross_penalty,
        'training_load_guard_cv':args.training_load_guard_cv,
        'training_cross_target':args.training_cross_target,
        'training_guard_window':training_guard_window,
        'evaluation_load_guard_cv':args.evaluation_load_guard_cv,
        'evaluation_load_guard_window':guard_window,
        'evaluation_cross_target':args.evaluation_cross_target,
        'resource_budget_each_type_network_total':list(budget),
        'resource_weights_cpu_bandwidth':list(resource_weights),
        'resource_model':args.resource_model,
        'valid_transaction_ratio':args.valid_transaction_ratio,
        'cpu_candidate_tps_per_core':args.cpu_tps_per_core,
        'resource_units':['CPU cores',args.bandwidth_unit],
        'seed':args.seed,'torch_threads':args.torch_threads,
        'input_files':[str(Path(path)) for path in args.input],
        'retain_blocks':False,'track_latency':False,
    }
    dump_json(args.output/'protocol.json',protocol)
    log(f"protocol written; train={args.train_transactions:,}, evaluate={args.evaluation_transactions:,}, "
        f"arrival={args.arrival_rate:,} Tx/s, shards={args.shards}")

    def make_environment(transaction_limit,arrival_rate=None):
        return Supervisor(file_paths=args.input,
            batch_size=(args.arrival_rate if arrival_rate is None else arrival_rate),
            window_size=args.window_size,transaction_limit=transaction_limit,retain_blocks=False,
            track_latency=False,enhanced_observation=True,resource_control=True,
            stability_observation=args.queue_readiness_state,
            resource_max=budget,initial_resources=initial,network_resource_budget=budget,
            resource_weights=resource_weights,
            resource_model=args.resource_model,
            valid_transaction_ratio=args.valid_transaction_ratio,
            cpu_tps_per_core=args.cpu_tps_per_core,
            block_min=args.block_min,block_max=args.block_max,
            arrival_process=args.arrival_process,
            arrival_order='csv',
            alpha=reward_locality_weight,cross_penalty=args.cross_penalty,
            decay_seconds=args.decay_seconds,zeta=args.balance_sensitivity,
            load_feature_overload_power=args.load_feature_overload_power,
            address_affinity=args.address_affinity_weight > 0,seed=args.seed,
            queue_service_mode=args.queue_service_mode,
            service_capacity_scale=service_capacity_scale,
            peak_queue_drift_weight=args.peak_queue_drift_weight,
            use_time_decay=True,
            use_spent_dependency=True,
            use_transaction_dependency=True,
            use_lyapunov_reward=True,
            normalize_dependency_observation=not args.preserve_dependency_magnitude)

    training_rate=(args.arrival_rate if args.training_arrival_schedule is None
                   else args.training_arrival_schedule[0][1])
    env = make_environment(total,training_rate)
    architecture = 'shared'
    agent = HPPO(env.observation_dim,args.shards,architecture=architecture,
        control_mode=args.control_mode,repeat_time=args.ppo_repeat,batch_size=args.rollout_steps,
        lr_a=args.actor_lr,lr_c=args.critic_lr,target_kl=args.target_kl,
        bmin=args.block_min,bmax=args.block_max,
        resource_max=budget,network_resource_budget=budget,seed=args.seed,
        placement_minibatch_size=args.placement_update_size,
        placement_guard_cv=args.training_load_guard_cv,
        placement_guard_window=training_guard_window,
        placement_cross_target=args.training_cross_target,
        placement_temporal_weight=args.placement_temporal_weight,
        freeze_placement=args.freeze_placement_policy,
        control_architecture=args.control_architecture,
        freeze_control=args.freeze_control_policy)
    explicit_prior=any(value is not None for value in
                       (args.dependency_weight,args.neighbor_count_weight,
                        args.neighbor_location_weight))
    if explicit_prior:
        agent.initialize_paper_state_prior(
            load_weight=args.load_weight,
            dependency_weight=(1-args.locality_weight if args.dependency_weight is None
                               else args.dependency_weight),
            neighbor_count_weight=(args.locality_weight if args.neighbor_count_weight is None
                                   else args.neighbor_count_weight),
            neighbor_location_weight=(args.address_affinity_weight
                                      if args.neighbor_location_weight is None
                                      else args.neighbor_location_weight),
            temperature=args.prior_temperature,reset_control=True)
    else:
        agent.initialize_locality_prior(args.locality_weight,args.load_weight,args.prior_temperature,
                                        args.address_affinity_weight)
    if args.resume_checkpoint is not None:
        saved=torch.load(args.resume_checkpoint,map_location='cpu',weights_only=False)
        compatible=(int(saved.get('num_states',-1)) == env.observation_dim
                    and int(saved.get('num_shards',-1)) == args.shards
                    and saved.get('architecture') == architecture
                    and saved.get('control_mode') == args.control_mode
                    and int(saved.get('bmin',-1)) == args.block_min
                    and int(saved.get('bmax',-1)) == args.block_max)
        if not compatible:
            raise ValueError('Resume checkpoint architecture/state/action dimensions are incompatible')
        if args.resume_placement_only:
            agent.load_placement_model(saved['actor'],saved.get('critic'))
            log(f'warm-started placement and critic from {args.resume_checkpoint.resolve()}; '
                'initialized a new low-level controller')
        else:
            saved_control_architecture=saved.get('control_architecture','dense')
            if saved_control_architecture != args.control_architecture:
                raise ValueError('Resume checkpoint control architecture is incompatible')
            agent.load_model(saved['actor'],saved['critic'])
            log(f'warm-started actor and critic from {args.resume_checkpoint.resolve()}')
        if args.reset_placement_prior:
            dependency_weight=(1-args.locality_weight if args.dependency_weight is None
                               else args.dependency_weight)
            neighbor_count_weight=(args.locality_weight
                                   if args.neighbor_count_weight is None
                                   else args.neighbor_count_weight)
            neighbor_location_weight=(args.address_affinity_weight
                                      if args.neighbor_location_weight is None
                                      else args.neighbor_location_weight)
            agent.initialize_paper_state_prior(
                load_weight=args.load_weight,
                dependency_weight=dependency_weight,
                neighbor_count_weight=neighbor_count_weight,
                neighbor_location_weight=neighbor_location_weight,
                temperature=args.prior_temperature,
                reset_control=False,
            )
            log('reset placement scorer from TD/Nbr/ANbr/load weights; retained stability controller')
        elif args.scale_placement_prior_alpha is not None:
            alpha=float(args.scale_placement_prior_alpha)
            beta=1.-alpha
            with torch.no_grad():
                weights=agent.actor.prior.weight
                feature_start=agent.num_states//agent.num_shards-3
                weights[0,6] *= beta/.5
                weights[0,feature_start:feature_start+3] *= alpha/.5
            log(f'rescaled retained placement prior for alpha={alpha:g}, beta={beta:g}')
        if args.placement_locality_prior_scale != 1:
            with torch.no_grad():
                weights=agent.actor.prior.weight
                feature_start=agent.num_states//agent.num_shards-3
                weights[0,feature_start:feature_start+3] *= args.placement_locality_prior_scale
            log('scaled retained TD/Nbr/ANbr placement prior by '
                f'{args.placement_locality_prior_scale:g} for trainable initialization')
    if args.queue_prior_weight:
        agent.add_queue_avoidance_prior(args.queue_prior_weight)
        log(f'added trainable Q(t) placement-prior coefficient '
            f'-{args.queue_prior_weight:g}')

    state = env.reset()
    env.begin_metric_reporting(args.report_interval)
    training_updates = []
    training_started = time.time()
    training_reported = 0
    next_checkpoint = 500_000
    pending_rollout_steps = 0
    latest_update = {'approx_kl':0.}
    schedule_index=0
    while env.accepted < args.train_transactions:
        if args.training_arrival_schedule is not None:
            while (schedule_index+1 < len(args.training_arrival_schedule)
                   and env.steps >= args.training_arrival_schedule[schedule_index+1][0]):
                schedule_index += 1
                scheduled_second,scheduled_rate=args.training_arrival_schedule[schedule_index]
                state=env.set_arrival_rate(scheduled_rate,resample_current=True)
                log(f'training arrival transition at t={scheduled_second}s -> '
                    f'{scheduled_rate:,} Tx/s')
        remaining = args.train_transactions-env.accepted
        if len(env.arriveTxArrive) > remaining:
            state = env.limit_current_arrivals(remaining)
        state,reward,done = agent.interact_batched(
            env,states=state,chunk_size=args.placement_chunk_size)
        if done and env.accepted < args.train_transactions:
            raise RuntimeError(f'Dataset exhausted during training at {env.accepted:,} transactions')
        pending_rollout_steps += 1
        updated = pending_rollout_steps >= args.rollout_steps or env.accepted == args.train_transactions
        if updated:
            latest_update = agent.update(agent.value(state))
            latest_update.update(step=env.steps,accepted=env.accepted,reward=reward,
                                 exploration_cross_shard_rate=env.cross_count/max(1,env.accepted),
                                 rolling_load_cv=env.metrics['load_cv'])
            training_updates.append(latest_update)
            pending_rollout_steps = 0
        while training_reported < len(env.metric_series):
            row = env.metric_series[training_reported]
            log(f"training window ending {row['transactions']:,}/{args.train_transactions:,}; "
                f"window cross/all={row['interval_cross_shard_rate']:.6f}; "
                f"window balance={row['interval_protocol_load_balance_degree']:.6f}; "
                f"window CV={row['interval_protocol_load_cv']:.6f}; "
                f"block(mean/min/max)={row['model_block_size_mean']:.1f}/"
                f"{row['model_block_size_min']}/{row['model_block_size_max']}; "
                f"KL={latest_update['approx_kl']:.6f}")
            training_reported += 1
        if updated and env.accepted >= next_checkpoint:
            dump_json(args.output/'training-progress.json',training_updates)
            latest_window = env.metric_series[-1] if env.metric_series else None
            partial = {
                'transactions':env.accepted,
                'coinbase_transactions':env.accepted-env.non_coinbase,
                'cross_shard_transactions':env.cross_count,
                'overall_exploration_cross_shard_rate':env.cross_count/env.accepted,
                'latest_complete_window':latest_window,
                'rolling_load_cv':env.metrics['load_cv'],
                'ppo_updates':len(training_updates),
                'complete':False,
            }
            agent.save_model(args.output/f'h2ppo-checkpoint-{env.accepted}.pt',
                             dict(protocol=protocol,training=partial))
            next_checkpoint += 500_000

    training_seconds = time.time()-training_started
    expected_training_rows = args.train_transactions//args.report_interval
    if len(env.metric_series) != expected_training_rows:
        raise AssertionError(
            f'Expected {expected_training_rows} training metric rows, got {len(env.metric_series)}')
    training_windows = list(env.metric_series)
    dump_json(args.output/'training-window-metrics.json',training_windows)
    write_evaluation_csv(args.output/'training-window-metrics.csv',training_windows)
    training_window_cross = [row['interval_cross_shard_rate'] for row in training_windows]
    training_window_cv = [row['interval_protocol_load_cv'] for row in training_windows]
    training_window_balance = [
        row['interval_protocol_load_balance_degree'] for row in training_windows
    ]
    training_controller_cv = [row['interval_load_cv'] for row in training_windows]
    training_snapshot = {
        'transactions':env.accepted,'coinbase_transactions':env.accepted-env.non_coinbase,
        'cross_shard_transactions':env.cross_count,
        'overall_exploration_cross_shard_rate':env.cross_count/env.accepted,
        'last_window_cross_shard_rate':training_window_cross[-1],
        'mean_window_cross_shard_rate':float(np.mean(training_window_cross)),
        'max_window_cross_shard_rate':max(training_window_cross),
        'last_window_load_cv':training_window_cv[-1],
        'mean_window_load_cv':float(np.mean(training_window_cv)),
        'max_window_load_cv':max(training_window_cv),
        'last_window_load_balance_degree':training_window_balance[-1],
        'mean_window_load_balance_degree':float(np.mean(training_window_balance)),
        'min_window_load_balance_degree':min(training_window_balance),
        'mean_window_controller_load_cv':float(np.mean(training_controller_cv)),
        'rolling_load_cv':env.metrics['load_cv'],'seconds':training_seconds,
        'metric_rows':len(training_windows),'ppo_updates':len(training_updates),
    }
    model_path = args.output/f'h2ppo-{args.train_transactions}.pt'
    agent.save_model(model_path,dict(protocol=protocol,training=training_snapshot))
    dump_json(args.output/'training-updates.json',training_updates)
    log(f"training complete in {training_seconds:.1f}s; model saved to {model_path}")

    if args.train_only:
        dump_json(args.output/'summary.json',{
            'protocol':protocol,'training':training_snapshot,
            'evaluation':None,'total_wall_seconds':time.time()-started,
            'model':str(model_path.resolve()),
        })
        log_stream.close()
        env.close()
        return

    agent.placement_guard_cv = args.evaluation_load_guard_cv
    agent.placement_guard_window = guard_window
    agent.placement_cross_target = args.evaluation_cross_target
    for name in ('_guard_environment','_guard_origin','_guard_accepted','_guard_cross','_guard_stats'):
        agent.__dict__.pop(name,None)
    if args.evaluation_from_start:
        env.close()
        env = make_environment(args.evaluation_transactions,args.arrival_rate)
        state = env.reset()
        evaluation_end = args.evaluation_transactions
        log('evaluation reset to transaction 0; evaluating the training prefix independently')
    else:
        evaluation_end = total
    env.begin_metric_reporting(args.report_interval)
    evaluation_started = time.time()
    reported = 0
    while env.accepted < evaluation_end:
        remaining = evaluation_end-env.accepted
        if len(env.arriveTxArrive) > remaining:
            state = env.limit_current_arrivals(remaining)
        state,_,done = agent.interact_batched(
            env,test=True,states=state,chunk_size=args.placement_chunk_size)
        if done and env.accepted < evaluation_end:
            raise RuntimeError(f'Dataset exhausted during evaluation at {env.accepted:,} total transactions')
        while reported < len(env.metric_series):
            row = env.metric_series[reported]
            log(f"evaluation window ending {row['transactions']:,}/{args.evaluation_transactions:,}; "
                f"window cross/all={row['interval_cross_shard_rate']:.6f}; "
                f"window balance={row['interval_protocol_load_balance_degree']:.6f}; "
                f"window CV={row['interval_protocol_load_cv']:.6f}; "
                f"block(mean/min/max)={row['model_block_size_mean']:.1f}/"
                f"{row['model_block_size_min']}/{row['model_block_size_max']}")
            reported += 1

    evaluation_seconds = time.time()-evaluation_started
    expected_rows = args.evaluation_transactions//args.report_interval
    if len(env.metric_series) != expected_rows:
        raise AssertionError(f'Expected {expected_rows} metric rows, got {len(env.metric_series)}')
    dump_json(args.output/'evaluation-metrics.json',env.metric_series)
    write_evaluation_csv(args.output/'evaluation-metrics.csv',env.metric_series)
    final = env.metric_series[-1]
    evaluation_window_cross = [row['interval_cross_shard_rate'] for row in env.metric_series]
    evaluation_window_cv = [row['interval_protocol_load_cv'] for row in env.metric_series]
    evaluation_window_balance = [
        row['interval_protocol_load_balance_degree'] for row in env.metric_series
    ]
    evaluation_controller_cv = [row['interval_load_cv'] for row in env.metric_series]
    summary = {
        'protocol':protocol,'training':training_snapshot,
        'evaluation':{
            'transactions':final['transactions'],
            'coinbase_transactions':final['coinbase_transactions'],
            'cross_shard_transactions':final['cross_shard_transactions'],
            'overall_cross_shard_rate':final['cross_shard_rate'],
            'last_window_cross_shard_rate':evaluation_window_cross[-1],
            'mean_window_cross_shard_rate':float(np.mean(evaluation_window_cross)),
            'max_window_cross_shard_rate':max(evaluation_window_cross),
            'last_window_load_cv':evaluation_window_cv[-1],
            'mean_window_load_cv':float(np.mean(evaluation_window_cv)),
            'max_window_load_cv':max(evaluation_window_cv),
            'last_window_load_balance_degree':evaluation_window_balance[-1],
            'mean_window_load_balance_degree':float(np.mean(evaluation_window_balance)),
            'min_window_load_balance_degree':min(evaluation_window_balance),
            'mean_window_controller_load_cv':float(np.mean(evaluation_controller_cv)),
            'cumulative_load_cv':final['protocol_load_cv'],
            'cumulative_load_balance_degree':final['protocol_load_balance_degree'],
            'cumulative_controller_load_cv':final['load_cv'],
            'cumulative_controller_load_balance_degree':final['load_balance_degree'],
            'metric_rows':len(env.metric_series),'seconds':evaluation_seconds,
            'guard_statistics':getattr(agent,'_guard_stats',{}),
        },
        'total_wall_seconds':time.time()-started,
        'model':str(model_path.resolve()),
    }
    dump_json(args.output/'summary.json',summary)
    log(f"evaluation complete in {evaluation_seconds:.1f}s; "
        f"mean window cross/all={np.mean(evaluation_window_cross):.6f}; "
        f"mean window balance={np.mean(evaluation_window_balance):.6f}; "
        f"mean window CV={np.mean(evaluation_window_cv):.6f}")
    log_stream.close()


if __name__ == '__main__':
    main()
    # Files are closed on successful return. Avoid traversing the complete replay
    # graph during interpreter teardown; never bypass an exception or a failure.
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)

"""H2PPO with conditional control, GAE bootstrapping and masked batches."""
from pathlib import Path
import math
import time
import numpy as np
import torch
from torch.nn import functional as F
from torch.distributions import Categorical,Normal
from algo.networks import Actor,Critic
from algo.rollout import HPPOBuffer


class HPPO:
    def __init__(self,num_states,num_shards,is_update_blocksize=2,lr_a=5e-3,lr_c=1e-4,
                 bmax=6000,bmin=2000,gamma=0.95,lam=0.95,repeat_time=10,batch_size=16,
                 eps_clip=0.2,w_entropy=0.001,*,seed=0,target_kl=None,resource_max=(200.,200.),
                 placement_ratio='factorized',architecture='dense',control_mode='hierarchical',network_resource_budget=None,
                 placement_guard_cv=None,placement_minibatch_size=None,placement_guard_window=None,
                 placement_cross_target=None,placement_temporal_weight=0.,freeze_placement=False,
                 control_architecture='dense',freeze_control=False):
        if not bmin < bmax or min(batch_size,repeat_time) < 1:
            raise ValueError('Invalid PPO parameters')
        torch.manual_seed(seed)
        self.actor = Actor(num_states,num_shards,is_update_blocksize,bmax,bmin,architecture,
                           shared_control=control_mode=='flat',
                           control_architecture=control_architecture)
        self.architecture,self.control_mode = architecture,control_mode
        self.control_architecture = control_architecture
        if control_mode not in ('hierarchical','flat'):
            raise ValueError('control_mode must be hierarchical or flat')
        if freeze_placement and control_mode != 'hierarchical':
            raise ValueError('freeze_placement is only defined for hierarchical control')
        self.freeze_placement = bool(freeze_placement)
        self.freeze_control = bool(freeze_control)
        if self.freeze_placement and self.freeze_control:
            raise ValueError('placement and control policies cannot both be frozen')
        if self.freeze_control and control_mode != 'hierarchical':
            raise ValueError('freeze_control is only defined for hierarchical control')
        if self.freeze_placement:
            # Hierarchical H2PPO has independent encoders for placement and
            # stability control.  Freeze only the high-level placement path so
            # queue-controller retraining preserves the validated crossing/CV
            # policy while control_base and both control heads remain trainable.
            for module in (self.actor.base,self.actor.shard_header,self.actor.prior):
                if module is not None:
                    for parameter in module.parameters():
                        parameter.requires_grad_(False)
        if self.freeze_control:
            for module in (self.actor.control_base,self.actor.control_local,
                           self.actor.switch_header,self.actor.block_header,
                           self.actor.control_action_header):
                if module is not None:
                    for parameter in module.parameters():
                        parameter.requires_grad_(False)
            self.actor.log_std.requires_grad_(False)
        self.critic = Critic(num_states)
        self.optimizer_a = torch.optim.Adam(
            [parameter for parameter in self.actor.parameters() if parameter.requires_grad],lr=lr_a)
        self.optimizer_c = torch.optim.Adam(self.critic.parameters(),lr=lr_c)
        self.buffer = HPPOBuffer()
        self.bmax,self.bmin = bmax,bmin
        self.gamma,self.lam = gamma,lam
        self.repeat_time,self.batch_size = repeat_time,batch_size
        self.eps_clip,self.w_entropy,self.target_kl = eps_clip,w_entropy,target_kl
        self.num_states,self.num_shards = num_states,num_shards
        self.resource_control = self.actor.resource_control
        self.resource_max = tuple(resource_max)
        self.network_resource_budget = None if network_resource_budget is None else tuple(network_resource_budget)
        if self.network_resource_budget is not None:
            budget = np.asarray(self.network_resource_budget)
            if not self.resource_control or budget.shape != (2,) or (budget <= 0).any() or not np.isfinite(budget).all():
                raise ValueError('Invalid network resource budget')
            if (budget > np.asarray(self.resource_max)).any():
                raise ValueError('Per-shard cap must cover each simplex budget')
        if placement_ratio not in ('joint','factorized'):
            raise ValueError('placement_ratio must be joint or factorized')
        self.placement_ratio = placement_ratio
        if placement_guard_cv is not None and (not np.isfinite(placement_guard_cv) or placement_guard_cv <= 0):
            raise ValueError('placement_guard_cv must be a positive finite value')
        self.placement_guard_cv = placement_guard_cv
        if placement_guard_window is not None and (not isinstance(placement_guard_window,(int,np.integer))
                                                   or placement_guard_window < 1):
            raise ValueError('placement_guard_window must be a positive integer')
        self.placement_guard_window = placement_guard_window
        if placement_cross_target is not None and (not np.isfinite(placement_cross_target)
                                                   or not 0 <= placement_cross_target <= 1):
            raise ValueError('placement_cross_target must be between zero and one')
        self.placement_cross_target = placement_cross_target
        if placement_minibatch_size is not None and (not isinstance(placement_minibatch_size,(int,np.integer))
                                                      or placement_minibatch_size < 1):
            raise ValueError('placement_minibatch_size must be a positive integer')
        self.placement_minibatch_size = placement_minibatch_size
        if not np.isfinite(placement_temporal_weight) or placement_temporal_weight < 0:
            raise ValueError('placement_temporal_weight must be nonnegative and finite')
        self.placement_temporal_weight = float(placement_temporal_weight)

    def guard_placement(self,env,tx,action):
        """Project inference onto the configured crossing/load trade-off.

        The projection uses only information available since this controller
        was activated.  It accounts for the transaction and its unique
        parents.  The actor's choice is preserved while it satisfies the load
        cap and the window crossing budget.  When the crossing budget is
        exhausted, a same-shard placement is preferred if one is feasible.
        """
        if self.placement_guard_cv is None and self.placement_cross_target is None:
            return int(action)
        reset_window = (self.placement_guard_window is not None and
                        getattr(self,'_guard_accepted',None) is not None and
                        env.accepted-self._guard_accepted >= self.placement_guard_window)
        new_environment = getattr(self,'_guard_environment',None) is not env
        if new_environment or reset_window:
            self._guard_environment = env
            self._guard_origin = env.workload_totals.copy()
            self._guard_accepted = getattr(env,'accepted',0)
            self._guard_cross = getattr(env,'cross_count',0)
        if new_environment:
            self._guard_stats = dict(decisions=0,actor_overrides=0,
                                     target_infeasible=0,cv_infeasible=0)
        base = (env.workload_totals-self._guard_origin).astype(float)
        parent_sids = [env.TxBelongShard[h][1] for h in env._parent_hashes(tx)]
        for sid in parent_sids:
            base[sid] += 1
        candidates = base[None,:]+np.eye(env.num_shards)
        means = np.maximum(1.,candidates.mean(1))
        cvs = candidates.std(1)/means
        crosses = np.array([any(parent != sid for parent in parent_sids)
                            for sid in range(env.num_shards)])
        feasible = (np.arange(env.num_shards) if self.placement_guard_cv is None else
                    np.flatnonzero(cvs <= self.placement_guard_cv+1e-12))
        if not len(feasible):
            self._guard_stats['cv_infeasible'] += 1
            feasible = np.flatnonzero(np.isclose(cvs,cvs.min(),rtol=0,atol=1e-12))

        target_allows_cross = True
        if self.placement_cross_target is not None:
            position = getattr(env,'accepted',0)-self._guard_accepted+1
            window_cross = getattr(env,'cross_count',0)-self._guard_cross
            target_allows_cross = window_cross+1 <= self.placement_cross_target*position+1e-12

        action = int(action)
        chosen = None
        if action in feasible and (not crosses[action] or target_allows_cross):
            chosen = action
        elif not target_allows_cross:
            local = feasible[~crosses[feasible]]
            if len(local):
                feasible = local
            else:
                self._guard_stats['target_infeasible'] += 1
        if chosen is None:
            best_cv = cvs[feasible].min()
            best = feasible[np.isclose(cvs[feasible],best_cv,rtol=0,atol=1e-12)]
            chosen = int(best[np.argmin(env.partition_sizes[best])])
        self._guard_stats['decisions'] += 1
        self._guard_stats['actor_overrides'] += int(chosen != action)
        return chosen

    def initialize_locality_prior(self,locality_weight=.8,load_weight=.3,temperature=16.,
                                  address_weight=0.):
        """Explicit engineering warm start; all initialized weights remain trainable.

        Only use the augmented 10K state and shared scorer. The prior comes
        from a declared policy, never from held-out validation/test labels.
        """
        self.initialize_paper_state_prior(
            load_weight=load_weight,
            dependency_weight=1-locality_weight,
            neighbor_count_weight=locality_weight,
            neighbor_location_weight=address_weight,
            temperature=temperature,
            reset_control=True,
        )

    def initialize_paper_state_prior(self,*,load_weight,dependency_weight,
                                     neighbor_count_weight,neighbor_location_weight,
                                     temperature=16.,reset_control=False):
        """Initialize the trainable shared scorer from Eqs. (23)-(24) state groups.

        The four coefficients address active-window load, TD(u), Nbr(u), and
        ANbr(u), respectively.  ``reset_control=False`` is used when retaining
        a trained queue/resource controller while recalibrating placement.
        """
        if (self.architecture != 'shared'
                or self.num_states not in (10*self.num_shards,12*self.num_shards)):
            raise ValueError('Locality initialization requires shared architecture and 10K/12K states')
        weights_to_validate=(load_weight,dependency_weight,neighbor_count_weight,
                             neighbor_location_weight)
        if any(not np.isfinite(value) or value < 0 for value in weights_to_validate):
            raise ValueError('Paper-state prior weights must be nonnegative finite values')
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError('Paper-state prior temperature must be positive and finite')
        with torch.no_grad():
            self.actor.shard_header[-1].weight.zero_()
            self.actor.shard_header[-1].bias.zero_()
            weights = self.actor.prior.weight
            weights.zero_()
            # Softmax temperature divides logits: larger temperatures produce
            # a softer warm start and leave useful gradient for PPO.  The old
            # multiplication made the default temperature saturate the policy
            # (for example, -24*17.5=-420 for load), so learned weights could
            # not overcome the engineering prior.
            scale=1.0/temperature
            weights[0,6] = -scale*load_weight
            feature_start=self.num_states//self.num_shards-3
            weights[0,feature_start] = scale*dependency_weight
            weights[0,feature_start+1] = scale*neighbor_count_weight
            weights[0,feature_start+2] = scale*neighbor_location_weight
            # The objective charges no resource price. High available service
            # is a useful initial control; PPO can subsequently change it.
            if reset_control:
                self.actor.switch_header[-1].weight.zero_()
                self.actor.switch_header[-1].bias.copy_(torch.tensor([-2.,2.]))
                if self.actor.control_architecture == 'equivariant':
                    self.actor.control_action_header[-1].weight.zero_()
                    self.actor.control_action_header[-1].bias.zero_()
                    self.actor.control_action_header[-1].bias[0] = 3.
                else:
                    self.actor.block_header[-1].weight.zero_()
                    self.actor.block_header[-1].bias.fill_(3.)

    def add_queue_avoidance_prior(self,weight):
        """Bias the trainable shared scorer away from current queue backlog.

        ``Q(t)`` is paper macro-state group 2 and is already normalized by the
        current per-shard arrival scale.  This is a warm-start coefficient,
        not an inference projection; PPO remains free to update it.
        """
        if (self.architecture != 'shared' or self.actor.prior is None
                or self.num_states not in (10*self.num_shards,12*self.num_shards)):
            raise ValueError('Queue prior requires a shared 10K/12K placement state')
        if not np.isfinite(weight) or weight < 0:
            raise ValueError('Queue prior weight must be nonnegative and finite')
        with torch.no_grad():
            self.actor.prior.weight[0,2] -= float(weight)

    @staticmethod
    def pool(states,mask):
        return (states*mask.unsqueeze(-1)).sum(1)/mask.sum(1).clamp_min(1).unsqueeze(-1)

    @staticmethod
    def block_logprob(dist,raw):
        # Tanh's Jacobian is evaluated at the stored latent action. Affine
        # range constant cancels in the PPO ratio; normalized action entropy.
        log_jac = 2*(math.log(2)-raw-F.softplus(-2*raw))
        return (dist.log_prob(raw)-log_jac).sum(-1)

    def resource_action(self,raw):
        values = raw[...,self.num_shards:].reshape(*raw.shape[:-1],self.num_shards,2)
        if self.network_resource_budget is None:
            return (values.tanh()+1)/2*values.new_tensor(self.resource_max)
        # Add one unallocated-budget component. This is a bijection R^K to
        # the open K-simplex, so allocations ALWAYS respect the network sum.
        logits = torch.cat((values,torch.zeros_like(values[...,:1,:])),-2)
        return logits.softmax(-2)[...,:-1,:]*values.new_tensor(self.network_resource_budget)

    def control_logprob(self,dist,raw):
        if self.network_resource_budget is None:
            return self.block_logprob(dist,raw)
        blocks = raw[...,:self.num_shards]
        jac_blocks = (2*(math.log(2)-blocks-F.softplus(-2*blocks))).sum(-1)
        values = raw[...,self.num_shards:].reshape(*raw.shape[:-1],self.num_shards,2)
        logits = torch.cat((values,torch.zeros_like(values[...,:1,:])),-2)
        # det(J)=prod(all K+1 proportions); constant affine scales cancel.
        jac_resources = logits.log_softmax(-2).sum((-1,-2))
        return dist.log_prob(raw).sum(-1)-jac_blocks-jac_resources

    def select_action(self,states,test=False):
        states = torch.as_tensor(np.asarray(states),dtype=torch.float32)
        if states.ndim != 2 or not len(states) or states.shape[1] != self.num_states:
            raise ValueError(f'Expected nonempty [N,{self.num_states}] observations')
        with torch.no_grad():
            probs,_,_,_ = self.actor(states)
            shard_dist = Categorical(probs)
            shards = probs.argmax(-1) if test else shard_dist.sample()
            _,gate_prob,mu,sigma = self.actor(states.mean(0))
            gate_dist,block_dist = Categorical(gate_prob),Normal(mu,sigma)
            gate = gate_prob.argmax(-1) if test else gate_dist.sample()
            if self.control_mode == 'flat':
                gate = torch.ones_like(gate)
            raw = mu if test else block_dist.sample()
            bounded = raw.tanh()
            sizes = (self.bmax+self.bmin)/2+(self.bmax-self.bmin)/2*bounded[:self.num_shards]
            if self.resource_control:
                resources = self.resource_action(raw)
            if not test:
                a = self.buffer.actor_buffer
                a.actions_shards.append(shards.tolist())
                a.actions_is_update_blocksize.append(gate.item())
                a.actions_blocksize.append(raw.tolist())
                a.logprobs_shards.append(shard_dist.log_prob(shards).tolist())
                a.logprobs_is_update_blocksize.append(gate_dist.log_prob(gate).item())
                a.logprobs_blocksize.append(self.control_logprob(block_dist,raw).item())
                a.control_states.append(states.mean(0).tolist())
                self.buffer.value_states.append(states.mean(0).tolist())
        controls = dict(block_sizes=sizes.tolist(),resources=resources.tolist()) if self.resource_control else sizes.tolist()
        return shards.tolist(),int(gate),controls

    def value(self,states):
        if len(states) == 0:
            return 0.0
        with torch.no_grad():
            return self.critic(torch.as_tensor(np.asarray(states),dtype=torch.float32).mean(0)).item()

    def interact(self,env,test=False):
        """Autoregressive placement and post-placement conditional control.

        The value baseline is stored before ANY action. Every placement keeps
        its own conditional observation. The controller has a distinct state
        after placement, including newly created backlog. Empty-arrival slots
        still train the stability controller.
        """
        initial = env.get_state(env.arriveTxArrive)
        value_state = initial.mean(0) if len(initial) else env.control_state()
        rows,actions,logps = [],[],[]
        low = {}
        def placement(current,tx):
            row = current.get_state([tx],planning=True)[0]
            with torch.no_grad():
                probs = self.actor.placement(torch.as_tensor(row,dtype=torch.float32))
                dist = Categorical(probs)
                action = probs.argmax(-1) if test else dist.sample()
                if self.placement_guard_cv is not None:
                    if not test:
                        raise ValueError('placement_guard_cv is an inference-only safety projection')
                    action = torch.as_tensor(self.guard_placement(current,tx,int(action)))
                rows.append(row)
                actions.append(int(action))
                logps.append(dist.log_prob(action).item())
            return int(action)
        def choose_control(row):
            with torch.no_grad():
                probs,mu,sigma = self.actor.control(torch.as_tensor(row,dtype=torch.float32))
                gd,bd = Categorical(probs),Normal(mu,sigma)
                gate = probs.argmax(-1) if test else gd.sample()
                if self.control_mode == 'flat':
                    gate = torch.ones_like(gate)
                raw = mu if test else bd.sample()
                bounded = raw.tanh()
                sizes = (self.bmax+self.bmin)/2+(self.bmax-self.bmin)/2*bounded[:self.num_shards]
                controls = sizes.tolist()
                if self.resource_control:
                    resources = self.resource_action(raw)
                    controls = dict(block_sizes=controls,resources=resources.tolist())
                low.update(gate=int(gate),raw=raw.tolist(),gate_logp=gd.log_prob(gate).item(),
                           block_logp=self.control_logprob(bd,raw).item(),state=row.tolist())
            return int(gate),controls
        def controller(current):
            return choose_control(current.control_state())
        if self.control_mode == 'flat':
            # Flat PPO samples one simultaneous hybrid action from one shared
            # pre-action representation for the shared policy.
            gate,controls = choose_control(value_state)
            next_state,reward,done = env.step(placement,gate,controls)
        else:
            next_state,reward,done = env.step(placement,control_policy=controller)
        if not test:
            a = self.buffer.actor_buffer
            a.actions_shards.append(actions)
            a.logprobs_shards.append(logps)
            a.actions_is_update_blocksize.append(low['gate'])
            a.actions_blocksize.append(low['raw'])
            a.logprobs_is_update_blocksize.append(low['gate_logp'])
            a.logprobs_blocksize.append(low['block_logp'])
            a.control_states.append(low['state'])
            self.buffer.value_states.append(value_state.tolist())
            self.buffer.states.append(np.asarray(rows,dtype=np.float32).reshape(-1,self.num_states))
            self.buffer.placement_rewards.append(list(env.last_placement_benefits))
            self.buffer.rewards.append(reward)
            self.buffer.is_terminals.append(done and not env.metrics.get('truncated',False))
        return next_state,reward,done

    def interact_batched(self,env,test=False,*,states=None,chunk_size=None):
        """Process one arrival slot with vectorized placement chunks.

        A chunk uses one ledger snapshot and one actor pass. Between chunks,
        earlier actions have already updated dependency and load state. This
        keeps the 8,000 Tx/s experiment practical while avoiding a whole-slot
        stale load signal. The low-level actor is sampled after all placements.
        """
        states = (env.get_state(env.arriveTxArrive) if states is None else
                  np.asarray(states,dtype=np.float32))
        if (states.ndim != 2 or len(states) != len(env.arriveTxArrive)
                or states.shape[1] != self.num_states):
            raise ValueError('Batched interaction requires one valid state per arrival')
        if chunk_size is None:
            chunk_size = max(1,len(states))
        if not isinstance(chunk_size,(int,np.integer)) or chunk_size < 1:
            raise ValueError('chunk_size must be a positive integer')
        value_state = states.mean(0) if len(states) else env.control_state()
        rows,actions,logps = [],[],[]
        low = {}
        self.last_decision = None

        def choose_control(row):
            decision_started = time.perf_counter_ns()
            row_tensor = torch.as_tensor(row,dtype=torch.float32)
            with torch.no_grad():
                gate_prob,mu,sigma = self.actor.control(row_tensor)
                gate_dist,block_dist = Categorical(gate_prob),Normal(mu,sigma)
                gate = gate_prob.argmax(-1) if test else gate_dist.sample()
                if self.control_mode == 'flat':
                    gate = torch.ones_like(gate)
                raw = mu if test else block_dist.sample()
                bounded = raw.tanh()
                sizes = ((self.bmax+self.bmin)/2+
                         (self.bmax-self.bmin)/2*bounded[:self.num_shards])
                controls = sizes.tolist()
                if self.resource_control:
                    resources = self.resource_action(raw)
                    controls = dict(block_sizes=controls,resources=resources.tolist())
                low.update(gate=int(gate),raw=raw.tolist(),controls=controls,
                           gate_logp=gate_dist.log_prob(gate).item(),
                           block_logp=self.control_logprob(block_dist,raw).item(),
                           state=np.asarray(row,dtype=np.float32).tolist())
                # Available at the environment's pre-service boundary.  The
                # deployment benchmark uses this immutable decision to run
                # R-Shard consensus before applying it to the execution state.
                self.last_decision = dict(shards=actions.copy(),gate=int(gate),controls=controls)
            profile = getattr(self,'_overhead_profile',None)
            if profile is not None:
                profile['control_inference_and_decision_ms'] = (
                    profile.get('control_inference_and_decision_ms',0.)+
                    (time.perf_counter_ns()-decision_started)/1e6)
            return int(gate),controls

        cursor = 0
        pending = []
        def placement(current,tx):
            nonlocal cursor,pending
            if not pending:
                upper = min(len(current.arriveTxArrive),cursor+chunk_size)
                # Never compute a child's policy state before an earlier
                # transaction in this same chunk has received its shard. This
                # preserves chronological observability while batching only
                # transactions that are mutually independent in ledger state.
                hashes = set()
                stop = cursor
                for position in range(cursor,upper):
                    candidate = current.arriveTxArrive[position]
                    if any(parent in hashes for parent in current._parent_hashes(candidate)):
                        break
                    hashes.add(candidate.Hash)
                    stop = position+1
                stop = max(cursor+1,stop)
                chunk_rows = current.get_state(current.arriveTxArrive[cursor:stop],planning=True)
                decision_started = time.perf_counter_ns()
                with torch.no_grad():
                    probs = self.actor.placement(torch.as_tensor(chunk_rows,dtype=torch.float32))
                    dist = Categorical(probs)
                    chunk_actions = probs.argmax(-1) if test else dist.sample()
                rows.extend(chunk_rows)
                pending = list(zip(chunk_actions.tolist(),probs.tolist()))[::-1]
                cursor = stop
                profile = getattr(self,'_overhead_profile',None)
                if profile is not None:
                    profile['placement_inference_and_decision_ms'] = (
                        profile.get('placement_inference_and_decision_ms',0.)+
                        (time.perf_counter_ns()-decision_started)/1e6)
            decision_started = time.perf_counter_ns()
            raw_action,probabilities = pending.pop()
            action = self.guard_placement(current,tx,int(raw_action))
            actions.append(action)
            logps.append(float(math.log(max(probabilities[action],1e-30))))
            profile = getattr(self,'_overhead_profile',None)
            if profile is not None:
                profile['placement_inference_and_decision_ms'] = (
                    profile.get('placement_inference_and_decision_ms',0.)+
                    (time.perf_counter_ns()-decision_started)/1e6)
            return action

        if self.control_mode == 'flat':
            gate,controls = choose_control(value_state)
            next_state,reward,done = env.step(placement,gate,controls)
        else:
            next_state,reward,done = env.step(
                placement,control_policy=lambda current: choose_control(current.control_state()))
        if not test:
            actor = self.buffer.actor_buffer
            actor.actions_shards.append(actions)
            actor.logprobs_shards.append(logps)
            actor.actions_is_update_blocksize.append(low['gate'])
            actor.actions_blocksize.append(low['raw'])
            actor.logprobs_is_update_blocksize.append(low['gate_logp'])
            actor.logprobs_blocksize.append(low['block_logp'])
            actor.control_states.append(low['state'])
            self.buffer.value_states.append(value_state.tolist())
            self.buffer.states.append(np.asarray(rows,dtype=np.float32).reshape(-1,self.num_states))
            self.buffer.placement_rewards.append(list(env.last_placement_benefits))
            self.buffer.rewards.append(reward)
            self.buffer.is_terminals.append(done and not env.metrics.get('truncated',False))
        return next_state,reward,done

    def environment_value(self,env):
        states = env.get_state(env.arriveTxArrive)
        return self.value(states if len(states) else env.control_state()[None,:])

    def get_gae(self,values,last_value=0.0):
        rewards,terminals = self.buffer.rewards_tensor,self.buffer.is_terminals_tensor
        advantages = torch.zeros_like(values)
        next_value,next_advantage = float(last_value),0.0
        for i in reversed(range(len(values))):
            mask = 1-terminals[i]
            delta = rewards[i]+self.gamma*mask*next_value-values[i]
            advantages[i] = delta+self.gamma*self.lam*mask*next_advantage
            next_value,next_advantage = values[i],advantages[i]
        returns = advantages+values
        normalized = ((advantages-advantages.mean())/(advantages.std(unbiased=False)+1e-8)
                      if len(advantages) > 1 else advantages)
        return returns,normalized

    def _clip_loss(self,logp,old,adv):
        ratio = (logp-old).clamp(-20,20).exp()
        return -torch.minimum(ratio*adv,ratio.clamp(1-self.eps_clip,1+self.eps_clip)*adv)

    def update(self,last_value=0.0):
        if not len(self.buffer):
            return {}
        b = self.buffer
        b.build_tensor()
        pooled = b.value_states_tensor
        with torch.no_grad():
            values = self.critic(pooled).squeeze(-1)
            targets,advantages = self.get_gae(values,last_value)
            # High-level decisions receive their corresponding Eq. (22)
            # placement benefit. Reusing one slot advantage for thousands of
            # actions destroys transaction-level credit assignment.
            placement_advantages = b.placement_rewards_tensor.clone()
            active_placement = placement_advantages[b.mask]
            if len(active_placement) > 1:
                placement_advantages = ((placement_advantages-active_placement.mean())/
                                        (active_placement.std(unbiased=False)+1e-8))
            else:
                placement_advantages.zero_()
            if self.placement_temporal_weight:
                # Feed the normalized environment GAE into placement credit
                # assignment.  This is the path by which Eq. (22)'s queue
                # drift teaches the high-level actor to avoid future backlog.
                placement_advantages = (placement_advantages+
                    self.placement_temporal_weight*advantages[:,None])
        records = []
        stop = False
        for _ in range(self.repeat_time):
            for ids in torch.randperm(len(b)).split(self.batch_size):
                states,mask = b.states_tensor[ids],b.mask[ids]
                actions = b.actions[ids]
                old_shard_logprob = b.old_shard_logprob[ids]
                placement_adv = placement_advantages[ids]
                if (self.placement_minibatch_size is not None and
                        states.shape[1] > self.placement_minibatch_size):
                    columns = torch.randperm(states.shape[1])[:self.placement_minibatch_size]
                    states,mask = states[:,columns],mask[:,columns]
                    actions = actions[:,columns]
                    old_shard_logprob = old_shard_logprob[:,columns]
                    placement_adv = placement_adv[:,columns]
                probs = self.actor.placement(states)
                gate_prob,mu,sigma = self.actor.control(b.control_states_tensor[ids])
                sd,gd,bd = Categorical(probs),Categorical(gate_prob),Normal(mu,sigma)
                shard_lp = sd.log_prob(actions)
                gate_lp = gd.log_prob(b.gates[ids])
                block_lp = self.control_logprob(bd,b.raw_blocks[ids])
                adv = advantages[ids]
                # Each tx is a discrete placement decision. Padding
                # contributes neither loss nor entropy.
                if self.placement_ratio == 'joint':
                    joint = (shard_lp*mask).sum(1)
                    old_joint = (old_shard_logprob*mask).sum(1)
                    entropy = (sd.entropy()*mask).sum(1)/mask.sum(1).clamp_min(1)
                    active_rows = mask.any(1)
                    shard_loss = ((self._clip_loss(joint,old_joint,adv)-self.w_entropy*entropy)[active_rows].mean()
                                  if active_rows.any() else gate_lp.new_zeros(()))
                else:
                    per_tx = (self._clip_loss(shard_lp,old_shard_logprob,placement_adv)
                              -self.w_entropy*sd.entropy())
                    active_rows = mask.any(1)
                    shard_loss = (((per_tx*mask).sum(1)/mask.sum(1).clamp_min(1))[active_rows].mean()
                                  if active_rows.any() else gate_lp.new_zeros(()))
                gate_loss = (self._clip_loss(gate_lp,b.old_gate_logprob[ids],adv)-self.w_entropy*gd.entropy()).mean()
                active = b.gates[ids].bool()
                block_loss = gate_loss.new_zeros(())
                if active.any():
                    raw_sample = bd.rsample()
                    entropy = -self.control_logprob(bd,raw_sample)
                    block_loss = (self._clip_loss(block_lp[active],b.old_block_logprob[ids][active],adv[active])
                                  -self.w_entropy*entropy[active]).mean()
                if self.control_mode == 'flat':
                    # One flat hybrid action, no switching decision or gate loss.
                    joint = (shard_lp*mask).sum(1)+block_lp
                    old = (old_shard_logprob*mask).sum(1)+b.old_block_logprob[ids]
                    ent = (sd.entropy()*mask).sum(1)/mask.sum(1).clamp_min(1)-self.control_logprob(bd,bd.rsample())
                    loss_a = (self._clip_loss(joint,old,adv)-self.w_entropy*ent).mean()
                else:
                    loss_a = shard_loss+gate_loss+block_loss
                loss_c = F.mse_loss(self.critic(pooled[ids]).squeeze(-1),targets[ids])
                if not torch.isfinite(loss_a+loss_c):
                    raise FloatingPointError('Non-finite PPO loss')
                self.optimizer_a.zero_grad(set_to_none=True)
                loss_a.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(),0.5)
                self.optimizer_a.step()
                self.optimizer_c.zero_grad(set_to_none=True)
                loss_c.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(),0.5)
                self.optimizer_c.step()
                with torch.no_grad():
                    d = shard_lp-old_shard_logprob
                    if self.placement_ratio == 'joint':
                        d = (d*mask).sum(1)
                        kl = (d.clamp(-20,20).exp()-1-d)[mask.any(1)].mean() if mask.any() else d.new_zeros(())
                    else:
                        kl = ((d.clamp(-20,20).exp()-1-d)*mask).sum()/mask.sum().clamp_min(1)
                    dg = gate_lp-b.old_gate_logprob[ids]
                    kl = torch.maximum(kl,(dg.clamp(-20,20).exp()-1-dg).mean())
                    if active.any():
                        dc = block_lp[active]-b.old_block_logprob[ids][active]
                        kl = torch.maximum(kl,(dc.clamp(-20,20).exp()-1-dc).mean())
                records.append((loss_a.item(),loss_c.item(),kl.item()))
                if self.target_kl is not None and kl > self.target_kl:
                    stop = True
                    break
            if stop:
                break
        info = dict(loss_actor=float(np.mean([r[0] for r in records])),
                    loss_critic=float(np.mean([r[1] for r in records])),
                    approx_kl=float(max(r[2] for r in records)),updates=len(records),
                    transactions=int(b.mask.sum()),transitions=len(b))
        b.info = info
        b.init()  # Every update discards old-policy rollouts.
        return info

    def save_model(self,filenames,info=None):
        path = Path(filenames)
        if path.suffix != '.pt':
            path = Path('resultckp'+str(filenames)+'.pt')
        path.parent.mkdir(parents=True,exist_ok=True)
        torch.save(dict(format_version=4,num_states=self.num_states,num_shards=self.num_shards,
                        bmin=self.bmin,bmax=self.bmax,
                        resource_max=self.resource_max,
                        network_resource_budget=self.network_resource_budget,
                        placement_guard_cv=self.placement_guard_cv,
                        placement_guard_window=self.placement_guard_window,
                        placement_cross_target=self.placement_cross_target,
                        freeze_placement=self.freeze_placement,
                        freeze_control=self.freeze_control,
                        placement_temporal_weight=self.placement_temporal_weight,
                        placement_minibatch_size=self.placement_minibatch_size,
                        placement_ratio=self.placement_ratio,architecture=self.architecture,control_mode=self.control_mode,
                        control_architecture=self.control_architecture,
                        gamma=self.gamma,lam=self.lam,target_kl=self.target_kl,
                        actor=self.actor.state_dict(),critic=self.critic.state_dict(),info=info),path)
        return str(path)

    def load_model(self,actor_dict,critic_dict):
        self.actor.load_state_dict(actor_dict)
        self.critic.load_state_dict(critic_dict)

    def load_placement_model(self,actor_dict,critic_dict=None):
        """Load only the placement path from a compatible H2PPO checkpoint."""
        current=self.actor.state_dict()
        prefixes=('base.','shard_header.','prior.')
        selected={key:value for key,value in actor_dict.items()
                  if key.startswith(prefixes) and key in current
                  and current[key].shape == value.shape}
        if not selected:
            raise ValueError('Checkpoint has no compatible placement parameters')
        current.update(selected)
        self.actor.load_state_dict(current)
        if critic_dict is not None:
            self.critic.load_state_dict(critic_dict)

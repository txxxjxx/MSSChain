"""Paper-aligned replay environment; assumptions are documented in README.md."""
from collections import defaultdict, deque
from contextlib import nullcontext
import copy
import csv
from decimal import Decimal
from pathlib import Path
import pickle
import sqlite3
import time
import zlib
import numpy as np
import blockchain.params as p
from blockchain.utils import data2tx,tx2shard
from shard.shard import Shard


class Supervisor:
    def __init__(self, file_paths=None, *, transactions=None, batch_size=2000,
                 window_size=100000, rounds_per_step=1, max_steps=None,
                 alpha=0.5, V=0.5, zeta=0.1, decay_seconds=86400.0,
                 spent_factor=0.8, block_min=2000, block_max=6000,
                 seed=0, output_dir=None, resource_control=True,
                 resource_max=(200.0,200.0), initial_resources=(200.0,200.0),
                 block_interval=8.0, slot_duration=1.0, total_nodes=1024, attack_rate=1e-5,
                 system_fault_fraction=0.25, shard_fault_fraction=0.333333,
                 enhanced_observation=False,stability_observation=False,
                 network_resource_budget=None,resource_weights=(1.,1.),
                 transaction_limit=None,retain_blocks=True,track_latency=True,cross_penalty=0.,
                 load_feature_overload_power=None,address_affinity=False,
                 resource_model='paper_eq10',valid_transaction_ratio=0.8,
                 cpu_tps_per_core=312.5,arrival_process='fixed',
                 arrival_order='csv',arrival_reorder_scan_multiple=8,
                  minimum_admission_fraction=0.95,
                  arrival_candidate_storage='memory',
                 queue_service_mode='protocol',service_capacity_scale=1.0,
                  peak_queue_drift_weight=0.0,use_time_decay=True,
                  use_spent_dependency=True,use_transaction_dependency=True,
                  use_lyapunov_reward=True,
                  normalize_dependency_observation=True):
        if any(not isinstance(x,(int,np.integer)) or x < 1 for x in (batch_size,window_size,rounds_per_step)):
            raise ValueError('batch/window/round counts must be positive')
        if (not 0 <= alpha <= 1 or V <= 0 or decay_seconds <= 0 or not 0 < spent_factor < 1
                or not np.isfinite(cross_penalty) or cross_penalty < 0):
            raise ValueError('invalid placement parameters')
        if not 0 < block_min < block_max:
            raise ValueError('require 0 < block_min < block_max')
        self.file_paths = list(file_paths if file_paths is not None else
                               [p.FileInput1, p.FileInput2, p.FileInput3])
        self.source_transactions = transactions
        if transaction_limit is not None and (not isinstance(transaction_limit,(int,np.integer)) or transaction_limit < 1):
            raise ValueError('transaction_limit must be a positive integer')
        self.transaction_limit = transaction_limit
        self.retain_blocks,self.track_latency = bool(retain_blocks),bool(track_latency)
        self.batch_size, self.window_size = batch_size, window_size
        if arrival_process not in ('fixed','poisson'):
            raise ValueError("arrival_process must be 'fixed' or 'poisson'")
        self.arrival_process = arrival_process
        if arrival_order not in ('csv','admitted_parent_ready','confirmed_parent_ready',
                                  'hybrid_parent_ready'):
            raise ValueError(
                "arrival_order must be 'csv', 'admitted_parent_ready', or "
                "'confirmed_parent_ready', or 'hybrid_parent_ready'")
        if (not isinstance(arrival_reorder_scan_multiple,(int,np.integer))
                or arrival_reorder_scan_multiple < 1):
            raise ValueError('arrival_reorder_scan_multiple must be a positive integer')
        self.arrival_order = arrival_order
        self.arrival_reorder_scan_multiple = int(arrival_reorder_scan_multiple)
        if (not np.isfinite(minimum_admission_fraction)
                or not 0 < minimum_admission_fraction <= 1):
            raise ValueError('minimum_admission_fraction must lie in (0,1]')
        self.minimum_admission_fraction=float(minimum_admission_fraction)
        if arrival_candidate_storage not in ('memory','compressed_memory','sqlite'):
            raise ValueError(
                "arrival_candidate_storage must be memory, compressed_memory, or sqlite")
        self.arrival_candidate_storage=arrival_candidate_storage
        if queue_service_mode not in ('protocol','paper_queue'):
            raise ValueError("queue_service_mode must be 'protocol' or 'paper_queue'")
        if not np.isfinite(service_capacity_scale) or service_capacity_scale <= 0:
            raise ValueError('service_capacity_scale must be positive and finite')
        if (not np.isfinite(peak_queue_drift_weight)
                or peak_queue_drift_weight < 0):
            raise ValueError('peak_queue_drift_weight must be nonnegative and finite')
        self.queue_service_mode = queue_service_mode
        self.service_capacity_scale = float(service_capacity_scale)
        self.peak_queue_drift_weight = float(peak_queue_drift_weight)
        self.use_time_decay = bool(use_time_decay)
        self.use_spent_dependency = bool(use_spent_dependency)
        self.use_transaction_dependency = bool(use_transaction_dependency)
        self.use_lyapunov_reward = bool(use_lyapunov_reward)
        self.normalize_dependency_observation = bool(normalize_dependency_observation)
        self.rounds_per_step, self.max_steps = rounds_per_step, max_steps
        self.alpha, self.V, self.zeta = alpha, V, zeta
        self.cross_penalty = float(cross_penalty)
        if (load_feature_overload_power is not None and
                (not np.isfinite(load_feature_overload_power) or load_feature_overload_power <= 0)):
            raise ValueError('load_feature_overload_power must be positive or None')
        self.load_feature_overload_power = load_feature_overload_power
        self.address_affinity = bool(address_affinity)
        self.decay_seconds, self.spent_factor = decay_seconds, spent_factor
        self.block_min, self.block_max = block_min, block_max
        self.seed, self.output_dir = seed, Path(output_dir) if output_dir else None
        self.num_shards = p.ShardNum
        self.resource_control = bool(resource_control)
        self.enhanced_observation = bool(enhanced_observation)
        self.stability_observation = bool(stability_observation)
        self.resource_max = np.asarray(resource_max,dtype=float)
        self.initial_resources = np.asarray(initial_resources,dtype=float)
        self.network_resource_budget = (None if network_resource_budget is None
                                        else np.asarray(network_resource_budget,dtype=float))
        self.resource_weights = np.asarray(resource_weights,dtype=float)
        if resource_model not in ('huang_eq2','paper_eq10','physical_bottleneck'):
            raise ValueError(
                'resource_model must be huang_eq2, paper_eq10, or physical_bottleneck'
            )
        if (not np.isfinite(valid_transaction_ratio) or
                not 0 < valid_transaction_ratio <= 1):
            raise ValueError('valid_transaction_ratio must lie in (0,1]')
        if not np.isfinite(cpu_tps_per_core) or cpu_tps_per_core <= 0:
            raise ValueError('cpu_tps_per_core must be positive and finite')
        self.resource_model = resource_model
        self.valid_transaction_ratio = float(valid_transaction_ratio)
        self.cpu_tps_per_core = float(cpu_tps_per_core)
        if (self.resource_weights.shape != (2,) or not np.isfinite(self.resource_weights).all()
                or (self.resource_weights <= 0).any()):
            raise ValueError('Resource weights must be two positive finite values')
        if (self.resource_max.shape != (2,) or self.initial_resources.shape != (2,)
                or not np.isfinite(self.resource_max).all() or not np.isfinite(self.initial_resources).all()
                or (self.resource_max <= 0).any() or (self.initial_resources < 0).any()
                or (self.initial_resources > self.resource_max).any()):
            raise ValueError('C4: resource allocations must lie within their per-type budgets')
        self._validate_resource_budget(np.tile(self.initial_resources,(self.num_shards,1)))
        self.block_interval = float(block_interval)
        self.slot_duration = float(slot_duration)
        if (not np.isfinite(self.block_interval) or self.block_interval <= 0
                or not np.isfinite(self.slot_duration) or self.slot_duration <= 0
                or not isinstance(total_nodes,(int,np.integer)) or total_nodes < self.num_shards
                or not np.isfinite(attack_rate) or attack_rate < 0
                or not 0 <= system_fault_fraction < 1 or not 0 < shard_fault_fraction < 1/3):
            raise ValueError('Invalid consensus/security configuration')
        self.security = dict(total_nodes=int(total_nodes),attack_rate=float(attack_rate),
            system_fault_fraction=float(system_fault_fraction),shard_fault_fraction=float(shard_fault_fraction))
        self._validate_security(self.block_interval)
        self.observation_dim = (9 if self.resource_control else 7) * self.num_shards
        self.observation_dim += self.num_shards*int(self.enhanced_observation)
        self.observation_dim += 2*self.num_shards*int(self.stability_observation)
        self._stream = None
        self._arrival_candidate_db = None
        self.init()

    def init(self):
        self.close()
        self.rng = np.random.default_rng(self.seed)
        self.shards = [Shard(sid) for sid in range(self.num_shards)]
        for shard in self.shards:
            shard.genGenesisBlock()
            shard.blocksize = min(self.block_max, max(self.block_min, p.BlockSize))
            shard.resources = self.initial_resources.tolist()
            shard.w = self.resource_weights.tolist()
            shard.blocktime = self.block_interval
            shard.slot_duration = self.slot_duration
            shard.resource_model = self.resource_model
            shard.valid_ratio = self.valid_transaction_ratio
            shard.cpu_tps_per_core = self.cpu_tps_per_core
        k = self.num_shards
        self.blockSize = self.shards[0].blocksize
        self.tx_num, self.load, self.cst, self.arrive, self.q, self.b = [np.zeros(k, dtype=np.int64) for _ in range(6)]
        self.ready_queue = np.zeros(k,dtype=np.int64)
        self.blocked_queue = np.zeros(k,dtype=np.int64)
        self.partition_sizes = np.zeros(k,dtype=np.int64)
        self.confirmed_by_shard = np.zeros(k,dtype=np.int64)
        self.workload_totals = np.zeros(k,dtype=np.int64)
        # Article-specific execution load: originals plus every protocol item
        # processed/generated by the relay-based cross-shard transaction flow.
        self.internal_original_totals = np.zeros(k,dtype=np.int64)
        self.cross_original_totals = np.zeros(k,dtype=np.int64)
        self.cross_subtransaction_totals = np.zeros(k,dtype=np.int64)
        self.protocol_workload_totals = np.zeros(k,dtype=np.int64)
        self.TxBelongShard, self.TxIsProcessed = {}, {}
        self.address_home = {}
        self.UTXOSpent, self.txes, self._td = {}, {}, {}
        self._window = deque()
        self.arriveTxArrive = []
        self._stream = self._read_transactions()
        self.steps = self.clock = self.accepted = self.confirmed = 0
        self.cross_count = self.non_coinbase = 0
        self._submitted_at, self._latencies, self._confirmation_log = {}, [], []
        self.metric_series,self._metric_reporting = [],None
        self._deferred_arrivals = deque()
        self._arrival_candidates = {}
        self._compressed_arrival_candidates={}
        self._compressed_arrival_pending={}
        self._compressed_arrival_waiting=defaultdict(list)
        self._arrival_ready_candidates = deque()
        self._arrival_pending_parents = {}
        self._arrival_waiting_by_parent = defaultdict(set)
        self._newly_admitted_hashes = deque()
        self._newly_confirmed_hashes = deque()
        self._source_exhausted = False
        self.source_transactions_scanned = 0
        self.arrival_reordered_transactions = 0
        self.max_arrival_candidate_pool = 0
        self._candidate_db_pending_candidates=[]
        self._candidate_db_pending_waiting=[]
        self._candidate_db_pending_hashes=set()
        if (self.arrival_order in ('confirmed_parent_ready','hybrid_parent_ready')
                and self.arrival_candidate_storage == 'sqlite'):
            if self.output_dir is None:
                raise ValueError('sqlite candidate storage requires output_dir')
            self.output_dir.mkdir(parents=True,exist_ok=True)
            db_path=self.output_dir/'arrival-candidates.sqlite3'
            db_path.unlink(missing_ok=True)
            self._arrival_candidate_db=sqlite3.connect(db_path)
            self._arrival_candidate_db.executescript('''
                PRAGMA journal_mode=WAL;
                PRAGMA synchronous=NORMAL;
                PRAGMA temp_store=FILE;
                PRAGMA cache_size=-65536;
                CREATE TABLE candidates(
                    hash TEXT PRIMARY KEY,
                    payload BLOB NOT NULL,
                    missing INTEGER NOT NULL
                ) WITHOUT ROWID;
                CREATE INDEX candidates_missing ON candidates(missing);
                CREATE TABLE waiting(
                    parent TEXT NOT NULL,
                    child TEXT NOT NULL,
                    PRIMARY KEY(parent,child)
                ) WITHOUT ROWID;
                CREATE INDEX waiting_child ON waiting(child);
                CREATE TEMP TABLE released_parent(parent TEXT PRIMARY KEY) WITHOUT ROWID;
                CREATE TEMP TABLE released_child(
                    child TEXT PRIMARY KEY,
                    released_count INTEGER NOT NULL
                ) WITHOUT ROWID;
            ''')
        # Runtime-overhead experiments may stop at the R-Shard boundary.  In
        # that mode the controller keeps only hash-indexed placement metadata
        # and per-shard queue counters; transaction bodies stay in the shared
        # mempool and are never copied into W-Shard pools.
        self._rshard_control_queue = None
        self.rshard_decision_ledger = []
        self._eof = self.done = False
        self.metrics = {}

    def _read_transactions(self):
        emitted = 0
        if self.source_transactions is not None:
            for tx in self.source_transactions:
                if self.transaction_limit is not None and emitted >= self.transaction_limit:
                    return
                emitted += 1
                yield copy.deepcopy(tx)
            return
        for path in self.file_paths:
            with open(path, newline='', encoding='utf-8-sig') as stream:
                reader = csv.reader(stream)
                next(reader, None)
                for line, row in enumerate(reader, 2):
                    try:
                        tx, ok = data2tx(row)
                    except (ValueError, IndexError) as exc:
                        raise ValueError(f'{path}:{line}: invalid transaction: {exc}') from exc
                    if ok:
                        if self.transaction_limit is not None and emitted >= self.transaction_limit:
                            return
                        emitted += 1
                        yield tx

    def _next_source_transaction(self):
        tx = next(self._stream)
        self.source_transactions_scanned += 1
        return tx

    def reset(self, episode=0, *, history_size=0, history_policy=None):
        self.init()
        # Historical CSV rows are already-confirmed Bitcoin transactions.
        # Restore their ledger as an identical hash-placed snapshot for every
        # policy. This is only initialization, never an evaluated transition.
        for _ in range(history_size):
            try:
                tx = self._next_source_transaction()
            except StopIteration as exc:
                raise ValueError('History size exceeds data length') from exc
            sid = history_policy(self,tx) if history_policy is not None else tx2shard(tx.Hash)
            if not isinstance(sid,(int,np.integer)) or not 0 <= sid < self.num_shards:
                raise ValueError('History policy returned an invalid shard')
            self._place(tx,sid)
            self.shards[sid].txPools.removeTx(tx.Hash,'')
            for inp in tx.TxIns:
                source = self.TxBelongShard[inp.SenptHash][1]
                self.shards[source].utxoSets.removeOutput(inp.SenptHash,inp.InputAddr,inp.position)
            for out in tx.TxOuts:
                self.shards[sid].utxoSets.add(tx.Hash,out.Addr,out.position,copy.deepcopy(out))
            self.TxIsProcessed[tx.Hash] = True
            self._submitted_at.pop(tx.Hash,None)
            self.confirmed += 1
            self.confirmed_by_shard[sid] += 1
        self.gen_tx()
        return self.get_state(self.arriveTxArrive)

    def gen_tx(self):
        self.arriveTxArrive = []
        arrival_count = (self.batch_size if self.arrival_process == 'fixed'
                         else int(self.rng.poisson(self.batch_size)))
        while self._deferred_arrivals and len(self.arriveTxArrive) < arrival_count:
            self.arriveTxArrive.append(self._deferred_arrivals.popleft())

        if self.arrival_order == 'csv':
            while len(self.arriveTxArrive) < arrival_count:
                try:
                    self.arriveTxArrive.append(self._next_source_transaction())
                except StopIteration:
                    self._source_exhausted = True
                    break
        elif self.arrival_order == 'admitted_parent_ready':
            # Keep the external arrival process at its requested rate while
            # preserving topological admission. Parents may be scheduled
            # earlier in this same batch; their children then enter a W-Shard
            # as blocked items until the parent actually confirms. Requiring
            # confirmation here would scan millions of future Bitcoin rows and
            # incorrectly move that backlog outside the measured system queue.
            newly_admitted=[]
            while self._newly_admitted_hashes:
                newly_admitted.append(self._newly_admitted_hashes.popleft())
            self._release_arrival_candidates_for_parents(newly_admitted)
            scheduled={tx.Hash for tx in self.arriveTxArrive}

            while (self._arrival_ready_candidates
                   and len(self.arriveTxArrive) < arrival_count):
                tx_hash=self._arrival_ready_candidates.popleft()
                tx=self._candidate_pop(tx_hash)
                if tx is not None:
                    self.arriveTxArrive.append(tx)
                    scheduled.add(tx.Hash)
                    self._release_arrival_candidates_for_parents((tx.Hash,))
                    self.arrival_reordered_transactions += 1

            scan_limit=max(1,arrival_count*self.arrival_reorder_scan_multiple)
            source_inspected=0
            while (len(self.arriveTxArrive) < arrival_count
                   and source_inspected < scan_limit and not self._source_exhausted):
                try:
                    tx=self._next_source_transaction()
                except StopIteration:
                    self._source_exhausted=True
                    break
                source_inspected += 1
                missing={parent for parent in self._parent_hashes(tx)
                         if parent not in self.TxIsProcessed and parent not in scheduled}
                if tx.is_coinbase or not missing:
                    self.arriveTxArrive.append(tx)
                    scheduled.add(tx.Hash)
                    self._release_arrival_candidates_for_parents((tx.Hash,))
                else:
                    self._defer_arrival_candidate(tx,missing)

                while (self._arrival_ready_candidates
                       and len(self.arriveTxArrive) < arrival_count):
                    tx_hash=self._arrival_ready_candidates.popleft()
                    ready_tx=self._candidate_pop(tx_hash)
                    if ready_tx is not None:
                        self.arriveTxArrive.append(ready_tx)
                        scheduled.add(ready_tx.Hash)
                        self._release_arrival_candidates_for_parents((ready_tx.Hash,))
                        self.arrival_reordered_transactions += 1

            self.max_arrival_candidate_pool=max(
                self.max_arrival_candidate_pool,self.arrival_candidate_pool_size)
        else:
            # Bitcoin's block order is topological, but replaying thousands of
            # transactions per second compresses many blocks into one slot. A
            # child can consequently reach a W-Shard before a cross-shard
            # parent has completed its request/response/final protocol. Keep
            # such children in the workload-generator pool and backfill the
            # slot with later independent transactions. This changes arrival
            # order only; no transaction is dropped or marked as confirmed.
            self._release_confirmed_arrival_candidates()
            while (self._arrival_ready_candidates
                   and len(self.arriveTxArrive) < arrival_count):
                tx_hash = self._arrival_ready_candidates.popleft()
                tx = self._candidate_pop(tx_hash)
                if tx is not None:
                    self.arriveTxArrive.append(tx)
                    self.arrival_reordered_transactions += 1

            scan_limit = max(1, arrival_count*self.arrival_reorder_scan_multiple)
            source_inspected = 0
            while (len(self.arriveTxArrive) < arrival_count
                   and source_inspected < scan_limit and not self._source_exhausted):
                try:
                    tx = self._next_source_transaction()
                except StopIteration:
                    self._source_exhausted = True
                    break
                source_inspected += 1
                missing = {parent for parent in self._parent_hashes(tx)
                           if not self.TxIsProcessed.get(parent,False)}
                if tx.is_coinbase or not missing:
                    if self.arrival_candidate_pool_size:
                        self.arrival_reordered_transactions += 1
                    self.arriveTxArrive.append(tx)
                else:
                    self._defer_arrival_candidate(tx,missing)

            if self.arrival_order == 'hybrid_parent_ready':
                # Prefer confirmed-parent transactions, but do not let the
                # workload generator silently lower a nominal arrival rate.
                # Once the confirmed-ready scan is exhausted, admit a bounded
                # topological prefix whose parents are already admitted (or
                # precede the child in this same batch). Such children remain
                # blocked in the W-Shard queue until their parents confirm.
                floor=min(arrival_count,int(np.ceil(
                    self.batch_size*self.minimum_admission_fraction)))
                admitted_parents=[]
                while self._newly_admitted_hashes:
                    admitted_parents.append(self._newly_admitted_hashes.popleft())
                admitted_parents.extend(tx.Hash for tx in self.arriveTxArrive)
                self._release_arrival_candidates_for_parents(admitted_parents)
                while self._arrival_ready_candidates and len(self.arriveTxArrive) < floor:
                    tx_hash=self._arrival_ready_candidates.popleft()
                    ready_tx=self._candidate_pop(tx_hash)
                    if ready_tx is None:
                        continue
                    self.arriveTxArrive.append(ready_tx)
                    self._release_arrival_candidates_for_parents((ready_tx.Hash,))
                    self.arrival_reordered_transactions += 1
                if len(self.arriveTxArrive) < floor:
                    raise RuntimeError(
                        'Hybrid dependency replay cannot maintain its minimum '
                        f'admission floor: required {floor}, found '
                        f'{len(self.arriveTxArrive)}')

            self.max_arrival_candidate_pool = max(
                self.max_arrival_candidate_pool, self.arrival_candidate_pool_size)
            if (not self.arriveTxArrive and self._source_exhausted
                    and self.arrival_candidate_pool_size and not self.queue_lengths().any()):
                tx = self.arrival_candidate_samples(1)[0]
                missing = self._candidate_missing(tx.Hash)
                raise RuntimeError(
                    'Dependency-ready arrival replay cannot make progress; '
                    f'{self.arrival_candidate_pool_size} transactions remain and '
                    f'the first has unresolved parents {missing[:3]}')

        if self._arrival_candidate_db is not None:
            self._flush_candidate_db()
            self._arrival_candidate_db.commit()
        self._eof = (self._source_exhausted and not self.arrival_candidate_pool_size
                     and not self._deferred_arrivals)
        return self.arriveTxArrive

    def _parents_confirmed(self, tx):
        return tx.is_coinbase or all(
            self.TxIsProcessed.get(parent,False) for parent in self._parent_hashes(tx))

    def _defer_arrival_candidate(self,tx,missing=None):
        missing = ({parent for parent in self._parent_hashes(tx)
                    if not self.TxIsProcessed.get(parent,False)}
                   if missing is None else set(missing))
        if self._candidate_contains(tx.Hash):
            raise ValueError(f'Duplicate deferred transaction: {tx.Hash}')
        if self.arrival_candidate_storage == 'compressed_memory':
            key=bytes.fromhex(tx.Hash)
            self._compressed_arrival_candidates[key]=zlib.compress(
                pickle.dumps(tx,protocol=5),level=1)
            self._compressed_arrival_pending[key]=len(missing)
            for parent in missing:
                self._compressed_arrival_waiting[bytes.fromhex(parent)].append(key)
        elif self._arrival_candidate_db is not None:
            payload=zlib.compress(pickle.dumps(tx,protocol=5),level=1)
            self._candidate_db_pending_candidates.append(
                (tx.Hash,payload,len(missing)))
            self._candidate_db_pending_waiting.extend(
                (parent,tx.Hash) for parent in missing)
            self._candidate_db_pending_hashes.add(tx.Hash)
            if len(self._candidate_db_pending_candidates) >= 5000:
                self._flush_candidate_db()
        else:
            self._arrival_candidates[tx.Hash] = tx
            self._arrival_pending_parents[tx.Hash] = missing
        if not missing:
            self._arrival_ready_candidates.append(tx.Hash)
        elif self.arrival_candidate_storage == 'memory':
            for parent in missing:
                self._arrival_waiting_by_parent[parent].add(tx.Hash)

    def _release_confirmed_arrival_candidates(self):
        confirmed=[]
        while self._newly_confirmed_hashes:
            confirmed.append(self._newly_confirmed_hashes.popleft())
        self._release_arrival_candidates_for_parents(confirmed)

    def _release_arrival_candidates_for_parents(self,parents):
        if self.arrival_candidate_storage == 'compressed_memory':
            for parent in parents:
                for child in self._compressed_arrival_waiting.pop(
                        bytes.fromhex(parent),()):
                    remaining=self._compressed_arrival_pending.get(child)
                    if remaining is None:
                        continue
                    remaining -= 1
                    self._compressed_arrival_pending[child]=remaining
                    if remaining == 0:
                        self._arrival_ready_candidates.append(child.hex())
            return
        if self._arrival_candidate_db is not None:
            unique=list(dict.fromkeys(parents))
            if not unique:
                return
            self._flush_candidate_db()
            db=self._arrival_candidate_db
            db.execute('DELETE FROM released_parent')
            db.executemany(
                'INSERT OR IGNORE INTO released_parent(parent) VALUES(?)',
                ((parent,) for parent in unique))
            released=list(db.execute('''
                SELECT waiting.child,COUNT(*)
                FROM waiting JOIN released_parent USING(parent)
                GROUP BY waiting.child
            '''))
            db.execute('''
                DELETE FROM waiting
                WHERE parent IN (SELECT parent FROM released_parent)
            ''')
            db.execute('DELETE FROM released_child')
            db.executemany(
                'INSERT INTO released_child(child,released_count) VALUES(?,?)',
                released)
            db.execute('''
                UPDATE candidates
                SET missing=missing-(
                    SELECT released_count FROM released_child
                    WHERE released_child.child=candidates.hash)
                WHERE hash IN (SELECT child FROM released_child)
            ''')
            self._arrival_ready_candidates.extend(
                row[0] for row in db.execute('''
                    SELECT candidates.hash
                    FROM candidates JOIN released_child
                    ON candidates.hash=released_child.child
                    WHERE candidates.missing=0
                '''))
            return
        for parent in parents:
            children = self._arrival_waiting_by_parent.pop(parent,())
            for child_hash in children:
                missing = self._arrival_pending_parents.get(child_hash)
                if missing is None:
                    continue
                missing.discard(parent)
                if not missing:
                    self._arrival_ready_candidates.append(child_hash)

    @property
    def arrival_candidate_pool_size(self):
        if self.arrival_candidate_storage == 'compressed_memory':
            return len(self._compressed_arrival_candidates)
        if self._arrival_candidate_db is not None:
            return len(self._candidate_db_pending_candidates)+int(
                self._arrival_candidate_db.execute(
                    'SELECT COUNT(*) FROM candidates').fetchone()[0])
        return len(self._arrival_candidates)

    def _flush_candidate_db(self):
        if (self._arrival_candidate_db is None
                or not self._candidate_db_pending_candidates):
            return
        self._arrival_candidate_db.executemany(
            'INSERT INTO candidates(hash,payload,missing) VALUES(?,?,?)',
            self._candidate_db_pending_candidates)
        self._arrival_candidate_db.executemany(
            'INSERT INTO waiting(parent,child) VALUES(?,?)',
            self._candidate_db_pending_waiting)
        self._candidate_db_pending_candidates.clear()
        self._candidate_db_pending_waiting.clear()
        self._candidate_db_pending_hashes.clear()

    def _candidate_contains(self,tx_hash):
        if self.arrival_candidate_storage == 'compressed_memory':
            return bytes.fromhex(tx_hash) in self._compressed_arrival_candidates
        if self._arrival_candidate_db is not None:
            if tx_hash in self._candidate_db_pending_hashes:
                return True
            return self._arrival_candidate_db.execute(
                'SELECT 1 FROM candidates WHERE hash=?',(tx_hash,)).fetchone() is not None
        return tx_hash in self._arrival_candidates

    def _candidate_pop(self,tx_hash):
        if self.arrival_candidate_storage == 'compressed_memory':
            key=bytes.fromhex(tx_hash)
            payload=self._compressed_arrival_candidates.pop(key,None)
            self._compressed_arrival_pending.pop(key,None)
            return (None if payload is None else
                    pickle.loads(zlib.decompress(payload)))
        if self._arrival_candidate_db is None:
            self._arrival_pending_parents.pop(tx_hash,None)
            return self._arrival_candidates.pop(tx_hash,None)
        self._flush_candidate_db()
        row=self._arrival_candidate_db.execute(
            'SELECT payload FROM candidates WHERE hash=?',(tx_hash,)).fetchone()
        if row is None:
            return None
        self._arrival_candidate_db.execute(
            'DELETE FROM candidates WHERE hash=?',(tx_hash,))
        return pickle.loads(zlib.decompress(row[0]))

    def _candidate_missing(self,tx_hash):
        if self.arrival_candidate_storage == 'compressed_memory':
            child=bytes.fromhex(tx_hash)
            missing=[]
            for parent,children in self._compressed_arrival_waiting.items():
                if child in children:
                    missing.append(parent.hex())
                    if len(missing) == 20:
                        break
            return missing
        if self._arrival_candidate_db is not None:
            self._flush_candidate_db()
            return [row[0] for row in self._arrival_candidate_db.execute(
                'SELECT parent FROM waiting WHERE child=? LIMIT 20',(tx_hash,))]
        return list(self._arrival_pending_parents.get(tx_hash,()))

    def arrival_candidate_samples(self,limit=20):
        if self.arrival_candidate_storage == 'compressed_memory':
            return [pickle.loads(zlib.decompress(payload)) for payload in
                    list(self._compressed_arrival_candidates.values())[:limit]]
        if self._arrival_candidate_db is None:
            return list(self._arrival_candidates.values())[:limit]
        self._flush_candidate_db()
        return [pickle.loads(zlib.decompress(row[0])) for row in
                self._arrival_candidate_db.execute(
                    'SELECT payload FROM candidates LIMIT ?',(int(limit),))]

    def set_arrival_rate(self,arrival_rate,*,resample_current=True):
        """Change total-system arrivals without skipping pending input data."""
        if not isinstance(arrival_rate,(int,np.integer)) or arrival_rate < 1:
            raise ValueError('Arrival rate must be a positive integer')
        self.batch_size=int(arrival_rate)
        if not resample_current:
            return self.get_state(self.arriveTxArrive)
        # The current batch has only been generated, not admitted. Return it
        # to the front before drawing the new phase's fixed/Poisson count.
        self._deferred_arrivals.extendleft(reversed(self.arriveTxArrive))
        self.arriveTxArrive=[]
        self._eof=False
        self.gen_tx()
        return self.get_state(self.arriveTxArrive)

    def limit_current_arrivals(self,count):
        """Shorten one slot without dropping its remaining transactions."""
        if (not isinstance(count,(int,np.integer)) or count < 0
                or count > len(self.arriveTxArrive)):
            raise ValueError('Arrival limit must lie within the current slot')
        remainder = self.arriveTxArrive[count:]
        self._deferred_arrivals.extendleft(reversed(remainder))
        self.arriveTxArrive = self.arriveTxArrive[:count]
        return self.get_state(self.arriveTxArrive)

    def _parent_hashes(self, tx):
        return list(dict.fromkeys(inp.SenptHash for inp in tx.TxIns))

    def begin_metric_reporting(self,interval=50000):
        if not isinstance(interval,(int,np.integer)) or interval < 1:
            raise ValueError('Metric interval must be a positive integer')
        self.metric_series = []
        self._metric_reporting = dict(interval=int(interval),accepted=self.accepted,
            cross=self.cross_count,non_coinbase=self.non_coinbase,
            workloads=self.workload_totals.copy(),partitions=self.partition_sizes.copy(),
            previous_accepted=self.accepted,previous_cross=self.cross_count,
            previous_workloads=self.workload_totals.copy(),
            protocol_workloads=self.protocol_workload_totals.copy(),
            previous_protocol_workloads=self.protocol_workload_totals.copy(),
            internal_originals=self.internal_original_totals.copy(),
            previous_internal_originals=self.internal_original_totals.copy(),
            cross_originals=self.cross_original_totals.copy(),
            previous_cross_originals=self.cross_original_totals.copy(),
            cross_subtransactions=self.cross_subtransaction_totals.copy(),
            previous_cross_subtransactions=self.cross_subtransaction_totals.copy())

    @staticmethod
    def _balance_metrics(values):
        values = np.asarray(values,dtype=float)
        cv = float(values.std()/max(1.,values.mean()))
        return cv,float(1./(1.+cv))

    def _record_metrics_if_due(self):
        r = self._metric_reporting
        if r is None:
            return
        total = self.accepted-r['accepted']
        if total == 0 or total % r['interval']:
            return
        cross = self.cross_count-r['cross']
        workloads = self.workload_totals-r['workloads']
        interval_workloads = self.workload_totals-r['previous_workloads']
        protocol_workloads=self.protocol_workload_totals-r['protocol_workloads']
        interval_protocol_workloads=(self.protocol_workload_totals-
                                     r['previous_protocol_workloads'])
        interval_internal=(self.internal_original_totals-
                           r['previous_internal_originals'])
        interval_cross_original=(self.cross_original_totals-
                                 r['previous_cross_originals'])
        interval_subtransactions=(self.cross_subtransaction_totals-
                                  r['previous_cross_subtransactions'])
        cumulative_cv,cumulative_balance = self._balance_metrics(workloads)
        interval_cv,interval_balance = self._balance_metrics(interval_workloads)
        protocol_cv,protocol_balance=self._balance_metrics(protocol_workloads)
        interval_protocol_cv,interval_protocol_balance=self._balance_metrics(
            interval_protocol_workloads)
        interval_total = self.accepted-r['previous_accepted']
        interval_cross = self.cross_count-r['previous_cross']
        non_coinbase = self.non_coinbase-r['non_coinbase']
        rolling_cv,rolling_balance = self._balance_metrics(self.load)
        block_sizes = np.asarray([shard.blocksize for shard in self.shards],dtype=int)
        row = dict(transactions=total,coinbase_transactions=total-non_coinbase,
            cross_shard_transactions=cross,cross_shard_rate=cross/total,
            interval_transactions=interval_total,interval_cross_shard_transactions=interval_cross,
            interval_cross_shard_rate=interval_cross/max(1,interval_total),
            workload=workloads.tolist(),load_cv=cumulative_cv,load_balance_degree=cumulative_balance,
            interval_workload=interval_workloads.tolist(),interval_load_cv=interval_cv,
            interval_load_balance_degree=interval_balance,
            protocol_workload=protocol_workloads.tolist(),
            protocol_load_cv=protocol_cv,
            protocol_load_balance_degree=protocol_balance,
            interval_protocol_workload=interval_protocol_workloads.tolist(),
            interval_protocol_load_cv=interval_protocol_cv,
            interval_protocol_load_balance_degree=interval_protocol_balance,
            interval_internal_originals_by_shard=interval_internal.tolist(),
            interval_cross_originals_by_shard=interval_cross_original.tolist(),
            interval_cross_subtransactions_by_shard=interval_subtransactions.tolist(),
            interval_internal_originals_total=int(interval_internal.sum()),
            interval_cross_originals_total=int(interval_cross_original.sum()),
            interval_cross_subtransactions_total=int(interval_subtransactions.sum()),
            rolling_load_cv=rolling_cv,rolling_load_balance_degree=rolling_balance,
            model_block_size_mean=float(block_sizes.mean()),
            model_block_size_min=int(block_sizes.min()),
            model_block_size_max=int(block_sizes.max()),
            model_block_sizes_by_shard=block_sizes.tolist())
        self.metric_series.append(row)
        r['previous_accepted'],r['previous_cross'] = self.accepted,self.cross_count
        r['previous_workloads'] = self.workload_totals.copy()
        r['previous_protocol_workloads']=self.protocol_workload_totals.copy()
        r['previous_internal_originals']=self.internal_original_totals.copy()
        r['previous_cross_originals']=self.cross_original_totals.copy()
        r['previous_cross_subtransactions']=self.cross_subtransaction_totals.copy()

    def dependency(self, tx):
        """Eqs. (1)-(4), direct-parent cache, scalar L1 denominator in Eq. (3).

        Unplaced in-batch parents have no known location; never hash-assign
        them as a side effect of observation. The runner supports sequential
        placement to expose their actual locations before the child's action.
        """
        accumulated = np.zeros(self.num_shards,dtype=np.float32)
        if not self.use_transaction_dependency:
            return accumulated.copy(), accumulated
        parents = self._parent_hashes(tx)
        known = [h for h in parents if h in self._td]
        if known:
            vectors = []
            for h in known:
                vector = self._td[h].copy()
                owner = self.TxBelongShard[h][1]
                # Eq. (4) updates column i only for transactions resident in i.
                vector[owner] /= max(1,self.tx_num[owner])
                vectors.append(vector)
            denominator = sum(float(v.sum()) for v in vectors)
            if denominator > 0:
                for h, vector in zip(known, vectors):
                    age = max(0.0, (tx.TimeStamp-self.txes[h].TimeStamp).total_seconds())
                    consumed = {i.position for i in tx.TxIns if i.SenptHash == h}
                    spent_mask = self.UTXOSpent[h]
                    remaining = any(not (spent_mask >> position) & 1 and position not in consumed
                                    for position in range(len(self.txes[h].TxOuts)))
                    factor = (1.0 if remaining else self.spent_factor) if self.use_spent_dependency else 1.0
                    decay = np.exp(-age/self.decay_seconds) if self.use_time_decay else 1.0
                    accumulated += decay*factor*vector/(len(parents)*denominator)
        return tx.TxOutCount/(self.tx_num+1) + accumulated, accumulated

    def Scost(self, tx):
        return self.dependency(tx)[0].tolist()

    def compute_load_balance(self):
        load = self.objective_load()
        mean = max(1.,float(load.mean()))
        # A raw-count exponential underflows after only tens of transactions
        # and removes the balance signal at paper-scale traffic volumes.
        # Relative deviation is dimensionless and remains informative as the
        # fixed report window fills.
        return np.exp(-self.zeta*np.abs(load/mean-1.))

    def objective_load(self):
        """Workload used by the policy objective.

        When fixed-interval reporting is active, optimize the same independent
        window that will be reported. A rolling window otherwise contains the
        preceding report interval at a boundary and trains against a different
        balance definition.
        """
        reporting = getattr(self,'_metric_reporting',None)
        if reporting is not None:
            return self.workload_totals-reporting['previous_workloads']
        return self.load

    def load_observation(self):
        load = self.objective_load().astype(float)
        relative = load/max(1.,float(load.mean()))
        if self.load_feature_overload_power is None:
            return relative
        return np.maximum(relative-1.,0.)**self.load_feature_overload_power

    def get_state(self, txlist, *, planning=False, dependency_values=None):
        k = self.num_shards
        scale = max(1, self.batch_size/k)
        macro = np.concatenate((self.cst/max(1, self.window_size/k), self.arrive/scale,
                                (self.queue_lengths() if planning else self.q)/scale,
                                np.array([s.blocksize for s in self.shards])/self.block_max))
        if self.resource_control:
            allocations = np.array([s.resources for s in self.shards])/self.resource_max
            # Current allocations are needed to make switch-off transitions
            # observable; they extend the paper's four macro state groups.
            macro = np.concatenate((macro,allocations[:,0],allocations[:,1]))
        if self.enhanced_observation:
            # Match the placement observation to the active balance objective.
            macro = np.concatenate((macro,self.load_observation()))
        if self.stability_observation:
            macro = np.concatenate((macro,self.ready_queue/scale,self.blocked_queue/scale))
        if dependency_values is not None and len(dependency_values) != len(txlist):
            raise ValueError('dependency_values must contain one entry per transaction')
        states = []
        dependency_iter = (iter(dependency_values) if dependency_values is not None else None)
        for tx in txlist:
            parents = self._parent_hashes(tx)
            counts = np.zeros(k)
            if self.use_transaction_dependency:
                for h in parents:
                    if h in self.TxBelongShard:
                        counts[self.TxBelongShard[h][1]] += 1
            td, _ = (next(dependency_iter) if dependency_iter is not None else self.dependency(tx))
            if self.enhanced_observation and self.normalize_dependency_observation:
                # Legacy engineering normalization.  It removes most of the
                # absolute time-decay magnitude for single-parent transactions,
                # so paper-aligned runs disable it explicitly.
                td = td/max(1e-12,td.sum())
            if not self.use_transaction_dependency:
                final_feature = np.zeros(k)
            elif self.address_affinity:
                address_counts = np.zeros(k)
                addresses = ([inp.InputAddr for inp in tx.TxIns]+
                             [out.Addr for out in tx.TxOuts])
                for address in addresses:
                    owner = self.address_home.get(address)
                    if owner is not None:
                        address_counts[owner] += 1
                final_feature = address_counts/max(1.,address_counts.sum())
            else:
                final_feature = counts > 0
            states.append(np.concatenate((macro,td,counts/max(1,len(parents)),final_feature)))
        return np.asarray(states, dtype=np.float32).reshape(-1, self.observation_dim)

    def control_state(self):
        """Full-width state for low-level control, with zero transaction features."""
        k = self.num_shards
        state = np.zeros(self.observation_dim,dtype=np.float32)
        scale = max(1,self.batch_size/k)
        state[:k] = self.cst/max(1,self.window_size/k)
        state[k:2*k] = self.arrive/scale
        state[2*k:3*k] = self.queue_lengths()/scale
        state[3*k:4*k] = np.array([s.blocksize for s in self.shards])/self.block_max
        if self.resource_control:
            allocation = np.array([s.resources for s in self.shards])/self.resource_max
            state[4*k:5*k],state[5*k:6*k] = allocation[:,0],allocation[:,1]
        if self.enhanced_observation:
            start = (6 if self.resource_control else 4)*k
            state[start:start+k] = self.load_observation()
        if self.stability_observation:
            start = ((6 if self.resource_control else 4)+int(self.enhanced_observation))*k
            state[start:start+k] = self.ready_queue/scale
            state[start+k:start+2*k] = self.blocked_queue/scale
        return state

    def _validate_admission(self, tx):
        if tx.Hash in self.TxBelongShard:
            raise ValueError(f'Duplicate transaction: {tx.Hash}')
        if (not isinstance(tx.Size,(int,np.integer)) or tx.Size <= 0
                or tx.TxInCount != len(tx.TxIns) or tx.TxOutCount != len(tx.TxOuts)
                or tx.is_coinbase != (len(tx.TxIns) == 0)):
            raise ValueError(f'Malformed transaction structure: {tx.Hash}')
        if [out.position for out in tx.TxOuts] != list(range(len(tx.TxOuts))):
            raise ValueError(f'Non-canonical output positions: {tx.Hash}')
        seen, total = set(), Decimal(0)
        for inp in tx.TxIns:
            key = (inp.SenptHash, inp.position)
            if key in seen:
                raise ValueError(f'Duplicate input in {tx.Hash}: {key}')
            seen.add(key)
            if inp.SenptHash not in self.txes:
                raise ValueError(f'Missing/unordered parent {inp.SenptHash}; replay its history first')
            parent = self.txes[inp.SenptHash]
            if not 0 <= inp.position < len(parent.TxOuts):
                raise ValueError(f'Invalid output index: {key}')
            output = parent.TxOuts[inp.position]
            if output.Addr != inp.InputAddr or (self.UTXOSpent[inp.SenptHash] >> inp.position) & 1:
                raise ValueError(f'Invalid or double-spent input: {key}')
            total += Decimal(str(output.Value))
        values = [Decimal(str(out.Value)) for out in tx.TxOuts]
        if any(not v.is_finite() or v < 0 for v in values):
            raise ValueError('Invalid transaction output value')
        if not tx.is_coinbase and sum(values,Decimal(0)) > total+Decimal('0.00000001'):
            raise ValueError(f'Transaction creates value: {tx.Hash}')

    def _place(self, tx, sid):
        profile=getattr(self,'_overhead_profile',None)
        placement_started=time.perf_counter_ns() if profile is not None else None
        phase_started=placement_started
        self._validate_admission(tx)
        if profile is not None:
            now=time.perf_counter_ns()
            profile['admission_validation_ms']=profile.get('admission_validation_ms',0.)+(now-phase_started)/1e6
            phase_started=now
        td, accumulated = self.dependency(tx)
        if profile is not None:
            now=time.perf_counter_ns()
            profile['dependency_cache_lookup_ms']=profile.get('dependency_cache_lookup_ms',0.)+(now-phase_started)/1e6
            phase_started=now
        benefit = self.alpha*td[sid]+(1-self.alpha)*self.compute_load_balance()[sid]
        parent_sids = [self.TxBelongShard[h][1] for h in self._parent_hashes(tx)]
        cross = any(s != sid for s in parent_sids)
        remote_parent_shards=set(parent_sids)-{sid}
        benefit -= self.cross_penalty*int(cross)
        self.cross_count += int(cross)
        self.non_coinbase += int(not tx.is_coinbase)
        if profile is not None:
            now=time.perf_counter_ns()
            profile['placement_scoring_ms']=profile.get('placement_scoring_ms',0.)+(now-phase_started)/1e6
            phase_started=now
        for inp in tx.TxIns:
            self.UTXOSpent[inp.SenptHash] |= 1 << inp.position
        # One integer stores all output-spent flags. This materially reduces
        # memory for the seven-million-transaction experiment.
        self.UTXOSpent[tx.Hash] = 0
        self.txes[tx.Hash] = tx
        self.TxBelongShard[tx.Hash] = (tx, sid)
        for out in tx.TxOuts:
            self.address_home.setdefault(out.Addr,sid)
        self.TxIsProcessed[tx.Hash] = False
        if self.arrival_order in ('admitted_parent_ready','hybrid_parent_ready'):
            self._newly_admitted_hashes.append(tx.Hash)
        if profile is not None:
            now=time.perf_counter_ns()
            profile['ledger_update_ms']=profile.get('ledger_update_ms',0.)+(now-phase_started)/1e6
            phase_started=now
        # Reserve input outpoints at admission; publish output UTXOs only on commit.
        self.shards[sid].txPools.addTx(tx.Hash, '', copy.deepcopy(tx))
        immediately_ready=(tx.is_coinbase or all(
            self.TxIsProcessed.get(inp.SenptHash,False)
            or self.TxBelongShard[inp.SenptHash][1] == sid for inp in tx.TxIns))
        if immediately_ready:
            self.ready_queue[sid] += 1
        else:
            self.blocked_queue[sid] += 1
        if self.track_latency:
            self._submitted_at[tx.Hash] = self.clock
        if profile is not None:
            now=time.perf_counter_ns()
            profile['queue_update_ms']=profile.get('queue_update_ms',0.)+(now-phase_started)/1e6
            phase_started=now
        self.accepted += 1
        counts = np.zeros(self.num_shards, dtype=np.int64)
        cross_counts = np.zeros_like(counts)
        counts[sid] += 1
        for parent_sid in parent_sids:  # Eq. (6), distinct parents including local ones.
            counts[parent_sid] += 1
        if cross:
            cross_counts[sid] += 1
            for parent_sid in set(parent_sids)-{sid}:
                cross_counts[parent_sid] += 1
        self.tx_num[sid] += 1
        self.partition_sizes[sid] += 1
        self.load += counts
        self.workload_totals += counts
        protocol_counts=np.zeros(self.num_shards,dtype=np.int64)
        if not cross:
            self.internal_original_totals[sid] += 1
            protocol_counts[sid] += 1
        else:
            self.cross_original_totals[sid] += 1
            protocol_counts[sid] += 1
            # Origin: m requests + m returned responses + one final tx.
            origin_subtransactions=2*len(remote_parent_shards)+1
            self.cross_subtransaction_totals[sid] += origin_subtransactions
            protocol_counts[sid] += origin_subtransactions
            # Each remote shard processes its grouped request/response item.
            for remote_sid in remote_parent_shards:
                self.cross_subtransaction_totals[remote_sid] += 1
                protocol_counts[remote_sid] += 1
        self.protocol_workload_totals += protocol_counts
        self.cst += cross_counts
        if profile is not None:
            now=time.perf_counter_ns()
            profile['placement_accounting_ms']=profile.get('placement_accounting_ms',0.)+(now-phase_started)/1e6
            phase_started=now
        stored = accumulated.copy()
        stored[sid] += tx.TxOutCount/self.tx_num[sid]
        # Lazy Eq. (4): normalize only the resident shard's component.
        self._td[tx.Hash] = stored.astype(np.float32,copy=True)
        self._td[tx.Hash][sid] *= max(1,self.tx_num[sid])
        if profile is not None:
            now=time.perf_counter_ns()
            profile['dependency_cache_update_ms']=profile.get('dependency_cache_update_ms',0.)+(now-phase_started)/1e6
            phase_started=now
        self._window.append((sid, counts, cross_counts))
        if len(self._window) > self.window_size:
            old_sid, old_counts, old_cross = self._window.popleft()
            self.tx_num[old_sid] -= 1
            self.load -= old_counts
            self.cst -= old_cross
        self._record_metrics_if_due()
        if profile is not None:
            now=time.perf_counter_ns()
            profile['placement_metrics_update_ms']=profile.get('placement_metrics_update_ms',0.)+(now-phase_started)/1e6
            profile['placement_application_ms']=profile.get('placement_application_ms',0.)+(now-placement_started)/1e6
        return benefit

    def queue_lengths(self):
        if self._rshard_control_queue is not None:
            return self._rshard_control_queue.copy()
        return np.array([len(s.txPools)+len(s.crossTxPool.OriginalTxsQueue)
                         for s in self.shards], dtype=np.int64)

    def enable_rshard_control_mode(self):
        """Use lightweight R-Shard metadata instead of executing W-Shards.

        The method is called after history restoration.  Existing queue state
        is retained as integer counters, while future commits update only the
        decision ledger, dependency metadata and aggregate shard queues.
        """
        if self._rshard_control_queue is None:
            self._rshard_control_queue = np.asarray(self.q, dtype=np.int64).copy()
        self.ready_queue = self._rshard_control_queue.copy()
        self.blocked_queue = np.zeros(self.num_shards, dtype=np.int64)

    def commit_rshard_control_decision(self, transactions, placements, gate, controls,
                                       dependency_updates, *, proposal_sha256,
                                       memory_stage=None):
        """Commit a hash-only R-Shard decision without running W-Shard work.

        ``transactions`` are references into the shared mempool.  They are used
        only to update the controller's dependency metadata; no transaction is
        copied into a shard pool and no W-Shard ledger or protocol queue is
        executed.  The returned timings are mutually exclusive phase totals.
        """
        if self._rshard_control_queue is None:
            raise RuntimeError('enable_rshard_control_mode() must be called first')
        if len(transactions) != len(placements):
            raise ValueError('one R-Shard placement is required per transaction hash')
        placements = np.asarray(placements, dtype=np.int64)
        if ((placements < 0).any() or (placements >= self.num_shards).any()):
            raise ValueError('invalid R-Shard placement')

        profile = {}
        started = time.perf_counter_ns()

        stage = memory_stage or (lambda _name: nullcontext())

        # The decision ledger stores hash ownership and a compact round record.
        # Transaction objects are shared-mempool references, not proposal data
        # or W-Shard copies.
        with stage('rshard_decision_ledger_update'):
            parent_lists = []
            for tx, sid in zip(transactions, placements):
                parents = self._parent_hashes(tx)
                parent_lists.append(parents)
                for inp in tx.TxIns:
                    if inp.SenptHash in self.UTXOSpent:
                        self.UTXOSpent[inp.SenptHash] |= 1 << inp.position
                self.UTXOSpent[tx.Hash] = 0
                self.txes[tx.Hash] = tx
                self.TxBelongShard[tx.Hash] = (tx, int(sid))
                self.TxIsProcessed[tx.Hash] = False
                for out in tx.TxOuts:
                    self.address_home.setdefault(out.Addr, int(sid))
            self.rshard_decision_ledger.append({
                'proposal_sha256': str(proposal_sha256),
                'transactions': len(transactions),
                'gate': int(gate),
            })
        checkpoint = time.perf_counter_ns()
        profile['decision_ledger_update_ms'] = (checkpoint - started) / 1e6

        # Persist the dependency vectors computed during feature extraction and
        # update the aggregate placement/load statistics required by the next
        # controller decision.
        phase_started = checkpoint
        with stage('rshard_dependency_statistics_update'):
            arrivals = np.zeros(self.num_shards, dtype=np.int64)
            for tx, sid_value, parents in zip(transactions, placements, parent_lists):
                sid = int(sid_value)
                parent_sids = [self.TxBelongShard[h][1] for h in parents
                               if h in self.TxBelongShard]
                remote_parent_shards = set(parent_sids) - {sid}
                cross = bool(remote_parent_shards)
                self.accepted += 1
                self.non_coinbase += int(not tx.is_coinbase)
                self.cross_count += int(cross)
                self.tx_num[sid] += 1
                self.partition_sizes[sid] += 1
                arrivals[sid] += 1

                counts = np.zeros(self.num_shards, dtype=np.int64)
                cross_counts = np.zeros_like(counts)
                counts[sid] += 1
                for parent_sid in parent_sids:
                    counts[parent_sid] += 1
                if cross:
                    cross_counts[sid] += 1
                    for parent_sid in remote_parent_shards:
                        cross_counts[parent_sid] += 1
                self.load += counts
                self.workload_totals += counts
                self.cst += cross_counts

                protocol_counts = np.zeros(self.num_shards, dtype=np.int64)
                if cross:
                    self.cross_original_totals[sid] += 1
                    protocol_counts[sid] += 2 * len(remote_parent_shards) + 2
                    for remote_sid in remote_parent_shards:
                        self.cross_subtransaction_totals[remote_sid] += 1
                        protocol_counts[remote_sid] += 1
                    self.cross_subtransaction_totals[sid] += 2 * len(remote_parent_shards) + 1
                else:
                    self.internal_original_totals[sid] += 1
                    protocol_counts[sid] += 1
                self.protocol_workload_totals += protocol_counts

                accumulated = dependency_updates.get(tx.Hash)
                if accumulated is None:
                    accumulated = np.zeros(self.num_shards, dtype=np.float32)
                stored = np.asarray(accumulated, dtype=np.float32).copy()
                stored[sid] += tx.TxOutCount / max(1, self.tx_num[sid])
                self._td[tx.Hash] = stored
                self._td[tx.Hash][sid] *= max(1, self.tx_num[sid])

                self._window.append((sid, counts, cross_counts))
                if len(self._window) > self.window_size:
                    old_sid, old_counts, old_cross = self._window.popleft()
                    self.tx_num[old_sid] -= 1
                    self.load -= old_counts
                    self.cst -= old_cross
                self._record_metrics_if_due()
        checkpoint = time.perf_counter_ns()
        profile['dependency_and_statistics_update_ms'] = (checkpoint - phase_started) / 1e6

        # Apply only resource metadata and the scalar queue conservation model.
        # Transaction execution and W-Shard queue mutation remain outside this
        # measurement boundary.
        phase_started = checkpoint
        with stage('rshard_resource_queue_update'):
            if int(gate):
                sizes = np.clip(np.asarray(controls['block_sizes'], dtype=float),
                                self.block_min, self.block_max).astype(int)
                allocations = np.asarray(controls['resources'], dtype=float)
                if sizes.shape != (self.num_shards,) or allocations.shape != (self.num_shards, 2):
                    raise ValueError('invalid R-Shard resource decision shape')
                self._validate_resource_budget(allocations)
                for shard, size, allocation in zip(self.shards, sizes, allocations):
                    shard.blocksize = int(size)
                    shard.resources = allocation.tolist()
                self.blockSize = int(sizes.mean())

            before = self._rshard_control_queue.copy()
            available = before + arrivals
            service = np.zeros(self.num_shards, dtype=np.int64)
            for sid, shard in enumerate(self.shards):
                capacity = min(
                    float(shard.blocksize),
                    shard.compute_resource(shard.resources, types=0) * self.slot_duration,
                )
                service[sid] = min(int(available[sid]),
                                   max(0, int(capacity * self.service_capacity_scale)))
            self._rshard_control_queue = available - service
            self.q = self._rshard_control_queue.copy()
            self.arrive = arrivals
            self.b = service
            self.ready_queue = self._rshard_control_queue.copy()
            self.blocked_queue = np.zeros(self.num_shards, dtype=np.int64)
            self.confirmed += int(service.sum())
            self.confirmed_by_shard += service
            self.clock += 1
            self.steps += 1
            self.metrics = dict(
                steps=self.steps, accepted=self.accepted, confirmed=self.confirmed,
                cross_transactions=self.cross_count, non_coinbase=self.non_coinbase,
                cross_ratio=self.cross_count / max(1, self.non_coinbase),
                load_cv=float(self.load.std() / max(1, self.load.mean())),
                queue_total=int(self.q.sum()), queue_max=int(self.q.max(initial=0)),
                resource_allocations=[s.resources[:] for s in self.shards],
                block_sizes=[s.blocksize for s in self.shards], rounds=self.clock,
            )
        checkpoint = time.perf_counter_ns()
        profile['resource_and_queue_update_ms'] = (checkpoint - phase_started) / 1e6
        profile['total_ms'] = (checkpoint - started) / 1e6
        profile['queue_total_after'] = int(self.q.sum())
        profile['queue_max_after'] = int(self.q.max(initial=0))
        return profile

    def _refresh_queue_readiness(self):
        """Classify current protocol queues once per slot for the next decision."""
        ready=np.zeros(self.num_shards,dtype=np.int64)
        blocked=np.zeros(self.num_shards,dtype=np.int64)
        for sid,shard in enumerate(self.shards):
            locally_ready=set()
            for tx in shard.txPools.TxsQueue.values():
                if tx.OrigTxHash:
                    ready[sid] += 1
                    continue
                parents=self._parent_hashes(tx)
                can_serve=all(self.TxIsProcessed.get(parent,False)
                              or parent in locally_ready for parent in parents)
                if can_serve:
                    ready[sid] += 1
                    if tx.is_coinbase or all(
                            self.TxBelongShard[parent][1] == sid for parent in parents):
                        locally_ready.add(tx.Hash)
                else:
                    blocked[sid] += 1
            blocked[sid] += len(shard.crossTxPool.OriginalTxsQueue)
        if not np.array_equal(ready+blocked,self.queue_lengths()):
            raise AssertionError('Queue readiness classification does not conserve queue items')
        self.ready_queue,self.blocked_queue=ready,blocked

    def tx_handing(self):
        if self.queue_service_mode == 'paper_queue':
            return self._paper_queue_handing()
        routed = np.zeros(self.num_shards, dtype=np.int64)
        for _ in range(self.rounds_per_step):
            # Simultaneous proposals, then commits, then next-round messages.
            blocks = [s.createProposeBlock(self.clock,self.TxIsProcessed,self.TxBelongShard)[0] for s in self.shards]
            for sid, (shard, block) in enumerate(zip(self.shards, blocks)):
                shard.blockchain.add(block)
                shard.blockchain.LatestBlock = block.Hash
                shard.processBlock(block.Transactions,sid,self.TxBelongShard,self.TxIsProcessed)
            for sid, (shard, block) in enumerate(zip(self.shards, blocks)):
                for tx in block.Transactions:
                    kind = shard.whatAmI(tx,sid,self.TxBelongShard)
                    target = None
                    if kind == 'crosstx':
                        target = self.TxBelongShard[tx.TxIns[0].SenptHash][1]
                    elif kind == 'crosstxresponse_C_in':
                        target = self.TxBelongShard[tx.OrigTxHash][1]
                    if target is not None:
                        self.shards[target].txPools.addTx(tx.Hash,tx.OrigTxHash,copy.deepcopy(tx))
                        routed[target] += 1
                    if kind in ('normal','finaltransaction'):
                        h = tx.OrigTxHash if kind == 'finaltransaction' else tx.Hash
                        self._newly_confirmed_hashes.append(h)
                        if self.track_latency and h in self._submitted_at:
                            latency = self.clock+1-self._submitted_at.pop(h)
                            self._latencies.append(latency)
                            self._confirmation_log.append((h,latency,self.clock+1))
                        self.confirmed += 1
                        self.confirmed_by_shard[self.TxBelongShard[h][1]] += 1
            if not self.retain_blocks:
                for shard in self.shards:
                    shard.blockchain.Blocks.clear()
            self.clock += 1
        return routed

    def _paper_queue_handing(self):
        """Serve the transaction-count queues used by Eqs. (10)-(11).

        The Lyapunov model defines B_i(t) as transactions removed from shard
        queue i in one slot.  It does not expand one original transaction into
        the request/response messages of the prototype cross-shard protocol.
        Keeping this service mode explicit prevents protocol-message latency
        from being counted a second time as unavailable Eq. (10) capacity.
        """
        routed = np.zeros(self.num_shards,dtype=np.int64)
        for sid,shard in enumerate(self.shards):
            raw_capacity = min(
                float(shard.blocksize),
                shard.compute_resource(shard.resources,types=0)*self.slot_duration,
            )
            service_limit = max(0,int(raw_capacity*self.service_capacity_scale))
            selected = list(shard.txPools.TxsQueue.values())[:service_limit]
            for tx in selected:
                shard.txPools.removeTx(tx.Hash,tx.OrigTxHash)
                self.TxIsProcessed[tx.Hash] = True
                self._newly_confirmed_hashes.append(tx.Hash)
                if self.track_latency and tx.Hash in self._submitted_at:
                    latency=self.clock+1-self._submitted_at.pop(tx.Hash)
                    self._latencies.append(latency)
                    self._confirmation_log.append((tx.Hash,latency,self.clock+1))
                self.confirmed += 1
                self.confirmed_by_shard[sid] += 1
            served=len(selected)
            cpu,bw=shard.compute_resource(
                shard.resources,
                served/(max(self.slot_duration,1e-12)*self.service_capacity_scale),
                types=1,
            )
            shard.cpu_needs += cpu
            shard.bw_needs += bw
            shard.last_reserved_bytes=served*shard.transaction_size/p.BYTE
        self.clock += 1
        return routed

    def step(self, shards, is_update_blocksize=0, blocksize=None, episode=0, *, control_policy=None):
        if self.done:
            raise RuntimeError('Episode ended; call reset before step')
        # A callable performs sequential placement within one time slot,
        # making earlier in-batch decisions observable without future leakage.
        sequential = callable(shards)
        if not sequential and len(shards) != len(self.arriveTxArrive):
            raise ValueError('One shard action is required per transaction')
        if not sequential and any(not isinstance(s,(int,np.integer)) or not 0 <= s < self.num_shards for s in shards):
            raise ValueError('Invalid shard action')
        if is_update_blocksize not in (0,1):
            raise ValueError('Switch must be binary')
        controls = blocksize if isinstance(blocksize,dict) else {}
        sizes_arg = controls.get('block_sizes',self.blockSize) if controls else blocksize
        sizes = np.asarray(sizes_arg if sizes_arg is not None else self.blockSize,dtype=float)
        if sizes.ndim == 0:
            sizes = np.repeat(sizes,self.num_shards)
        if sizes.shape != (self.num_shards,) or not np.isfinite(sizes).all():
            raise ValueError('Block size must be a finite scalar or K-vector')
        allocations = np.asarray(controls.get('resources',[s.resources for s in self.shards]),dtype=float)
        if (allocations.shape != (self.num_shards,2) or not np.isfinite(allocations).all()
                or (allocations < 0).any() or (allocations > self.resource_max).any()):
            raise ValueError('C4: resources must have shape [K,2] and satisfy per-type budgets')
        self._validate_resource_budget(allocations)
        self._validate_security(self.block_interval)
        q_before = self.queue_lengths()
        benefit = 0.0
        self.last_placement_benefits = []
        arrivals = np.zeros(self.num_shards,dtype=np.int64)
        for i,tx in enumerate(self.arriveTxArrive):
            sid = shards(self,tx) if sequential else shards[i]
            if not isinstance(sid,(int,np.integer)) or not 0 <= sid < self.num_shards:
                raise ValueError('Invalid shard action')
            placement_benefit = self._place(tx,int(sid))
            benefit += placement_benefit
            self.last_placement_benefits.append(float(placement_benefit))
            arrivals[sid] += 1
        if control_policy is not None:
            # The stability actor must observe A(t), not the preceding slot's
            # arrival vector. queue_lengths already contains these admissions.
            self.arrive = arrivals.copy()
            is_update_blocksize,controls = control_policy(self)
            if is_update_blocksize not in (0,1):
                raise ValueError('Switch must be binary')
            if isinstance(controls,dict):
                sizes = np.asarray(controls['block_sizes'],dtype=float)
                allocations = np.asarray(controls['resources'],dtype=float)
            else:
                sizes = np.asarray(controls,dtype=float)
            if sizes.shape != (self.num_shards,) or not np.isfinite(sizes).all():
                raise ValueError('Invalid low-level block-size action')
            if (allocations.shape != (self.num_shards,2) or not np.isfinite(allocations).all()
                    or (allocations < 0).any() or (allocations > self.resource_max).any()):
                raise ValueError('C4: low-level resources violate budgets')
            self._validate_resource_budget(allocations)
        if is_update_blocksize:
            profile=getattr(self,'_overhead_profile',None)
            resource_started=time.perf_counter_ns() if profile is not None else None
            sizes = np.clip(sizes,self.block_min,self.block_max).astype(int)
            for shard,size,allocation in zip(self.shards,sizes,allocations):
                shard.blocksize = int(size)
                shard.resources = allocation.tolist()
            self.blockSize = int(sizes.mean())
            if profile is not None:
                profile['resource_application_ms']=profile.get('resource_application_ms',0.)+(time.perf_counter_ns()-resource_started)/1e6
        arrivals += self.tx_handing()
        q_after = self.queue_lengths()
        self._refresh_queue_readiness()
        service = q_before+arrivals-q_after
        if np.any(service < 0):
            raise AssertionError('Queue conservation violated')
        delta = arrivals-service
        drift = float(np.sum(2.0*q_before*delta+delta.astype(float)**2))
        # The sum term controls total backlog.  The worst-shard term gives the
        # controller a direct learning signal for the shard that determines
        # the recovery tail after a burst.  This affects training reward only;
        # inference actions are never overridden.
        peak_queue_drift = float(q_after.max(initial=0)**2-q_before.max(initial=0)**2)
        augmented_drift = drift+self.peak_queue_drift_weight*peak_queue_drift
        # Text above Eq. (22): NEGATIVE of P3. One fixed positive rescaling.
        reward_drift = augmented_drift if self.use_lyapunov_reward else 0.0
        reward = (self.V*benefit-reward_drift)/max(1,self.batch_size**2)
        self.q,self.arrive,self.b = q_after,arrivals,service
        self.steps += 1
        self.gen_tx()
        truncated = self.max_steps is not None and self.steps >= self.max_steps
        self.done = (self._eof and not self.arriveTxArrive and not q_after.any()) or truncated
        self.metrics = dict(steps=self.steps,accepted=self.accepted,confirmed=self.confirmed,
            cross_transactions=self.cross_count,non_coinbase=self.non_coinbase,
            cross_ratio=self.cross_count/max(1,self.non_coinbase),
            load_cv=float(self.load.std()/max(1,self.load.mean())),queue_total=int(q_after.sum()),
            queue_max=int(q_after.max()),mean_latency_rounds=float(np.mean(self._latencies)) if self._latencies else None,
            reward=reward,drift=drift,peak_queue_drift=peak_queue_drift,
            augmented_drift=augmented_drift,reward_drift=reward_drift,benefit=benefit,
            truncated=bool(truncated),exhausted=bool(self._eof),rounds=self.clock)
        self.metrics['resource_allocations'] = [s.resources[:] for s in self.shards]
        self.metrics['block_sizes'] = [s.blocksize for s in self.shards]
        return self.get_state(self.arriveTxArrive),float(reward),self.done

    def close(self):
        if self._stream is not None and hasattr(self._stream,'close'):
            self._stream.close()
        if getattr(self,'_arrival_candidate_db',None) is not None:
            self._flush_candidate_db()
            self._arrival_candidate_db.commit()
            self._arrival_candidate_db.close()
            self._arrival_candidate_db=None

    def _validate_security(self,interval):
        """Enforce the numeric C5 constraint, not a claim of protocol security."""
        s = self.security
        lhs = s['attack_rate']*s['total_nodes']*s['system_fault_fraction']*interval
        rhs = (s['total_nodes']//self.num_shards)*s['shard_fault_fraction']
        if lhs > rhs:
            raise ValueError(f'C5 violated: attack exposure {lhs} exceeds shard tolerance {rhs}')

    def _validate_resource_budget(self,allocations):
        budget = self.network_resource_budget
        if budget is not None:
            if budget.shape != (2,) or not np.isfinite(budget).all() or (budget <= 0).any():
                raise ValueError('Network resource budget must contain two positive finite values')
            if (allocations.sum(0) > budget+1e-6).any():
                raise ValueError('C4: allocations exceed the total network resource budget')

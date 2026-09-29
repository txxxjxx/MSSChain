"""Padded on-policy rollouts; no truncation of transactions."""
import torch


class ActorBuffer:
    def __init__(self):
        for name in ('actions_shards','actions_is_update_blocksize','actions_blocksize',
                     'logprobs_shards','logprobs_is_update_blocksize','logprobs_blocksize','control_states'):
            setattr(self,name,[])
        self.info = {}


class HPPOBuffer:
    def __init__(self):
        self.states,self.rewards,self.is_terminals,self.value_states,self.placement_rewards = [],[],[],[],[]
        self.actor_buffer = ActorBuffer()
        self.info = {}

    def __len__(self):
        return len(self.states)

    def build_tensor(self):
        lengths = [len(s) for s in self.states]
        if not lengths:
            raise ValueError('Rollout is empty')
        t,n = len(lengths),max(1,max(lengths))
        d = len(self.value_states[0]) if self.value_states else len(next(s for s in self.states if len(s))[0])
        a = self.actor_buffer
        if any(len(getattr(a,name)) != t for name in vars(a) if name != 'info'):
            raise ValueError('Actions and transitions must have equal lengths')
        if len(self.rewards) != t or len(self.is_terminals) != t:
            raise ValueError('Incomplete rollout transitions')
        if self.placement_rewards and len(self.placement_rewards) != t:
            raise ValueError('Incomplete placement rewards')
        if self.value_states and len(self.value_states) != t:
            raise ValueError('Missing pre-action value state')
        self.states_tensor = torch.zeros(t,n,d)
        self.mask = torch.zeros(t,n,dtype=torch.bool)
        self.actions = torch.zeros(t,n,dtype=torch.long)
        self.old_shard_logprob = torch.zeros(t,n)
        self.placement_rewards_tensor = torch.zeros(t,n)
        for i,length in enumerate(lengths):
            if len(a.actions_shards[i]) != length or len(a.logprobs_shards[i]) != length:
                raise ValueError('Action count differs from state count')
            if length:
                self.states_tensor[i,:length] = torch.as_tensor(self.states[i],dtype=torch.float32)
            self.mask[i,:length] = True
            self.actions[i,:length] = torch.as_tensor(a.actions_shards[i],dtype=torch.long)
            self.old_shard_logprob[i,:length] = torch.as_tensor(a.logprobs_shards[i])
            placement = (self.placement_rewards[i] if len(self.placement_rewards) == t
                         else [self.rewards[i]]*length)
            if len(placement) != length:
                raise ValueError('Placement reward count differs from state count')
            self.placement_rewards_tensor[i,:length] = torch.as_tensor(placement,dtype=torch.float32)
        self.gates = torch.tensor(a.actions_is_update_blocksize,dtype=torch.long)
        self.raw_blocks = torch.tensor(a.actions_blocksize,dtype=torch.float32)
        self.old_gate_logprob = torch.tensor(a.logprobs_is_update_blocksize,dtype=torch.float32)
        self.old_block_logprob = torch.tensor(a.logprobs_blocksize,dtype=torch.float32)
        self.rewards_tensor = torch.tensor(self.rewards,dtype=torch.float32)
        self.is_terminals_tensor = torch.tensor(self.is_terminals,dtype=torch.float32)
        self.control_states_tensor = torch.tensor(a.control_states,dtype=torch.float32)
        self.value_states_tensor = (torch.tensor(self.value_states,dtype=torch.float32) if self.value_states else
            (self.states_tensor*self.mask.unsqueeze(-1)).sum(1)/self.mask.sum(1).clamp_min(1).unsqueeze(-1))

    def init(self):
        info = self.info.copy()
        for name in ('states_tensor','mask','actions','old_shard_logprob','gates','raw_blocks',
                     'old_gate_logprob','old_block_logprob','rewards_tensor',
                     'is_terminals_tensor','control_states_tensor','value_states_tensor',
                     'placement_rewards_tensor'):
            self.__dict__.pop(name,None)
        self.__init__()
        self.info = info

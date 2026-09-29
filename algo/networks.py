"""Independent placement/control actors and one pooled slot-value critic."""
import torch
from torch import nn


def mlp(n):
    return nn.Sequential(nn.Linear(n,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU())


class Actor(nn.Module):
    def __init__(self,num_states,num_shards,is_update_block,bmax,bmin,architecture='dense',
                 shared_control=False,control_architecture='dense'):
        super().__init__()
        self.resource_control = num_states in (9*num_shards,10*num_shards,11*num_shards,12*num_shards)
        self.num_shards = num_shards
        self.architecture = architecture
        self.shared_control = bool(shared_control)
        self.control_architecture = control_architecture
        if architecture not in ('dense','shared'):
            raise ValueError('architecture must be dense or shared')
        if control_architecture not in ('dense','equivariant'):
            raise ValueError('control_architecture must be dense or equivariant')
        if self.shared_control and control_architecture != 'dense':
            raise ValueError('equivariant control is only defined for hierarchical H2PPO')
        self.macro_dim = num_states-3*num_shards
        if self.macro_dim % num_shards:
            raise ValueError('Macro state must contain complete per-shard feature groups')
        self.macro_groups = self.macro_dim//num_shards
        controls = (3 if self.resource_control else 1)*num_shards
        self.base = mlp(num_states if architecture == 'dense' else 3*num_states//num_shards)
        self.shard_header = nn.Sequential(nn.Linear(128,64),nn.ReLU(),nn.Linear(64,num_shards if architecture=='dense' else 1))
        self.prior = nn.Linear(num_states//num_shards,1,bias=False) if architecture == 'shared' else None
        self.control_base = (None if self.shared_control or control_architecture == 'equivariant'
                             else mlp(self.macro_dim))
        self.control_local = (mlp(3*self.macro_groups)
                              if control_architecture == 'equivariant' else None)
        # A shared shard scorer consumes one shard plus population summaries,
        # while Flat PPO's simultaneous block/resource action consumes the
        # complete pre-action state.  Keep the placement encoder equivariant
        # and add a full-state encoder only for that joint control action.
        self.joint_control_base = (
            mlp(num_states) if self.shared_control and architecture == 'shared' else None
        )
        self.switch_header = nn.Sequential(nn.Linear(128,64),nn.ReLU(),nn.Linear(64,is_update_block))
        self.block_header = (None if control_architecture == 'equivariant' else
                             nn.Sequential(nn.Linear(128,64),nn.ReLU(),nn.Linear(64,controls)))
        self.control_action_header = (
            nn.Sequential(nn.Linear(128,64),nn.ReLU(),nn.Linear(64,3 if self.resource_control else 1))
            if control_architecture == 'equivariant' else None
        )
        self.log_std = nn.Parameter(torch.full((controls,),-0.7))
        for module in self.modules():
            if isinstance(module,nn.Linear):
                nn.init.orthogonal_(module.weight,gain=2**0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for head in (self.shard_header,self.switch_header,self.block_header,
                     self.control_action_header):
            if head is None:
                continue
            nn.init.orthogonal_(head[-1].weight,gain=0.01)
        if self.control_architecture == 'equivariant':
            # Start from maximum blocks and equal resource logits.  PPO then
            # learns only state-dependent departures from this stable point.
            with torch.no_grad():
                self.switch_header[-1].bias.copy_(torch.tensor([-2.,2.]))
                self.control_action_header[-1].bias.zero_()
                self.control_action_header[-1].bias[0] = 3.
        if self.prior is not None:
            nn.init.zeros_(self.prior.weight)

    def forward(self,x):
        placement = self.placement(x)
        gate,mu,sigma = self.control(x)
        return placement,gate,mu,sigma

    def placement(self,x):
        if self.architecture == 'shared':
            # Shared shard scorer: permuting shard labels permutes probabilities.
            local = x.reshape(*x.shape[:-1],-1,self.num_shards).transpose(-1,-2)
            mean = local.mean(-2,keepdim=True).expand_as(local)
            maximum = local.amax(-2,keepdim=True).expand_as(local)
            logits = self.shard_header(self.base(torch.cat((local,mean,maximum),-1)))+self.prior(local)
            return logits.squeeze(-1).softmax(-1)
        return self.shard_header(self.base(x)).softmax(-1)

    def control(self,x):
        if self.control_architecture == 'equivariant':
            macro=x[...,:self.macro_dim]
            local=macro.reshape(*macro.shape[:-1],self.macro_groups,self.num_shards).transpose(-1,-2)
            mean=local.mean(-2,keepdim=True).expand_as(local)
            maximum=local.amax(-2,keepdim=True).expand_as(local)
            encoded=self.control_local(torch.cat((local,mean,maximum),-1))
            gate=self.switch_header(encoded.mean(-2)).softmax(-1)
            per_shard=self.control_action_header(encoded)
            blocks=per_shard[...,0]
            if self.resource_control:
                mu=torch.cat((blocks,per_shard[...,1:].reshape(*blocks.shape[:-1],-1)),-1)
            else:
                mu=blocks
            sigma=self.log_std.clamp(-5,1).exp().expand_as(mu)
            return gate,mu,sigma
        if self.shared_control:
            control = self.base(x) if self.joint_control_base is None else self.joint_control_base(x)
        else:
            control = self.control_base(x[...,:self.macro_dim])
        gate = self.switch_header(control).softmax(-1)
        mu = self.block_header(control)
        sigma = self.log_std.clamp(-5,1).exp().expand_as(mu)
        return gate,mu,sigma


class Critic(nn.Module):
    def __init__(self,num_states):
        super().__init__()
        self.net = nn.Sequential(mlp(num_states),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))

    def forward(self,x):
        return self.net(x)

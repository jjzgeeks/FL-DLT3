import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import math
from .initialize import *

class PolicyNetworkBase(nn.Module):
    """ Base network class for policy function """
    def __init__(self, state_space, action_space, action_range):
        super(PolicyNetworkBase, self).__init__()
        self._state_space = state_space
        self._state_shape = state_space.shape
        if len(self._state_shape) == 1:
            self._state_dim = self._state_shape[0]
        else:  # high-dim state
            pass  
        self._action_space = action_space
        self._action_shape = action_space.shape
        if len(self._action_shape) < 1:  # Discrete space
            self._action_dim = action_space.n
        else:
            self._action_dim = self._action_shape[0]
        self.action_range = action_range

    def forward(self):
        pass
    
    def evaluate(self):
        pass 
    
    def get_action(self):
        pass

    def sample_action(self,):
        a=torch.FloatTensor(self._action_dim).uniform_(-1, 1)
        return self.action_range*a.numpy()

class DPG_PolicyNetwork(PolicyNetworkBase):
    """
    Deterministic policy gradient network
    """
    def __init__(self, state_space, action_space, hidden_dim, action_range=1., init_w=3e-3):
        super().__init__(state_space, action_space, action_range)
        
        self.linear1 = nn.Linear(self._state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, self._action_dim) # output dim = dim of action

        # weights initialization
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
    

    def forward(self, state):
        activation=F.relu
        x = activation(self.linear1(state)) 
        x = activation(self.linear2(x))
        x = F.tanh(self.linear3(x)).clone() # need clone to prevent in-place operation (which cause gradients not be drived)
        # x = self.linear3(x) # for simplicity, no restriction on action range

        return x

    def evaluate(self, state, noise_scale=0.0):
        '''
        evaluate action within GPU graph, for gradients flowing through it, noise_scale controllable
        '''
        normal = Normal(0, 1)
        action = self.forward(state)
        noise = noise_scale * normal.sample(action.shape).cuda()
        action = self.action_range*action+noise
        return action


    def get_action(self, state, noise_scale=1.0):
        '''
        select action for sampling, no gradients flow, noisy action, return .cpu
        '''
        state = torch.FloatTensor(state).unsqueeze(0).cuda() # state dim: (N, dim of state)
        normal = Normal(0, 1)
        action = self.forward(state)
        noise = noise_scale * normal.sample(action.shape).cuda()
        action=self.action_range*action + noise
        return action.detach().cpu().numpy()[0]

    def sample_action(self):
        normal = Normal(0, 1)
        random_action=self.action_range*normal.sample( (self._action_dim,) )

        return random_action.cpu().numpy()

class DPG_PolicyNetworkLSTM(PolicyNetworkBase):
    """
    Deterministic policy gradient network with LSTM structure.
    The network follows two-branch structure as in paper:  Sim-to-Real Transfer of Robotic Control with Dynamics Randomization
    """
    def __init__(self, state_space, action_space, hidden_dim, action_range=1., init_w=3e-3):
        super().__init__(state_space, action_space, action_range)
        self.hidden_dim = hidden_dim

        self.linear1 = nn.Linear(self._state_dim, hidden_dim)
        self.linear2 = nn.Linear(self._state_dim+self._action_dim, hidden_dim)
        self.lstm1 = nn.LSTM(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(2*hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, self._action_dim) # output dim = dim of action

        # weights initialization
        self.linear4.weight.data.uniform_(-init_w, init_w)
        self.linear4.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, last_action, hidden_in):
        """ 
        state shape: (batch_size, sequence_length, state_dim)
        output shape: (batch_size, sequence_length, action_dim)
        for lstm needs to be permuted as: (sequence_length, batch_size, -1)
        """
        state = state.permute(1,0,2)
        last_action = last_action.permute(1,0,2)
        activation=F.relu
        # branch 1
        fc_branch = activation(self.linear1(state)) 
        # branch 2
        lstm_branch = torch.cat([state, last_action], -1)
        lstm_branch = activation(self.linear2(lstm_branch))   # lstm_branch: sequential data
        # hidden only for initialization, later on hidden states are passed automatically for sequential data
        lstm_branch,  lstm_hidden = self.lstm1(lstm_branch, hidden_in)    # no activation after lstm
        # merged
        merged_branch=torch.cat([fc_branch, lstm_branch], -1)   
        x = activation(self.linear3(merged_branch))
        x = F.tanh(self.linear4(x))
        x = x.permute(1,0,2)  # permute back
        return x, lstm_hidden    # lstm_hidden is actually tuple: (hidden, cell)

    def evaluate(self, state, last_action, hidden_in, noise_scale=0.0):
        '''
        evaluate action within GPU graph, for gradients flowing through it, noise_scale controllable
        '''
        normal = Normal(0, 1)
        action, hidden_out = self.forward(state, last_action, hidden_in)
        noise = noise_scale * normal.sample(action.shape).cuda()
        action = self.action_range*action+noise
        return action, hidden_out

    def get_action(self, state, last_action, hidden_in,  noise_scale=1.0):
        '''
        select action for sampling, no gradients flow, noisy action, return .cpu
        '''
        state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).cuda() # increase 2 dims to match with training data
        last_action = torch.FloatTensor(last_action).unsqueeze(0).unsqueeze(0).cuda()
        normal = Normal(0, 1)
        action, hidden_out = self.forward(state, last_action, hidden_in)
        noise = noise_scale * normal.sample(action.shape).cuda()
        action=self.action_range*action + noise
        return action.detach().cpu().numpy()[0][0], hidden_out

    def sample_action(self):
        normal = Normal(0, 1)
        random_action=self.action_range*normal.sample( (self._action_dim,) )
        return random_action.cpu().numpy()




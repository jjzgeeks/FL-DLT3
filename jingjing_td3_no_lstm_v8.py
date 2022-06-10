#  Part of the code comes from https://github.com/quantumiracle/Popular-RL-Algorithms
import math
import random
# import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

from IPython.display import clear_output
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import display
from reacher import Reacher
from itertools import chain

import gym.spaces as spaces  # Env
import argparse
import scipy.io as scio
import time

torch.manual_seed(1234)  # Reproducibility
GPU = True
device_idx = 0
if GPU:
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(device)

parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=False)
args = parser.parse_args()  # Namespace(test=False, train=False)


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch)) # stack for each element
        ''' 
        the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;
        zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;
        the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
        np.stack((1,2)) => array([1, 2])
        '''
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, init_w=3e-3):
        super(ValueNetwork, self).__init__()
        
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, 1)
        # weights initialization
        self.linear4.weight.data.uniform_(-init_w, init_w)
        self.linear4.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x
        
        
class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(QNetwork, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, 1)
        
        self.linear4.weight.data.uniform_(-init_w, init_w)
        self.linear4.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1) # the dim 0 is number of samples
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x
        
        
class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, action_range=1., init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)
        
        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

        self.action_range = action_range
        self.num_actions = num_actions

        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))

        mean  = F.tanh(self.mean_linear(x))
        # mean = F.leaky_relu(self.mean_linear(x))
        # mean = torch.clamp(mean, -30, 30)

        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max) # clip the log_std into reasonable range
        
        return mean, log_std
    
    def evaluate(self, state, deterministic, eval_noise_scale, epsilon=1e-6):
        '''
        generate action with state as input wrt the policy network, for calculating gradients
        '''
        mean, log_std = self.forward(state)
        std = log_std.exp() # no clip in evaluation, clip affects gradients flow
        
        normal = Normal(0, 1)
        z      = normal.sample() 
        action_0 = torch.tanh(mean + std*z.to(device)) # TanhNormal distribution as actions; reparameterization trick
        action = self.action_range*mean if deterministic else self.action_range*action_0
        log_prob = Normal(mean, std).log_prob(mean+ std*z.to(device)) - torch.log(1. - action_0.pow(2) + epsilon) -  np.log(self.action_range)
        # both dims of normal.log_prob and -log(1-a**2) are (N,dim_of_action); 
        # the Normal.log_prob outputs the same dim of input features instead of 1 dim probability, 
        # needs sum up across the features dim to get 1 dim prob; or else use Multivariate Normal.
        log_prob = log_prob.sum(dim=1, keepdim=True)
        ''' add noise '''
        eval_noise_clip = 2*eval_noise_scale
        noise = normal.sample(action.shape) * eval_noise_scale
        noise = torch.clamp(
        noise,
        -eval_noise_clip,
        eval_noise_clip)
        action = action + noise.to(device)

        return action, log_prob, z, mean, log_std
        
    
    def get_action(self, state, deterministic, explore_noise_scale):
        '''
        generate action for interaction with env
        '''
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(0, 1)
        z      = normal.sample().to(device)
        
        action = mean.detach().cpu().numpy()[0] if deterministic else torch.tanh(mean + std*z).detach().cpu().numpy()[0]

        ''' add noise '''
        noise = normal.sample(action.shape) * explore_noise_scale
        action = self.action_range*action + noise.numpy()

        return action


    def sample_action(self,):
        a=torch.FloatTensor(self.num_actions).uniform_(0, 10.0)
        return self.action_range*a.numpy()




####################################################################################################
n_index = 80
class my_environment:
    def __init__(self,
                 data,
                 ):
        data_set = list(data['All_clients_dataset_info'])  ### time varying 20000 Rows, 10 Columns
        self.data_set = np.array(data_set)
        # print(isinstance(self.data_set, tuple))
        self.bandwidth = np.array(data['All_clients_bandwidth_info'])  ### time varying 20000 Rows, 10 Columns
        self.trans_power = np.array(
            data['All_clients_transmission_power_info'])  ### time varying 20000 Rows, 10 Columns
        self.energy_harvesting = np.array(data['energy_harvesting'])  ## time varying
        self.server_trans_power = 100 * self.trans_power  # unit: Watt
        self.cpu_freq = np.array(data['f'])  # unit: Hz
        self.max_episodes, self.K = self.data_set.shape
        self.max_trans_power = 1.50
        self.min_trans_power = 1.0e-01  # unit: Watt
        self.upload_data_size = 5.0e4 * np.ones(self.K)  # unit: bit
        self.download_data_size = 1.0e4 * np.ones(self.K)
        self.cpu_cycles = 20  # unit cost of each data sample   cycles/bit
        self.channel_gain = np.array(data['G'])  ##time varying
        self.Gaussian_noise = 1.0e-08 * np.ones(self.K),
        self.mu = 4.2e-9

        self.action_low = np.concatenate([np.zeros(self.K), self.min_trans_power * np.ones(self.K)], axis=0)
        self.action_high = np.concatenate([np.ones(self.K), self.max_trans_power * np.ones(self.K)], axis=0)
        self.observation_low = np.concatenate( [2.0e6 * np.ones(self.K), 1.0e4 * np.ones(self.K), 1.0e-3 * np.ones(self.K), 1.0 * np.ones(self.K), 1.0 * np.ones(self.K)], axis=0)
        self.observation_high = np.concatenate([1.0e8 * np.ones(self.K), 1.0e5 * np.ones(self.K), 1.0e-1 * np.ones(self.K), 1.0e3 * np.ones(self.K), 50 * np.ones(self.K)], axis=0)
        self.action_space = spaces.Box(low=self.action_low, high=self.action_high)
        self.observation_space = spaces.Box(low=self.observation_low, high=self.observation_high)

    def step(self, action, t):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        #print(action)
        device_select = action[:self.K]
        device_select[device_select <= 0.5] = 0
        device_select[device_select > 0.5] = 1
        upload_trans_power = action[self.K: len(action)]
        number_of_local_epochs = 4
        capacitance_coefficient = 1.0e-28
        energy = 8.0e2 * np.ones(self.K)
        energy_cmp = number_of_local_epochs * capacitance_coefficient * self.cpu_cycles * self.data_set[t, :] * (self.cpu_freq ** 2)
        #print(energy_cmp)
        energy_up = (upload_trans_power * self.upload_data_size) / self.bandwidth[t, :] * np.log2(1 + (upload_trans_power * self.channel_gain[t, :]) / (self.Gaussian_noise * self.bandwidth[t, :]))
        #print(energy_up)
        energy_c = 1.0e-3 * (energy_cmp + energy_up)  # Unit: KJ
        energy = energy - device_select * energy_c + self.energy_harvesting[t, :]
        energy = list(chain(*energy))
        energy = np.array(energy)  # Unit: kJ
        time_train = number_of_local_epochs * self.cpu_cycles * self.data_set[t, :] / self.cpu_freq
        time_down = self.download_data_size / self.bandwidth[t, :] * np.log2(
            1 + (self.server_trans_power[t, :] * self.channel_gain[t, :]) / (
                        self.Gaussian_noise * self.bandwidth[t, :]))
        time_up = self.upload_data_size / self.bandwidth[t, :] * np.log2(
            1 + (upload_trans_power * self.channel_gain[t, :]) / (self.Gaussian_noise * self.bandwidth[t, :]))
        time_delay = time_train + time_down + time_up
        time_delay = list(chain(*time_delay))
        time_delay = np.array(time_delay)
        # print(time_delay)
        # self.channel_gain = list(chain(*self.channel_gain))
        next_state = np.concatenate(
            [self.data_set[t, :], self.bandwidth[t, :], self.channel_gain[t, :], energy, time_delay], axis=0)

        device_select = np.array([device_select])
        final_accuracy = np.log(1 + self.mu * np.dot(device_select, (self.data_set[t, :]).T))  ##  a real number
        #print(final_accuracy)
        final_energy_cmp = np.dot(device_select, energy_c.T)  ##  a real number
        #print(final_energy_cmp)
        epsilon = 1.0e-9
        eta  = 1.0
        penalty = epsilon * np.mean(device_select * self.data_set[t,:] - eta * sum(self.data_set[t,:]))
        reward = final_accuracy / final_energy_cmp + penalty
        #print(reward) # log2

        if all(energy) == 0:
            done = 1
        else:
            done = 0
        next_state = list(np.array(next_state).flatten())
        next_state = np.array(next_state)
        return final_accuracy, final_energy_cmp, next_state, penalty,  reward, device_select, done

    def reset(self):
        energy_initial = 8.0e2 * np.ones(self.K)
        time_delay = np.zeros(self.K)
        initial_state = np.concatenate(
            [self.data_set[0, :], self.bandwidth[0, :], self.channel_gain[0, :], energy_initial, time_delay], axis=0)
        return initial_state


class TD3_Trainer():
    def __init__(self, replay_buffer, hidden_dim, action_range, policy_target_update_interval=1):
        self.replay_buffer = replay_buffer

        self.q_net1 = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.q_net2 = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_q_net1 = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_q_net2 = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, action_range).to(device)
        self.target_policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, action_range).to(device)
        print('Q Network (1,2): ', self.q_net1)
        print('Policy Network: ', self.policy_net)

        self.target_q_net1 = self.target_ini(self.q_net1, self.target_q_net1)
        self.target_q_net2 = self.target_ini(self.q_net2, self.target_q_net2)
        self.target_policy_net = self.target_ini(self.policy_net, self.target_policy_net)

        q_lr = 3e-4
        policy_lr = 3e-4
        self.update_cnt = 0
        self.policy_target_update_interval = policy_target_update_interval

        self.q_optimizer1 = optim.Adam(self.q_net1.parameters(), lr=q_lr)
        self.q_optimizer2 = optim.Adam(self.q_net2.parameters(), lr=q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)

    def target_ini(self, net, target_net):
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(param.data)
        return target_net

    def target_soft_update(self, net, target_net, soft_tau):
        # Soft update the target net
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau)

        return target_net

    def update(self, batch_size, deterministic, eval_noise_scale, reward_scale=1.0, gamma=0.99, soft_tau=5e-3):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        # print('sample:', state, action,  reward, done)

        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(
            device)  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

        predicted_q_value1 = self.q_net1(state, action)
        predicted_q_value2 = self.q_net2(state, action)
        new_action, log_prob, z, mean, log_std = self.policy_net.evaluate(state, deterministic,
                                                                          eval_noise_scale=0.0)  # no noise, deterministic policy gradients
        new_next_action, _, _, _, _ = self.target_policy_net.evaluate(next_state, deterministic,
                                                                      eval_noise_scale=eval_noise_scale)  # clipped normal noise

        reward = reward_scale * (reward - reward.mean(dim=0)) / (reward.std(
            dim=0) + 1e-6)  # normalize with batch mean and std; plus a small number to prevent numerical problem

        # Training Q Function
        target_q_min = torch.min(self.target_q_net1(next_state, new_next_action),
                                 self.target_q_net2(next_state, new_next_action))

        target_q_value = reward + (1 - done) * gamma * target_q_min  # if done==1, only reward

        q_value_loss1 = (
                    (predicted_q_value1 - target_q_value.detach()) ** 2).mean()  # detach: no gradients for the variable
        q_value_loss2 = ((predicted_q_value2 - target_q_value.detach()) ** 2).mean()
        self.q_optimizer1.zero_grad()
        q_value_loss1.backward()
        self.q_optimizer1.step()
        self.q_optimizer2.zero_grad()
        q_value_loss2.backward()
        self.q_optimizer2.step()

        if self.update_cnt % self.policy_target_update_interval == 0:
            # Training Policy Function
            ''' implementation 1 '''
            # predicted_new_q_value = torch.min(self.q_net1(state, new_action),self.q_net2(state, new_action))
            ''' implementation 2 '''
            predicted_new_q_value = self.q_net1(state, new_action)

            policy_loss = - predicted_new_q_value.mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            # Soft update the target nets
            self.target_q_net1 = self.target_soft_update(self.q_net1, self.target_q_net1, soft_tau)
            self.target_q_net2 = self.target_soft_update(self.q_net2, self.target_q_net2, soft_tau)
            self.target_policy_net = self.target_soft_update(self.policy_net, self.target_policy_net, soft_tau)

        self.update_cnt += 1

        return predicted_q_value1.mean()

    def save_model(self, path):
        torch.save(self.q_net1.state_dict(), path + '_q1')
        torch.save(self.q_net2.state_dict(), path + '_q2')
        torch.save(self.policy_net.state_dict(), path + '_policy')

    def load_model(self, path):
        self.q_net1.load_state_dict(torch.load(path + '_q1'))
        self.q_net2.load_state_dict(torch.load(path + '_q2'))
        self.policy_net.load_state_dict(torch.load(path + '_policy'))
        self.q_net1.eval()
        self.q_net2.eval()
        self.policy_net.eval()


def plot(rewards):
    clear_output(True)
    plt.figure(figsize=(20, 5))
    plt.plot(rewards)
    plt.savefig('./results_v8_no_lstm/td3_no_'f'{n_index}''.png')
    # plt.show()


###########################################

datafile = './source_data/v2/original_data_1000_'f'{n_index}''_uniform.mat'
data = scio.loadmat(datafile)
env = my_environment(data)
action_space = env.action_space ##### Box(-2.0, 2.0, (1,), float32)
# print( spaces.Box((np.array([-1,0,0]),np.array([+1,+1,+1]))))
state_space = env.observation_space  ##### Box(-8.0, 8.0, (3,), float32)
action_range = 1.50
action_dim = 2 * n_index
state_dim =  5 * n_index

replay_buffer_size = 5e5
replay_buffer = ReplayBuffer(replay_buffer_size)
# hyper-parameters for RL training
max_episodes  = 1000
max_steps   = 20   # Pendulum needs 150 steps per episode to learn well, cannot handle 20
frame_idx   = 0
batch_size  = 45
explore_steps = 0  # for random action sampling in the beginning of training
update_itr = 1
hidden_dim = 512
policy_target_update_interval = 10 # delayed update for the policy network and target networks
DETERMINISTIC=True  # DDPG: deterministic policy gradient
explore_noise_scale = 0.1  # 0.5 noise is required for Pendulum-v0, 0.1 noise for HalfCheetah-v2
eval_noise_scale = 0.5
reward_scale = 1.
rewards     = []
fl_accuracy = []
fl_penalty = []
energy_consump = []
accumulate_select = np.zeros(n_index)
model_path = './model/td3_no_lstm_'f'{n_index}'

td3_trainer=TD3_Trainer(replay_buffer, hidden_dim=hidden_dim, policy_target_update_interval=policy_target_update_interval, action_range=action_range )


#################################################################
if __name__ == '__main__':
    if args.train:
        # training loop
        for eps in range(max_episodes):
            state = env.reset()  ##### initinal state     state =  [0.78021184 0.62551537 0.09575707]
            #print(state)
            episode_reward = 0
            episode_ener = 0
            episode_acc = 0
            episode_penalty = 0
            for step in range(max_steps):
                if frame_idx > explore_steps:
                    action = td3_trainer.policy_net.get_action(state, deterministic=DETERMINISTIC, explore_noise_scale=explore_noise_scale)
                else:
                    action = td3_trainer.policy_net.sample_action()

                final_accuracy, final_energy_cmp, next_state, penalty,  reward, device_select, done, = env.step(action, eps)
                #accumulate_select = accumulate_select + device_select
                replay_buffer.push(state, action, reward, next_state, done)

                state = next_state
                reward = list(chain(*reward))
                reward = np.array(reward)
                episode_reward += reward
                episode_acc += final_accuracy
                episode_ener += final_energy_cmp
                episode_penalty +=  penalty
                frame_idx += 1

                if len(replay_buffer) > batch_size:
                    for i in range(update_itr):
                        _ = td3_trainer.update(batch_size, deterministic=DETERMINISTIC, eval_noise_scale=eval_noise_scale, reward_scale=reward_scale)
                if done:
                    break

            episode_reward = episode_reward / max_steps
            episode_acc = episode_acc / max_steps
            episode_ener = episode_ener / max_steps
            episode_penalty = episode_penalty / max_steps
            accumulate_select = accumulate_select + device_select
            #episode_reward = episode_reward / max_steps
            if eps % 10 == 0 and eps > 0:
                plot(rewards)
                np.save('./results_v8/datasize_experiment_results/normal_distribution/rewards_td3_no_lstm_1000_30_'f'{n_index}''_normal', rewards)
                td3_trainer.save_model(model_path)

            print('Episode: ', eps, '| Episode Reward: ', episode_reward)
            rewards.append(episode_reward)
            fl_accuracy.append(episode_acc)
            energy_consump.append(episode_ener)
            fl_penalty.append(episode_penalty)
        td3_trainer.save_model(model_path)
        #np.save('./results_v8_no_lstm/accuracy_td3_no_lstm_'f'{n_index}', fl_accuracy)
        #np.save('./results_v8_no_lstm/energy_td3_no_lstm_'f'{n_index}', energy_consump)
        #np.save('./results_v8_no_lstm/select_td3_no_lstm_'f'{n_index}', accumulate_select)
        #np.save('./results_v8_no_lstm/penalty_td3_no_lstm_'f'{n_index}', fl_penalty)
        np.save('./results_v8/datasize_experiment_results/normal_distribution/accuracy_td3_no_lstm_1000_30_'f'{n_index}''_normal', fl_accuracy)
        np.save('./results_v8/datasize_experiment_results/normal_distribution/energy_td3_no_lstm_1000_30_'f'{n_index}''_normal', energy_consumption)
        np.save('./results_v8/datasize_experiment_results/normal_distribution/select_td3_no_lstm_1000_30_'f'{n_index}''_normal', accumulate_select)
        np.save('./results_v8/datasize_experiment_results/normal_distribution/penalty_td3_no_lstm_1000_30_'f'{n_index}''_normal', fl_penalty)
        

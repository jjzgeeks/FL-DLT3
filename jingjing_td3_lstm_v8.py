# Part of the code comes from https://github.com/quantumiracle/Popular-RL-Algorithms
import math
import random
# import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

from common.buffers import *
from common.value_networks import *
from common.policy_networks import *
from itertools import chain
from IPython.display import clear_output
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import display
#from reacher import Reacher  # Env
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
####################################################################################################
n_index = 60

class my_environment:
    def __init__(self,
                 data,
    ):  
        data_set = list(data['All_clients_dataset_info'])  ### time varying 20000 Rows, 10 Columns
        self.div = 1.0  ############## diversity
        self.data_set = self.div * np.array(data_set)
        #print(isinstance(self.data_set, tuple))
        self.bandwidth = np.array(data['All_clients_bandwidth_info'])  ### time varying 20000 Rows, 10 Columns
        self.trans_power = np.array(data['All_clients_transmission_power_info'])  ### time varying 20000 Rows, 10 Columns
        self.energy_harvesting = np.array(data['energy_harvesting']) ## time varying
        self.server_trans_power = 100 * self.trans_power # unit: Watt
        self.cpu_freq = np.array(data['f']) #unit: Hz
        self.max_episodes, self.K = self.data_set.shape
        self.max_trans_power = 1.5
        self.min_trans_power = 1.0e-01  # unit: Watt
        self.upload_data_size = 5.0e4 * np.ones(self.K) # unit: bit
        self.download_data_size = 1.0e4 * np.ones(self.K)
        self.cpu_cycles = 20 # unit cost of each data sample   cycles/bit
        self.channel_gain =  np.array(data['G']) ##time varying
        self.Gaussian_noise = 1.0e-08 * np.ones(self.K),
        self.mu = 4.2e-9 #1.2e-10



        self.action_low = np.concatenate([np.zeros(self.K), self.min_trans_power * np.ones(self.K)], axis=0)
        self.action_high = np.concatenate([np.ones(self.K), self.max_trans_power * np.ones(self.K)], axis=0)
        self.observation_low = np.concatenate([2.0e6 * np.ones(self.K), 1.0e4 * np.ones(self.K), 1.0e-3 * np.ones(self.K), 1.0 * np.ones(self.K), 1.0*np.ones(self.K)], axis=0)
        self.observation_high = np.concatenate([9.0e6 * np.ones(self.K), 1.0e5 * np.ones(self.K), 1.0e-2 * np.ones(self.K), 1.0e3 * np.ones(self.K), 50 * np.ones(self.K)], axis=0)
        self.action_space = spaces.Box(low=self.action_low, high=self.action_high)
        self.observation_space = spaces.Box(low=self.observation_low, high= self.observation_high)


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
        energy_c = 1.0e-3 * (energy_cmp + energy_up) #unit : KJ
        energy = energy - device_select * energy_c + self.energy_harvesting[t,:]
        energy = list(chain(*energy))
        energy = np.array(energy) #Unit: kJ
        time_train = number_of_local_epochs * self.cpu_cycles * self.data_set[t,:] / self.cpu_freq
        time_down = self.download_data_size / self.bandwidth[t, :] * np.log2(1 + (self.server_trans_power[t,:] * self.channel_gain[t, :]) / (self.Gaussian_noise * self.bandwidth[t, :]))
        time_up = self.upload_data_size / self.bandwidth[t, :] * np.log2(1 + (upload_trans_power * self.channel_gain[t, :]) / (self.Gaussian_noise * self.bandwidth[t, :]))
        time_delay  = time_train + time_down + time_up
        time_delay = list(chain(*time_delay))
        time_delay = np.array(time_delay)
        T_max = max(device_select * time_delay)
        #print(T_max)
        #print(time_delay)
        #self.channel_gain = list(chain(*self.channel_gain))
        next_state = np.concatenate([self.data_set[t, :], self.bandwidth[t, :], self.channel_gain[t, :], energy, time_delay], axis = 0)
       
        device_select = np.array([device_select])
        final_accuracy =  np.log(1 + self.mu * np.dot(device_select, (self.data_set[t, :]).T))  ##  a real number
        #print(final_accuracy)
        final_energy_cmp = np.dot(device_select, energy_c.T) ##  a real number
        #print(final_energy_cmp)
        #epsilon = 5.0e-9
        epsilon = 1.0e-10
        eta = 1
        penalty = epsilon * np.mean(device_select * self.data_set[t,:] - eta * sum(self.data_set[t,:]))
        reward = final_accuracy / final_energy_cmp  + penalty ##  a real number
        #print(penalty)
        #print(reward) # log2

        if all(energy) == 0:
            done = 1
        else:
            done = 0
        next_state = list(np.array(next_state).flatten())
        next_state = np.array(next_state)
        return  next_state, penalty, reward, final_accuracy, final_energy_cmp, device_select, T_max, done

    def reset(self):
        energy_initial = 8.0e2 * np.ones(self.K)
        time_delay = np.zeros(self.K)
        initial_state = np.concatenate([self.data_set[0,:], self.bandwidth[0, :], self.channel_gain[0,:], energy_initial, time_delay],axis=0)
        return initial_state


#######################################################################################################
class TD3_Trainer():
    def __init__(self, replay_buffer, state_space, action_space, hidden_dim, action_range,
                 policy_target_update_interval=1):
        self.replay_buffer = replay_buffer
        self.hidden_dim = hidden_dim

        self.q_net1 = QNetworkLSTM(state_space, action_space, hidden_dim).to(device)
        self.q_net2 = QNetworkLSTM(state_space, action_space, hidden_dim).to(device)
        self.target_q_net1 = QNetworkLSTM(state_space, action_space, hidden_dim).to(device)
        self.target_q_net2 = QNetworkLSTM(state_space, action_space, hidden_dim).to(device)
        self.policy_net = DPG_PolicyNetworkLSTM(state_space, action_space, hidden_dim).to(device)
        self.target_policy_net = DPG_PolicyNetworkLSTM(state_space, action_space, hidden_dim).to(device)
        print('Q Network (1,2): ', self.q_net1)
        print('Policy Network: ', self.policy_net)

        self.target_q_net1 = self.target_ini(self.q_net1, self.target_q_net1)
        self.target_q_net2 = self.target_ini(self.q_net2, self.target_q_net2)
        self.target_policy_net = self.target_ini(self.policy_net, self.target_policy_net)

        q_lr = 3e-4
        #q_lr = 3e-4
        policy_lr = 3e-4
        #policy_lr = 3e-4
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
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

        return target_net

    def update(self, batch_size, deterministic, eval_noise_scale, reward_scale=1., gamma=0.99, soft_tau=5e-3):
        hidden_in, hidden_out, state, action, last_action, reward, next_state, done = self.replay_buffer.sample(
            batch_size)
        # print('sample:', state, action,  reward, done)
        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.FloatTensor(action).to(device)
        last_action = torch.FloatTensor(last_action).to(device)

        reward = torch.FloatTensor(reward).unsqueeze(-1).to(device)

        done = torch.FloatTensor(np.float32(done)).unsqueeze(-1).to(device)
        predicted_q_value1, _ = self.q_net1(state, action, last_action, hidden_in)
        predicted_q_value2, _ = self.q_net2(state, action, last_action, hidden_in)
        new_action, _ = self.policy_net.evaluate(state, last_action, hidden_in,
                                                 noise_scale=0.0)  # no noise, deterministic policy gradients
        new_next_action, _ = self.target_policy_net.evaluate(next_state, action, hidden_out,
                                                             noise_scale=eval_noise_scale)  # clipped normal noise
        # reward = reward_scale * (reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6) # normalize with batch mean and std; plus a small number to prevent numerical problem

        # Training Q Function
        predicted_target_q1, _ = self.target_q_net1(next_state, new_next_action, action, hidden_out)
        predicted_target_q2, _ = self.target_q_net2(next_state, new_next_action, action, hidden_out)
        target_q_min = torch.min(predicted_target_q1, predicted_target_q2)
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
            predicted_new_q_value, _ = self.q_net1(state, new_action, last_action, hidden_in)

            policy_loss = - predicted_new_q_value.mean()
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
            # Soft update the target nets
            self.target_q_net1 = self.target_soft_update(self.q_net1, self.target_q_net1, soft_tau)
            self.target_q_net2 = self.target_soft_update(self.q_net2, self.target_q_net2, soft_tau)
            self.target_policy_net = self.target_soft_update(self.policy_net, self.target_policy_net, soft_tau)

        self.update_cnt += 1

        return predicted_q_value1.mean()  # for debug

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
    plt.savefig('./results_v8/number_of_devices_experiment_results/noise_0_5/rewards_td3_lstm_'f'{n_index}''.png')
    #plt.show()

#K = 10
#max_episodes = 1000

datafile = './source_data/v2/original_data_1000_'f'{n_index}''_uniform.mat'
data = scio.loadmat(datafile)
env = my_environment(data)
action_space = env.action_space ##### Box(-2.0, 2.0, (1,), float32)
# print( spaces.Box((np.array([-1,0,0]),np.array([+1,+1,+1]))))
state_space = env.observation_space  ##### Box(-8.0, 8.0, (3,), float32)
action_range = 1.50

###################################################################################
replay_buffer_size = 5e5
replay_buffer = ReplayBufferLSTM2(replay_buffer_size)
# hyper-parameters for RL training
max_episodes = 1000
max_steps = 20  # 
frame_idx = 0
batch_size = 45 #32 40  48 56 # each sample contains an episode for lstm policy
explore_steps = 0  # for random action sampling in the beginning of training
update_itr = 1
hidden_dim = 512
policy_target_update_interval = 10  # delayed update for the policy network and target networks
DETERMINISTIC = True  # DDPG: deterministic policy gradient
#explore_noise_scale = 0.1
explore_noise_scale = 0.5 
#explore_noise_scale = 0.9
eval_noise_scale = 0.5
reward_scale = 1.
rewards = []
fl_accuracy = []
device_energy_consumption = []
T_round_delay = []
fl_penalty = []
episode_runtime = []
accumulate_select = np.zeros(n_index)
model_path = './model/jingjing_td3_lstm_'f'{n_index}'

td3_trainer = TD3_Trainer(replay_buffer, state_space, action_space, hidden_dim=hidden_dim, \
                          policy_target_update_interval=policy_target_update_interval, action_range=action_range)




###########################################################################
if __name__ == '__main__':
    if args.train:
        # training loop
        for eps in range(max_episodes):
            state = env.reset()  ##### initinal state     state =  [0.78021184 0.62551537 0.09575707]
            #print(state.shape)
            last_action = env.action_space.sample()  #### last action [-1.1974558]
            #print(last_action.shape)
            episode_state = []
            episode_action = []
            episode_last_action = []
            episode_reward = []
            episode_next_state = []
            episode_done = []
            episode_accuracy = []
            episode_energy_cmp = []
            episode_time_delay = []
            episode_penalty = []
            start = time.time()
            hidden_out = (torch.zeros([1, 1, hidden_dim], dtype=torch.float).cuda(), \
                          torch.zeros([1, 1, hidden_dim],
                                      dtype=torch.float).cuda())  # initialize hidden state for lstm, (hidden, cell), each is (layer, batch, dim)
            #start = time.time()
            for step in range(max_steps):
                hidden_in = hidden_out
                action, hidden_out = td3_trainer.policy_net.get_action(state, last_action, hidden_in, noise_scale=explore_noise_scale)
                next_state, penalty, reward, final_accuracy, final_energy_cmp, device_select, T_max, done = env.step(action, eps) ###### important
                if step == 0:
                    ini_hidden_in = hidden_in
                    ini_hidden_out = hidden_out
                episode_state.append(state)
                episode_action.append(action)
                episode_last_action.append(last_action)
                episode_reward.append(reward)
                episode_next_state.append(next_state)
                episode_done.append(done)
                episode_accuracy.append(final_accuracy)
                episode_penalty.append(penalty)
                episode_energy_cmp.append(final_energy_cmp)
                episode_time_delay.append(T_max)
                state = next_state
                last_action = action  ## update last action
                frame_idx += 1
               # accumulate_select = accumulate_select + device_select                
                #print(accumulate_select)
                if len(replay_buffer) > batch_size:
                    for i in range(update_itr):
                        _ = td3_trainer.update(batch_size, deterministic=DETERMINISTIC, eval_noise_scale=eval_noise_scale, reward_scale=reward_scale)

                if done:
                    break
            #episode_accuracy.append(final_accuracy)
            #episode_penalty.append(penalty)
            #episode_energy_cmp.append(final_energy_cmp)
            #episode_time_delay.append(T_max)
            accumulate_select = accumulate_select + device_select                
            end = time.time()
            e_runtime = end - start
            episode_runtime.append(e_runtime)
            #print(episode_runtime)
            replay_buffer.push(ini_hidden_in, ini_hidden_out, episode_state, episode_action, episode_last_action, \
                               episode_reward, episode_next_state, episode_done)

            if eps % 10 == 0 and eps > 0:
                plot(rewards)
                #np.save('./results_v8/1e_9/rewards_td3_lstm_'f'{n_index}', rewards)
                np.save('./results_v8/number_of_devices_experiment_results/noise_0_5/rewards_td3_lstm_'f'{n_index}', rewards)
                td3_trainer.save_model(model_path)

            print('Episode: ', eps, '| Episode Reward: ', np.mean(episode_reward))
            #print('Episode: ', eps, '| Episode Accuracy: ', np.mean(episode_accuracy))
            #print('Episode: ', eps, '| Episode Energy consumption: ', np.mean(episode_energy_cmp))
            rewards.append(np.mean(episode_reward))
            fl_accuracy.append(np.mean(episode_accuracy))
            fl_penalty.append(np.mean(episode_penalty))
            device_energy_consumption.append(np.mean(episode_energy_cmp))
            T_round_delay.append(np.mean(episode_time_delay))
        td3_trainer.save_model(model_path)

        #np.save('./results_v8/1e_9/accuracy_td3_lstm_'f'{n_index}', fl_accuracy)
        #np.save('./results_v8/1e_9/energy_td3_lstm_'f'{n_index}', device_energy_consumption)
        #np.save('./results_v8/1e_9/select_td3_lstm_'f'{n_index}', accumulate_select)
        #np.save('./results_v8/1e_9/penalty_td3_lstm_'f'{n_index}', fl_penalty)
        #np.save('./results_v8/1e_9/runtime_td3_lstm_'f'{n_index}', episode_runtime)

        np.save('./results_v8/number_of_devices_experiment_results/noise_0_5/accuracy_td3_lstm_'f'{n_index}', fl_accuracy)
        np.save('./results_v8/number_of_devices_experiment_results/noise_0_5/energy_td3_lstm_'f'{n_index}', device_energy_consumption)
        np.save('./results_v8/number_of_devices_experiment_results/noise_0_5/select_td3_lstm_'f'{n_index}', accumulate_select)
        np.save('./results_v8/number_of_devices_experiment_results/noise_0_5/penalty_td3_lstm_'f'{n_index}', fl_penalty)
        np.save('./results_v8/number_of_devices_experiment_results/noise_0_5/runtime_td3_lstm_'f'{n_index}', episode_runtime)

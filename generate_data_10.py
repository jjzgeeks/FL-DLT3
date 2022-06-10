# Uniformly distributed random number generation
import numpy as np
from scipy.io import savemat
#np.random.seed(1234) #set random seed 1234


T_round = 1000 # the total number of t_round or iteration

#The_all_number_of_devices = [20,30, 40]
The_all_number_of_devices = [10]
#The_all_number_of_devices = [10, 20,30,40,50, 60, 70, 80
for Number_of_clients in The_all_number_of_devices:
    All_clients_dataset_info = np.random.uniform(2.0e6, 1.0e7, size = (T_round, Number_of_clients)) # all clien [1, 100]MB 
    All_clients_bandwidth_info = np.random.uniform(1.0e4, 5.0e4, size = (T_round, Number_of_clients)) # all client's bandwidth info, which follow uniform distribution [5,15] unit: Hz
    All_clients_transmission_power_info = np.random.uniform(0.1, 1.50, size = (T_round, Number_of_clients)) # unit dBm
    energy_harvesting = np.random.uniform(0.05, 0.2, size = (T_round, Number_of_clients)) # unit: KJ

    G = np.random.uniform(1.0e-3, 1.0e-2, size = (T_round, Number_of_clients)) # channel gain

    unit_cost =  np.random.uniform(20, 20, size =  (T_round, Number_of_clients))  # each data cost   unit:  cycles/bit

    f = np.random.uniform(2.0,4.0, size = (1, Number_of_clients))*1.0e+09 # unifrnd can generate fraction  1GHz = 1.0e+09 Hz
    # unidrnd(2,4, 1, 10) %% unidrnd Generate interger

    #save original_data_1000_100  All_clients_dataset_info All_clients_bandwidth_info All_clients_transmission_power_info G  f  unit_cost
    savemat("./original_data_1000_"f'{Number_of_clients}'"_uniform.mat", {'All_clients_dataset_info' : All_clients_dataset_info, 'All_clients_bandwidth_info': All_clients_bandwidth_info, 'All_clients_transmission_power_info': All_clients_transmission_power_info, 'G':G, 'unit_cost': unit_cost, 'f':f, 'energy_harvesting':energy_harvesting})


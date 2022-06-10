import numpy as np
import scipy.io
from scipy.io import savemat

T_round = 1000 # the total number of t_round or iteration
Total_number_devices = 80

Add_number_of_devices = 10 ## fixed number
previous_number_devices = Total_number_devices - Add_number_of_devices

################## load previous IoT devices' information
mat = scipy.io.loadmat('./original_data_1000_'f'{previous_number_devices}''_uniform.mat')

All_clients_dataset_info = mat['All_clients_dataset_info']
All_clients_bandwidth_info = mat['All_clients_bandwidth_info']
All_clients_transmission_power_info = mat['All_clients_transmission_power_info']
G = mat['G']

unit_cost = mat['unit_cost']
f = mat['f']
#print(f)
energy_harvesting = mat['energy_harvesting']


##################### generate new information 10 IoT devices, then splicing
All_clients_dataset_info_new = np.random.uniform(2.0e6, 1.0e7, size = (T_round, Add_number_of_devices)) # all clien [1, 100]MB 
All_clients_dataset_info = np.append(All_clients_dataset_info, All_clients_dataset_info_new, axis = 1)
#print(All_clients_dataset_info.shape)

All_clients_bandwidth_info_new = np.random.uniform(1.0e4, 5.0e4, size = (T_round, Add_number_of_devices)) # all client's bandwidth info, which follow uniform distribution [5,15] unit: Hz
All_clients_bandwidth_info = np.append(All_clients_bandwidth_info, All_clients_bandwidth_info_new, axis = 1)


All_clients_transmission_power_info_new = np.random.uniform(0.1, 1.50, size = (T_round, Add_number_of_devices)) # unit dBm
All_clients_transmission_power_info = np.append(All_clients_transmission_power_info, All_clients_transmission_power_info_new, axis = 1)

G_new = np.random.uniform(1.0e-3, 1.0e-2, size = (T_round, Add_number_of_devices)) # channel gain
G = np.append(G, G_new, axis = 1)

unit_cost_new =  np.random.uniform(20, 20, size =  (T_round, Add_number_of_devices))  # each data cost   unit:  cycles/bit
unit_cost = np.append(unit_cost, unit_cost_new, axis = 1)

f_new = np.random.uniform(2.0,4.0, size = (1, Add_number_of_devices))*1.0e+09 # unifrnd can generate fraction  1GHz = 1.0e+09 Hz
f = np.append(f, f_new, axis = 1)
#print(f)

energy_harvesting_new = np.random.uniform(0.05, 0.2, size = (T_round, Add_number_of_devices)) # unit: KJ
energy_harvesting = np.append(energy_harvesting, energy_harvesting_new, axis = 1)


savemat("./original_data_1000_"f'{Total_number_devices}'"_uniform.mat", {'All_clients_dataset_info' : All_clients_dataset_info, 'All_clients_bandwidth_info': All_clients_bandwidth_info, 'All_clients_transmission_power_info': All_clients_transmission_power_info, 'G':G, 'unit_cost': unit_cost, 'f':f, 'energy_harvesting':energy_harvesting})


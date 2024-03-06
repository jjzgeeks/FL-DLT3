# About FL-DLT3


Welcome to  FL-DLT3, created for experimental research in an article published on IEEE Internet of Things Journal.


Here is a structure of FL-DLT3:

![Image alt text.](https://github.com/jjzgeeks/FL-DLT3/blob/main/readme_pics/FL-DLT3.png) 
Illustration of the proposed FL-DLT3 framework, where a policy network, a policy target network, two critic networks, and two critic target networks are trained. Each of the networks consists of a feedforward branch and a recurrent branch. A training round of FL is performed by the selected IoT devices with the allocated transmit power.Then, the edge server gets the reward $R_{\alpha}$ and the state update $S_{\alpha^{'}}^o$ of the IoT devices. The transition  {($A_{\alpha^{-}}$, $S_{\alpha}^o$, $A_{\alpha}$, $R_{\alpha}$, $S_{\alpha^{'}}^o$)}  is stored into the replay buffer $\mathcal{B}$, and a mini-batch of transitions is randomly sampled from B to train the policy and critic networks on the edge server.


J. Zheng, K. Li, N. Mhaisen, W. Ni, E. Tovar and M. Guizani, "Exploring Deep Reinforcement Learning-Assisted Federated Learning for Online Resource Allocation in Privacy-Preserving EdgeIoT," in IEEE Internet of Things Journal, doi: 10.1109/JIOT.2022.3176739.  The more details can be found [here](https://ieeexplore.ieee.org/document/9779339)


## If your computer has two GPUs, run the program as following command
$ CUDA_VISIBLE_DEVICES=0,1 python jingjing_td3_lstm_v8.py --train

# Citation
```
@article{zheng2022exploring,
  title={Exploring Deep-Reinforcement-Learning-Assisted federated learning for Online Resource Allocation in Privacy-Preserving EdgeIoT},
  author={Zheng, Jingjing and Li, Kai and Mhaisen, Naram and Ni, Wei and Tovar, Eduardo and Guizani, Mohsen},
  journal={IEEE Internet of Things Journal},
  volume={9},
  number={21},
  pages={21099--21110},
  year={2022},
  publisher={IEEE}
}
```

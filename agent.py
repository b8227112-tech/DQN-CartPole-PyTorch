import random

import numpy as np
import torch
import torch.nn as nn



class Replaymome: # 定义Reply Buffer
    def __init__(self, n_s,n_a): # 观察变量的个数
        self.n_s = n_s
        self.n_a = n_a
        self.memory_size = 1000
        self.batch_size = 64

        # 随机和空会不会有区别（？）(猜测：为了随机采样)
        self.all_s = np.empty(shape= (self.memory_size,self.n_s), dtype=np.float32)
        self.all_a = np.random.randint(low=0, high=n_a, size=self.memory_size, dtype=np.uint8)
        self.all_r = np.empty(self.memory_size, dtype=np.float32)
        self.all_terminal = np.random.randint(low=0, high=2, size=self.memory_size,dtype=np.uint8)
        self.all_s_ = np.empty(shape= (self.memory_size,self.n_s),dtype=np.float32)

        self.t_mome = 0
        self.t_max = 0

    def add_mome(self, s, a, r, terminal, s_): # add_mome
        self.all_s[self.t_mome] = s
        self.all_a[self.t_mome] = a
        self.all_r[self.t_mome] = r
        self.all_terminal[self.t_mome] = terminal
        self.all_s_[self.t_mome] = s_
        self.t_mome = (self.t_mome +1) % self.memory_size
        self.t_max = max(self.t_max, self.t_mome)

    def sample(self): # sample
        # t_max大于要抽取的batch，随机抽取
        # t_max + random 随机取idexs
        if self.t_max > self.batch_size:
            idexs = random.sample(range(0,self.t_max), self.batch_size) # random.sample 无放回抽样

        # 小于要抽取的batch，全部拿出来
        else:
            idexs = range(0, self.t_max)

        # 把选择的数据拿出来
        batch_s = []
        batch_a = []
        batch_r = []
        batch_terminated = []
        batch_s_ = []

        for idx in idexs:
            batch_s.append(self.all_s[idx])
            batch_a.append(self.all_a[idx])
            batch_r.append(self.all_r[idx])
            batch_terminated.append(self.all_terminal[idx])
            batch_s_.append(self.all_s_[idx])

        # torch不能识别numpy，把数据转为tensor
        # a、r和done要升维，unsqueeze(-1)，因为神经网络的输入需要
        batch_s = torch.as_tensor(batch_s, dtype=torch.float32)
        batch_a = torch.as_tensor(batch_a, dtype=torch.int64).unsqueeze(-1)
        batch_r = torch.as_tensor(batch_r, dtype=torch.float32).unsqueeze(-1)
        batch_terminated = torch.as_tensor(batch_terminated, dtype=torch.uint8).unsqueeze(-1)
        batch_s_ = torch.as_tensor(batch_s_, dtype=torch.float32)

        # 返回数据
        return batch_s, batch_a, batch_r, batch_terminated, batch_s_

class DQN(nn.Module): # 写两个神经网络，一个类就够了
     def __init__(self,n_input,n_output):
         super(DQN,self).__init__() # 继承父类

         self.net = nn.Sequential(
             nn.Linear(n_input, 88),
             nn.Tanh(),
             nn.Linear(88,n_output)
         ) # 定义net 中间是激活函数

     def forward(self,x): # 斜/反向传播
         return self.net(x)

     def act(self,obs): # 定义act
         obs_tensor = torch.as_tensor(obs, dtype=torch.float32) # 把obs转化为tensor
         q_values = self.forward(obs_tensor.unsqueeze(0)) # q_value转为行向量
         max_q_idx = torch.argmax(q_values,dim=1) # 找到最大的q_value的序号
         action = max_q_idx.detach().item() # 找到最大的q_value对应的action，detach的用法（？）

         return action

class Agent: # 定义agent类
    def __init__(self,n_input,n_output):
        self.n_input = n_input
        self.n_output = n_output

        self.Gamma = 0.9 # Gamma
        self.lr = 1e-3 # learning rate

        self.mome = Replaymome(self.n_input,self.n_output) # mome

        self.online_net = DQN(self.n_input,self.n_output) # online network
        self.target_net = DQN(self.n_input,self.n_output) # target network

        self.optimizer = torch.optim.Adam(self.online_net.parameters(),lr=self.lr) # 梯度下降使用torch.optim.Adam去优化online_net的参数
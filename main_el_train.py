import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from agent import Agent

env = gym.make('CartPole-v1')
s, info = env.reset() # 必须解包

N_EPISODE = 500
N_STEP = 500
EPSILON_BEGIN = 1
EPSILON_END = 0.02
EPSILON_DECAY = 10000
REWARD_BUFFER = np.zeros(N_EPISODE)
C = 10

n_state = len(s)
n_action = env.action_space.n
agent = Agent(n_input=n_state,n_output=n_action)

for episode_i in range(N_EPISODE): # 遍历每一个episode
    episode_reward = 0
    for step_i in range(N_STEP):# 每一个episode里面走 T 步
        epsilon = np.interp(episode_i * N_STEP +step_i,[0,EPSILON_DECAY],[EPSILON_BEGIN,EPSILON_END])# 设置epsilon
        sample = np.random.uniform()# 采样

        #选择action，获得 rt 和 xt+1
        if sample < epsilon:
            action = env.action_space.sample() # 随便算一个
        else:
            action = agent.online_net.act(s)

        s_, r,terminated, truncated, info = env.step(action)
        agent.mome.add_mome(s, action, r, terminated, s_) # 把经验(Φt, at, rt, Φt+1)放到经验池当中
        s = s_# 状态转移 st+1=st, at, xt+1
        episode_reward += r

        if terminated: # 判断游戏是否结束
            env.reset() #环境重置
            REWARD_BUFFER[episode_i] = episode_reward # 只存奖励，用于监控和评估
            break

        batch_s,batch_a,batch_r,batch_terminated,batch_s_=agent.mome.sample() # 提取一个mini_batch{(st, at, rt, st+1)}

        # 计算target output (yt = rt + gamma max_a q(Φt,at;θ))
        target_q_value = agent.target_net(batch_s_)
        q_hat = target_q_value.max(dim=1,keepdims=True) [0]
        target_output = batch_r + (1-batch_terminated) * agent.Gamma * q_hat

        q_value = agent.online_net(batch_s) # 更新main network 对应的q value
        a_q_value = torch.gather(input=q_value,dim=1,index=batch_a) # 获得每一个s下最大的q value且要与action对应,即在s状态下采取的a

        loss = nn.functional.smooth_l1_loss(target_output,a_q_value) # 计算Loss，关于(yt-q(Φt,at;θ))^2

        # 梯度下降
        agent.optimizer.zero_grad()
        loss.backward()  #反向传播
        agent.optimizer.step()

    # 每C步后令q_hat=q
    if episode_i % C == 0:
        agent.target_net.load_state_dict(agent.online_net.state_dict())

        print("Episode: {}".format(episode_i))
        print("Average Reward: {}".format(np.mean(REWARD_BUFFER)))
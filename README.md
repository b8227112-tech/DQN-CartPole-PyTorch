# DQN Implementation in PyTorch

本项目基于 PyTorch 独立复现了 DeepMind 团队发表于 Nature 的经典深度强化学习算法，并在标准 CartPole 环境下进行了测试。

**📄 Reference Paper:**
> Mnih, V., Kavukcuoglu, K., Silver, D. et al. *Human-level control through deep reinforcement learning*. Nature 518, 529–533 (2015).

## 核心机制实现
- **Replay Buffer (经验回放池)**：打破序列数据相关性。
- **Target Network (目标网络)**：采用双网络结构，定期硬更新参数，提升训练稳定性。

## 运行环境
- Python 3.12
- PyTorch
- OpenAI Gym (CartPole-v1)

## 快速开始
运行主程序即可开始训练智能体：
`python main_el_train.py`

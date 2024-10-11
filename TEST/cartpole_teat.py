from collections import deque
from operator import truediv

from numpy import random
import matplotlib.pyplot as plt  # 导入 matplotlib
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# 环境初始化：
#
# 使用 gym.make('CartPole-v1') 创建了 CartPole-v1 环境，这是一个常见的强化学习环境。
# 策略网络：
#
# PolicyNetwork 类是一个两层全连接神经网络。它将环境状态作为输入，并输出不同动作的概率。
# fc1 是第一层全连接层，输入是环境状态的特征，输出维度为 64。
# fc2 是第二层全连接层，输出动作概率，动作的数量由环境的动作空间大小决定。
# 优化器：
#
# Adam 优化器用于更新策略网络的参数，学习率设置为 1e-2。
# 策略评估函数 evaluate_policy：
#
# evaluate_policy 函数用于评估策略网络的表现，计算若干回合的平均奖励。
# 每一回合中，策略网络根据当前状态选择动作，执行该动作并累积奖励，直到回合结束。
# terminated 和 truncated 是新的 gym 版本中的状态标志，表示任务是否由于成功/失败或时间限制而结束。
# 模型保存：
#
# 使用 torch.save 将训练好的策略网络参数保存到文件 'policy_net_model.pth'，便于后续加载模型进行继续训练或评估。
# # 创建CartPole环境
env = gym.make('CartPole-v1')
# 加载模型权重（如果需要的话）

# policy_net.eval()  # 切换到评估模式
# 修复 numpy bool8 的问题，防止某些版本的 gym 依赖 numpy.bool8 引发错误
if not hasattr(np, 'bool8'):
    np.bool8 = bool  # 如果 numpy.bool8 不存在，则将其设为标准的 bool 类型

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)

# 定义策略网络类，继承自PyTorch的nn.Module基类
class PolicyNetwork(nn.Module):

    # 初始化方法，接收输入维度和输出维度作为参数
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()  # 调用父类的初始化方法
        # 定义第一个全连接层，输入维度为input_dim，输出维度为64
        self.fc1 = nn.Linear(input_dim, 64)
        # 定义第二个全连接层，输入维度为64，输出维度为output_dim
        self.fc2 = nn.Linear(64, output_dim)

    # 前向传播方法，接收状态作为输入，返回动作概率
    def forward(self, state):
        # 将输入状态通过第一个全连接层，并应用ReLU激活函数
        x = torch.relu(self.fc1(state))
        # 将经过ReLU激活的输出通过第二个全连接层，并应用softmax函数
        # softmax函数用于将输出转化为概率分布，dim=-1表示在最后一个维度上应用softmax
        action_probs = torch.softmax(self.fc2(x), dim=-1)
        # 返回动作概率分布
        return action_probs

# 获取输入状态维度和动作空间维度
input_dim = env.observation_space.shape[0]  # 状态空间维度
output_dim = env.action_space.n  # 动作空间大小
# 创建策略网络实例
policy_net = PolicyNetwork(input_dim, output_dim)
policy_net.load_state_dict(torch.load('policy_net_model.pth',weights_only=True))
# 使用Adam优化器，学习率为1e-2
optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
# 选择动作
def select_action(state):
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    action_probs = policy_net(state_tensor)
    action = torch.multinomial(action_probs, num_samples=1).item()
    return action

# 定义评估策略网络的函数
def evaluate_policy(episodes):
    replay_buffer = ReplayBuffer(10000)  # 经验重放缓存
    # 评估模型的表现，计算平均奖励
    best_avg_reward = -float('inf')
    total_rewards = 0  # 总奖励初始化为0
    avg_reward_history = []
    no_improvement_count = 0
    rewards_=[]
    for i in range(episodes):
        # 重置环境，获取初始状态
        state, _ = env.reset()  # 最新版本的gym中，reset()会返回一个状态和信息字典
        done = False  # done表示回合是否结束
        episode_reward = 0  # 初始化当前回合的奖励为0
        achieve_=0

        rewards = []  # 记录奖励
        states = []  # 记录状态
        actions = []  # 记录动作
        # 开始一回合的交互
        while not done:
            if random.random() < 0.1:
                action = env.action_space.sample()
            else:
                # 将状态转换为PyTorch的Tensor，并扩展维度以符合输入要求
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                # 使用策略网络计算动作概率，关闭梯度计算以节省资源
                # with torch.no_grad():
                action = select_action(state)
                # 采取动作，返回新的状态、奖励、是否结束（done），以及其他信息
            next_state, reward, terminated, truncated, _ = env.step(action)
            # 新版本的gym将done拆分为terminated和truncated，需要合并判断回合是否结束
            done = terminated or truncated
            # 从经验重放中采样并更新策略
            if replay_buffer.size() >= 32:
                experiences = replay_buffer.sample(32)
                state, action, reward, next_state, done=experiences

            if done == terminated:
                achieve_+=1
            # 存储状态、动作和奖励
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            # 累积本回合的奖励
            episode_reward += reward

            # 将状态更新为下一步的状态
            state = next_state

        # 累积所有回合的奖励
        rewards_.append(episode_reward)
        total_rewards += episode_reward
        avg_reward = total_rewards / len(rewards)  # 计算平均奖励
        discounted_rewards = []  # 存储折扣奖励
        for t in range(len(rewards)):
            Gt = sum(rewards[t:] )  # 未来的总奖励
            discounted_rewards.append(Gt)
        # 转换为tensor
        discounted_rewards_tensor = torch.FloatTensor(np.array(discounted_rewards))
        actions_tensor = torch.LongTensor(np.array(actions))
        # 计算损失
        action_probs = policy_net(torch.FloatTensor(np.array(states)))
        action_probs = action_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze()
        loss = -torch.mean(torch.log(action_probs) * discounted_rewards_tensor)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_reward_history.append(avg_reward)
        print(f'Episode {i + 1}={episode_reward}, Total Reward: {total_rewards}, Average Reward: {avg_reward}')

        # print(f'Episode {i + 1}, Total Reward: {episode_reward}')
        # 检查是否收敛
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            no_improvement_count = 0  # 重置计数
            # 保存模型
            torch.save(policy_net.state_dict(), 'policy_net_model.pth')
            print(f'Model saved with average reward: {avg_reward}')
        else:
            no_improvement_count += 1

        # 如果在patience轮次内没有改进，停止训练
        if no_improvement_count >= 500:
            print("No improvement in average reward. Stopping training.")
            break
    return rewards_

episodes = 1000
# 使用上文定义的PolicyNetwork和初始化的环境进行策略评估

qq=evaluate_policy(episodes)

# 关闭环境
env.close()
# 保存模型权重到文件中，以便后续加载或评估
torch.save(policy_net.state_dict(), 'policy_net_model.pth')
# 绘制奖励图
plt.figure(figsize=(12, 6))
plt.plot(qq, label='Episode Rewards')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Reward per Episode')
plt.legend()
plt.grid()
plt.show()
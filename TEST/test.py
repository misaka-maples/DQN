from collections import deque
from operator import truediv
from numpy import random
import matplotlib.pyplot as plt  # 导入 matplotlib
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import StepLR
# # 创建CartPole环境
env = gym.make('CartPole-v1')
if not hasattr(np, 'bool8'):
    np.bool8 = bool  # 如果 numpy.bool8 不存在，则将其设为标准的 bool 类型

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, experience):
        self.buffer.append(experience)
    def sample(self, batch_size):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        return random.sample(self.buffer, batch_size)
    def size(self):
        return len(self.buffer)

# 定义策略网络类，继承自PyTorch的nn.Module基类
class PolicyNetwork(nn.Module):

    # 初始化方法，接收输入维度和输出维度作为参数
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()  # 调用父类的初始化方法
        # 定义第一个全连接层，输入维度为input_dim，输出维度为64
        self.fc1 = nn.Linear(input_dim, 128)
        # 定义第二个全连接层，输入维度为64，输出维度为output_dim
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)  # 输出层
    # 前向传播方法，接收状态作为输入，返回动作概率
    def forward(self, state):
        # 将输入状态通过第一个全连接层，并应用ReLU激活函数
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        # 将经过ReLU激活的输出通过第二个全连接层，并应用softmax函数
        # softmax函数用于将输出转化为概率分布，dim=-1表示在最后一个维度上应用softmax
        action_probs = torch.softmax(self.fc3(x), dim=-1)
        # 返回动作概率分布
        return action_probs

# 获取输入状态维度和动作空间维度
input_dim = env.observation_space.shape[0]  # 状态空间维度
output_dim = env.action_space.n  # 动作空间大小
# 创建策略网络实例
policy_net = PolicyNetwork(input_dim, output_dim)
#加载模型
# policy_net.load_state_dict(torch.load('policy_net_model_.pth',weights_only=True))
# 使用Adam优化器
optimizer = optim.RMSprop(policy_net.parameters(), lr=1e-2)  # 换成RMSprop
scheduler = StepLR(optimizer, step_size=150, gamma=0.9)  # 调整学习率调度器
# 选择动作
def select_action(state):
    epsilon = 0.1  # 设置探索概率
    if np.random.rand() < epsilon:
        action = env.action_space.sample()  # 随机探索动作
    else:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = policy_net(state_tensor)
        action_probs = torch.softmax(action_probs / 1.0, dim=-1)
        action_probs = torch.softmax(action_probs - torch.max(action_probs), dim=-1)
        # 检查概率分布是否包含无效值

        # 避免数值问题，将概率值限制在一个合理范围内
        action_probs = torch.clamp(action_probs, min=1e-8, max=1.0)
        if torch.any(action_probs == float('inf')) or torch.any(torch.isnan(action_probs)):
            print("Action probabilities contain inf or nan")
        # 检查概率分布是否包含负数
        if torch.any(action_probs < 0):
            print("Action probabilities contain negative values")

        # 根据这个分布随机选择一个动作
        action = torch.multinomial(action_probs, num_samples=1).item()
        # 选取概率最高的动作
        # action = torch.argmax(action_probs, dim=1).item()
    return action


# 定义训练策略网络的函数
def train_policy(episodes):
    #eposodes为训练次数
    replay_buffer = ReplayBuffer(10000)  # 经验重放缓存
    # 评估模型的表现，计算平均奖励
    best_avg_reward = 0
    total_rewards = 0  # 总奖励初始化为0
    avg_reward_history = []
    no_improvement_count = 0
    rewards_=[]
    for i in range(episodes):
        # 重置环境，获取初始状态
        state, _ = env.reset()  # 最新版本的gym中，reset()会返回一个状态和信息字典
        done = False  # done表示回合是否结束
        episode_reward = 0  # 初始化当前回合的奖励为0

        rewards = []  # 记录奖励
        states = []  # 记录状态
        actions = []  # 记录动作
        # 开始一回合的交互
        while not done:
            action = select_action(state)
            # 采取动作，返回新的状态、奖励、是否结束（done），以及其他信息
            next_state, reward, terminated, truncated, _ = env.step(action)
            # 新版本的gym将done拆分为terminated和truncated，需要合并判断回合是否结束
            done = terminated or truncated
            # 从经验重放中采样并更新策略
            if replay_buffer.size() >= 64:
                experiences = replay_buffer.sample(64)
                state, action, reward, next_state, done=experiences

            # 存储状态、动作和奖励
            # print(f"State: {state}")
            if np.any(np.isnan(state)) or np.any(np.isinf(state)):
                raise ValueError("Invalid state with NaN or Inf encountered.")

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            # 累积本回合的奖励
            episode_reward += reward

            # 将状态更新为下一步的状态
            state = next_state
            scheduler.step()
        # 累积所有回合的奖励
        rewards_.append(episode_reward)
        total_rewards += episode_reward
        avg_reward = total_rewards / len(rewards_)  # 计算平均奖励
        gamma = 0.95  # 折扣因子，通常在0到1之间
        discounted_rewards = []
        for t in range(len(rewards)):
            Gt = 0
            for reward in rewards[t:]:
                Gt = reward + gamma * Gt
            discounted_rewards.append(Gt)
        # 转换为tensor
        discounted_rewards_tensor = torch.FloatTensor(np.array(discounted_rewards))
        actions_tensor = torch.LongTensor(np.array(actions))
        # 计算损失
        action_probs = policy_net(torch.FloatTensor(np.array(states)))
        action_probs = action_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze()
        advantages = discounted_rewards_tensor - torch.mean(discounted_rewards_tensor)
        loss = -torch.mean(torch.log(action_probs) * advantages)
        entropy_loss = -torch.sum(action_probs * torch.log(action_probs))
        loss = loss - 0.01 * entropy_loss  # 0.01 是权重，可以调节

        # 反向传播和优化
        optimizer.zero_grad()
        # loss.backward()# 添加梯度裁剪
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)  # 限制梯度最大值为 1.0
        optimizer.step()
        avg_reward_history.append(avg_reward)
        print(f'Episode {i + 1}={episode_reward}, Total Reward: {total_rewards}, {len(rewards)}Average Reward: {avg_reward}')

        # print(f'Episode {i + 1}, Total Reward: {episode_reward}')
        # 检查是否收敛
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            no_improvement_count = 0  # 重置计数
            # 保存模型
            # torch.save(policy_net.state_dict(), 'policy_net_model_.pth')
            print(f'Model saved with average reward: {avg_reward}')
        else:
            no_improvement_count += 1

        # 如果在patience轮次内没有改进，停止训练
        if no_improvement_count >= 500:
            print("No improvement in average reward. Stopping training.")
            break
    return rewards_

rewards_=train_policy(10000)

def validate_policy(env, policy_net, episodes=10):
    """
    验证策略网络在给定环境中的表现。

    参数:
        env: Gym 环境
        policy_net: 策略网络
        episodes: 验证时运行的回合数

    返回:
        avg_reward: 验证过程中每回合的平均奖励
    """
    total_rewards = 0  # 累计所有回合的总奖励

    # 将策略网络设为评估模式（不更新梯度）
    policy_net.eval()

    with torch.no_grad():  # 关闭梯度计算以提高验证性能
        for episode in range(episodes):
            state, _ = env.reset()  # 重置环境，获取初始状态
            done = False
            episode_reward = 0  # 初始化本回合的奖励

            while not done:
                # 选择动作
                action = select_action(state)  # 使用已有的select_action函数
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated  # 判断回合是否结束
                episode_reward += reward  # 累计本回合奖励
                state = next_state  # 更新状态为下一步状态

            # 累计所有回合的总奖励
            total_rewards += episode_reward
            print(f'Validation Episode {episode + 1}: Reward = {episode_reward}')

    # 计算验证集上的平均奖励
    avg_reward = total_rewards / episodes
    print(f'Average Reward in Validation: {avg_reward}')

    # 恢复策略网络为训练模式
    policy_net.train()

    return avg_reward
# 绘制奖励图
plt.figure(figsize=(12, 6))
plt.plot(rewards_, label='Episode Rewards')
# plt.plot(validate_policy(env,policy_net,100), label='Validation Average Reward')
plt.xlabel('Episode')
plt.ylabel(' Reward')
plt.title('Training and Validation Rewards')
plt.legend()
plt.grid()
plt.show()

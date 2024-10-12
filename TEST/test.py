from collections import deque
from operator import truediv
from numpy import random
import matplotlib.pyplot as plt  # 导入 matplotlib
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from torch.optim.lr_scheduler import StepLR
# # 创建CartPole环境
env = gym.make('CartPole-v1')
# if not hasattr(np, 'bool8'):
#     np.bool8 = bool  # 如果 numpy.bool8 不存在，则将其设为标准的 bool 类型
if not hasattr(np, 'bool_'):
    np.bool8 = bool  # 如果 numpy.bool_ 不存在，则将其设为标准的 bool 类型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        # 使用TD-error进行优先级采样
        priorities = np.array([abs(exp[2]) for exp in self.buffer])  # TD-error作为优先级
        probabilities = priorities / sum(priorities)
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        return [self.buffer[i] for i in indices]

    def size(self):
        return len(self.buffer)

# 定义策略网络类，继承自PyTorch的nn.Module基类
class PolicyNetwork(nn.Module):

    # 初始化方法，接收输入维度和输出维度作为参数
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()  # 调用父类的初始化方法
        # 定义第一个全连接层，输入维度为input_dim，输出维度为64
        self.fc1 = nn.Linear(input_dim, 256)
        # 定义第二个全连接层，输入维度为64，输出维度为output_dim
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)  # 输出层
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
policy_net = PolicyNetwork(input_dim, output_dim).to(device)
target_net = PolicyNetwork(input_dim, output_dim).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()  # 目标网络不会更新

def boltzmann_policy(state, policy_net, tau=1.0):
    """
    Boltzmann 探索，这会将动作选择与其估计值概率性关联，能在一定程度上保持探索。
    :param state: 状态
    :param policy_net: 网络实例化
    :param tau:
    :return:
    """
    state_tensor = torch.FloatTensor(state,device=device).unsqueeze(0)
    action_probs = torch.softmax(policy_net(state_tensor) / tau, dim=-1)
    action = torch.multinomial(action_probs, 1).item()
    return action

def epsilon_greedy_policy(state, policy_net, epsilon, device):
    """
    ε-greedy策略函数，用于选择动作。

    参数:
    state -- 当前状态
    policy_net -- 策略网络模型
    epsilon -- 探索概率
    device -- 运行模型的设备（如CPU或GPU）

    返回:
    action -- 选择的动作
    """
    if np.random.rand() < epsilon:
        # 探索：随机选择一个动作
        action = env.action_space.sample()
    else:
        # 利用：根据策略网络选择动作
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action_probs = policy_net(state_tensor)
        action = torch.argmax(action_probs).item()
    return action
#加载模型
policy_net.load_state_dict(torch.load('policy_net_model_.pth',weights_only=True,map_location=device))
# 使用Adam优化器
optimizer = optim.AdamW(policy_net.parameters(), lr=1e-3)

# optimizer = optim.RMSprop(policy_net.parameters(), lr=1e-2)  # 换成RMSprop
scheduler = StepLR(optimizer, step_size=200, gamma=0.9)  # 调整学习率调度器
# 选择动作
def select_action(state, loop):
    epsilon = 1.0  # 初始的ε值
    epsilon_decay = 0.9  # ε的衰减率
    i = 0
    if i < loop:
        epsilon *= epsilon_decay
    if np.random.uniform(0, 1) < epsilon:
        action = np.random.choice([0, 1])  # 随机探索动作
    else:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action_probs = policy_net(state_tensor)
        action_probs = torch.softmax(action_probs / 1.0, dim=-1)
        action = torch.argmax(action_probs, dim=1).item()
    return action


# 定义训练策略网络的函数
def train_policy(episodes):
    #eposodes为训练次数
    replay_buffer = ReplayBuffer(50000)  # 经验重放缓存
    # 评估模型的表现，计算平均奖励
    best_avg_reward = 0
    total_rewards = 0  # 总奖励初始化为0
    avg_reward_history = []
    no_improvement_count = 0
    rewards_=[]
    """
       训练策略网络。

       参数:
       episodes -- 训练的回合数
       policy_net -- 策略网络模型
       env -- Gym环境
       device -- 运行模型的设备（如CPU或GPU）
       """
    epsilon_start = 1.0  # 初始探索概率
    epsilon_end = 0.01  # 最终探索概率
    epsilon_decay = 1000  # ε衰减周期
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
            action = select_action(state,episodes)
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-1. * i / epsilon_decay)
            action = epsilon_greedy_policy(state, policy_net, epsilon, device)
            # 采取动作，返回新的状态、奖励、是否结束（done），以及其他信息
            next_state, reward, terminated, truncated, _ = env.step(action)
            if not isinstance(terminated, (bool, np.bool_)):
                terminated = bool(terminated)
            if not isinstance(truncated, (bool, np.bool_)):
                truncated = bool(truncated)
            # 新版本的gym将done拆分为terminated和truncated，需要合并判断回合是否结束
            done = terminated or truncated

            # 从经验重放中采样并更新策略
            if replay_buffer.size() >= 256:
                batch = replay_buffer.sample(256)
                states, actions, rewards, next_states, dones = zip(*batch)
                states = torch.tensor(states, dtype=torch.float32).to(device)
                actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(device)  # 将actions转换为张量并增加一个维度
                rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
                next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
                dones = torch.tensor(dones, dtype=torch.float32).to(device)

                # 前向传播
                action_probs = policy_net(states)
                action_log_probs = torch.log(action_probs.gather(1, actions))  # 使用gather来选择动作概率

                # 计算损失
                loss = -torch.sum(action_log_probs * rewards) / 64

                # 反向传播
                scheduler.step()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # 存储状态、动作和奖励
            # print(f"State: {state}")
            if np.any(np.isnan(state)) or np.any(np.isinf(state)):
                raise ValueError("Invalid state with NaN or Inf encountered.")

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            if abs(next_state[2]) < 0.05:  # 杆子的角度接近竖直
                reward += 1.0  # 给予额外奖励
            # 累积本回合的奖励
            episode_reward += reward

            # 将状态更新为下一步的状态
            state = next_state
            scheduler.step()
        # 累积所有回合的奖励
        rewards_.append(episode_reward)
        total_rewards += episode_reward
        avg_reward = total_rewards / len(rewards_)  # 计算平均奖励
        avg_reward_history.append(avg_reward)
        print(f'Episode {i + 1}={episode_reward}, Total Reward: {total_rewards}, {len(rewards)}Average Reward: {avg_reward}')
        if no_improvement_count >= 500:
            epsilon = 0.5  # 重新提高探索率进行更多的探索
        if i % 100 == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # 检查是否收敛
        try:
            with open('max_value.json', 'r') as f:
                best_avg_reward= json.load(f)
        except FileNotFoundError:
            return None  # 文件不存在时返回None
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            no_improvement_count = 0  # 重置计数
            with open('max_value.json', 'w') as f:
                json.dump(best_avg_reward, f)
            # 保存模型
            torch.save(policy_net.state_dict(), 'policy_net_model_.pth')
            print(f'Model saved with average reward: {avg_reward}')
        else:
            no_improvement_count += 1

        # 如果在patience轮次内没有改进，停止训练
        if no_improvement_count >= 2000:
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

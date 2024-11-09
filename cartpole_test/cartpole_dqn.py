import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt
import pygame


# 创建CartPole环境
# env = gym.make('CartPole-v1',)
# if not hasattr(np, 'bool8'):
#     np.bool8 = bool  # 如果 numpy.bool8 不存在，则将其设为标准的 bool 类型

# 定义Q网络
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        # 定义全连接层
        self.fc1 = nn.Linear(input_dim, 128)  # 第一层，输入维度到128
        self.fc2 = nn.Linear(128, 128)  # 第二层，128到128
        self.fc3 = nn.Linear(128, output_dim)  # 第三层，128到输出维度

    def forward(self, x):
        # 定义前向传播过程
        x = torch.relu(self.fc1(x))  # ReLU 激活函数
        x = torch.relu(self.fc2(x))  # ReLU 激活函数
        return self.fc3(x)  # 输出层


# 经验重放缓冲区类
class ReplayBuffer:
    def __init__(self, capacity, important_ratio=0.7):
        # 使用两个缓冲区：普通缓冲区和重要缓冲区
        self.normal_buffer = deque(maxlen=int(capacity * (1 - important_ratio)))  # 普通缓冲区
        self.important_buffer = deque(maxlen=int(capacity * important_ratio))  # 重要缓冲区
        self.important_ratio = important_ratio  # 重要经验的比例

    def add(self, experience, important=False):
        """
        添加经验到缓冲区
        :param experience: 经验数据 (state, action, reward, next_state, done)
        :param important: 是否为重要经验
        """
        if important:
            self.important_buffer.append(experience)  # 将重要经验添加到重要缓冲区
        else:
            self.normal_buffer.append(experience)  # 将普通经验添加到普通缓冲区

    def sample(self, batch_size):
        # 计算需要从重要缓冲区和普通缓冲区采样的数量
        important_size = int(batch_size * self.important_ratio)
        normal_size = batch_size - important_size

        # 从两个缓冲区按比例采样
        important_samples = random.sample(self.important_buffer, min(len(self.important_buffer), important_size))
        normal_samples = random.sample(self.normal_buffer, min(len(self.normal_buffer), normal_size))

        # 合并采样结果
        batch = important_samples + normal_samples
        random.shuffle(batch)  # 混合顺序

        # 解包并返回
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.vstack(states), actions, rewards, np.vstack(next_states), dones

    def size(self):
        # 返回缓冲区中存储的总经验数
        return len(self.normal_buffer) + len(self.important_buffer)


# ε-greedy策略
def epsilon_greedy_policy(state, epsilon, action_space, q_network, device):
    if random.random() < epsilon:  # 以ε的概率选择随机动作
        return action_space.sample()
    else:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)  # 将状态转为Tensor并转到设备
        q_values = q_network(state)  # 计算当前状态的所有动作的Q值
        return q_values.argmax().item()  # 选择Q值最大的动作


# DQN训练函数
def train_dqn(env, episodes, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=1000,
              lr=1e-3, buffer_size=10000, batch_size=64, target_update=10, load_model=True,
              reward_threshold=300.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 选择设备（GPU或CPU）
    input_dim = env.observation_space.shape[0]  # 获取输入维度（状态的维度）
    output_dim = env.action_space.n  # 获取输出维度（动作空间的大小）
    q_network = DQN(input_dim, output_dim).to(device)  # 创建Q网络模型

    # 检查是否需要加载已保存的模型
    if load_model:
        try:
            q_network = torch.load('dqn_cartpole.pth')  # 加载保存的模型
            print("模型加载成功，从现有模型继续训练。")
        except FileNotFoundError:
            print("未找到模型文件，将从头开始训练。")
    else:
        print("不加载模型文件，将从头开始训练。")

    target_q_network = DQN(input_dim, output_dim).to(device)  # 创建目标Q网络
    target_q_network.load_state_dict(q_network.state_dict())  # 将Q网络的权重复制到目标网络

    optimizer = optim.Adam(q_network.parameters(), lr=lr)  # 使用Adam优化器
    replay_buffer = ReplayBuffer(buffer_size)  # 创建经验重放缓冲区
    epsilon = epsilon_start  # 初始化ε值
    total_rewards = []  # 存储每一回合的奖励
    best_avg_reward = -float('inf')  # 最好的平均奖励（用于提前停止）

    for episode in range(episodes):
        state = env.reset()[0]  # 初始化环境并获取初始状态
        episode_reward = 0  # 初始化当前回合的奖励
        done = False

        while not done:
            action = epsilon_greedy_policy(state, epsilon, env.action_space, q_network, device)  # 选择动作
            # 采取动作，返回新的状态、奖励、是否结束（done），以及其他信息
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated  # 判断回合是否结束

            # 根据奖励判断是否为重要经验
            is_important = reward >= reward_threshold
            replay_buffer.add((state, action, reward, next_state, done), important=is_important)  # 将经验添加到缓冲区

            state = next_state
            episode_reward += reward

            # 如果缓冲区中有足够的经验，则开始训练
            if replay_buffer.size() >= batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

                states = torch.FloatTensor(states).to(device)
                actions = torch.LongTensor(actions).unsqueeze(1).to(device)
                rewards = torch.FloatTensor(rewards).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                dones = torch.FloatTensor(dones).to(device)

                # 当前Q值
                q_values = q_network(states).gather(1, actions)

                # 目标Q值
                next_q_values = target_q_network(next_states).max(1)[0].detach()
                target_q_values = rewards + gamma * next_q_values * (1 - dones)

                loss = nn.MSELoss()(q_values, target_q_values.unsqueeze(1))  # 计算损失

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_rewards.append(episode_reward)

        # 更新目标Q网络
        if episode % target_update == 0:
            target_q_network.load_state_dict(q_network.state_dict())

        # 衰减 ε
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-1. * episode / epsilon_decay)

        # 输出训练进度
        avg_reward = np.mean(total_rewards[-100:])
        print(f'Episode {episode}, Avg Reward: {avg_reward:.2f}, Reward: {episode_reward:.2f}')

        if episode > 500:
            torch.save(q_network, 'dqn_cartpole.pth')  # 保存模型

        if avg_reward >= 475:  # 如果平均奖励足够高，提前停止训练
            print(f'Solved in episode {episode} with average reward {avg_reward:.2f}!')
            break

    return total_rewards


# 定义模型测试函数
def test_dqn(env, episodes=10, render=False):
    # 加载已保存的模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_network = torch.load('dqn_cartpole.pth', weights_only=False).to(device)
    q_network.eval()  # 切换到评估模式

    total_rewards = []
    for episode in range(episodes):
        state = env.reset()[0]
        done = False
        episode_reward = 0

        while not done:
            if render:
                env.render()  # 可视化环境

            # 选择动作
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action = q_network(state_tensor).argmax().item()  # 选择Q值最大的动作

            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            episode_reward += reward

        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward}")

    avg_reward = np.mean(total_rewards[-100:])
    print(f"Average reward over {episodes} episodes: {avg_reward}")
    env.close()
    return total_rewards


if __name__ == "__main__":
    env = gym.make('CartPole-v1',render_mode = 'human')

    # Choose whether to train or test the model
    is_training = False  # Set to True for training, False for testing

    if is_training:
        # Train the model and plot results
        rewards = train_dqn(env, 10000)
        plot_title = 'DQN Training Rewards on CartPole-v1'
        file_name = 'dqn_cartpole_rewards_train.png'  # Save as training image
    else:
        # Test the model and plot results
        rewards = test_dqn(env, 100)
        plot_title = 'DQN Testing Rewards on CartPole-v1'
        file_name = 'dqn_cartpole_rewards_test.png'  # Save as testing image

    # Plot the rewards
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(plot_title)
    # Save the plot with the appropriate file name
    plt.savefig(file_name)
    plt.show()
    env.close()  # 关闭环境


# 导入必要的库
from collections import deque  # 用于创建一个双端队列，用于经验回放
from operator import truediv  # 用于数学运算，这里未使用
from numpy.random import choice  # 用于随机数生成，用于环境交互时的探索
from numpy import random
import matplotlib.pyplot as plt  # 用于绘制训练过程中的奖励曲线
import gym  # 导入gym库，用于创建和管理强化学习环境
import torch  # 导入PyTorch库，用于构建和训练神经网络
import torch.nn as nn  # 导入PyTorch的神经网络模块
import torch.optim as optim  # 导入PyTorch的优化器模块
import numpy as np  # 导入NumPy库，用于数值计算

# 创建CartPole环境
env = gym.make('CartPole-v1')  # 使用gym库创建CartPole环境

# 修复numpy bool8的问题，防止某些版本的gym依赖numpy.bool8引发错误
if not hasattr(np, 'bool8'):
    np.bool8 = bool  # 如果numpy.bool8不存在，则将其设为标准的bool类型

# 定义经验回放缓冲区类
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)  # 初始化一个双端队列，最大长度为capacity

    def add(self, experience):
        self.buffer.append(experience)  # 将经验添加到缓冲区

    def sample(self, batch_size):
        indices = choice(range(len(self.buffer)), size=batch_size, replace=False)
        return [self.buffer[i] for i in indices]  # 从缓冲区随机采样batch_size个经验

    def size(self):
        return len(self.buffer)  # 返回缓冲区中的经验数量

# 定义策略网络类，继承自PyTorch的nn.Module基类
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()  # 调用父类的初始化方法
        self.fc1 = nn.Linear(input_dim, 64)  # 定义第一个全连接层，输入维度为input_dim，输出维度为64
        self.fc2 = nn.Linear(64, output_dim)  # 定义第二个全连接层，输入维度为64，输出维度为output_dim

    def forward(self, state):
        x = torch.relu(self.fc1(state))  # 将输入状态通过第一个全连接层，并应用ReLU激活函数
        action_probs = torch.softmax(self.fc2(x), dim=-1)  # 将经过ReLU激活的输出通过第二个全连接层，并应用softmax函数，得到动作概率分布
        return action_probs  # 返回动作概率分布

# 获取输入状态维度和动作空间维度
input_dim = env.observation_space.shape[0]  # 状态空间维度
output_dim = env.action_space.n  # 动作空间大小

# 创建策略网络实例
policy_net = PolicyNetwork(input_dim, output_dim)  # 使用输入维度和输出维度创建策略网络实例
policy_net.load_state_dict(torch.load('policy_net_model.pth',weights_only=True))
# 使用Adam优化器
optimizer = optim.Adam(policy_net.parameters(), lr=1e-2)  # 使用Adam优化器更新策略网络的参数，学习率设置为1e-4

# 选择动作的函数
def select_action(state):
    state_tensor = torch.FloatTensor(state).unsqueeze(0)  # 将状态转换为PyTorch的Tensor，并扩展维度以符合输入要求
    action_probs = policy_net(state_tensor)  # 使用策略网络计算动作概率
    action = torch.multinomial(action_probs, num_samples=1).item()
    # action = torch.argmax(action_probs, dim=1).item()  # 选取概率最高的动作
    return action  # 返回选取的动作

# 定义训练策略网络的函数
def train_policy(episodes):
    replay_buffer = ReplayBuffer(10000)  # 创建经验回放缓冲区实例，容量为10000
    best_avg_reward = -float('inf')  # 初始化最佳平均奖励为负无穷
    total_rewards = 0  # 初始化总奖励为0
    avg_reward_history = []  # 用于记录每轮的平均奖励
    no_improvement_count = 0  # 用于记录连续多少轮没有改进
    rewards_ = []  # 用于记录每轮的奖励

    for i in range(episodes):
        state, _ = env.reset()  # 重置环境，获取初始状态
        done = False  # 初始化回合结束标志为False
        episode_reward = 0  # 初始化当前回合的奖励为0
        rewards = []  # 记录奖励
        states = []  # 记录状态
        actions = []  # 记录动作

        while not done:
            if random.random() < 0.1:
                action = env.action_space.sample()  # 以一定的概率随机选择动作，用于探索
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action = select_action(state)  # 选择动作

            next_state, reward, terminated, truncated, _ = env.step(action)  # 采取动作，返回新的状态、奖励、是否结束（done），以及其他信息
            done = terminated or truncated  # 新版本的gym将done拆分为terminated和truncated，需要合并判断回合是否结束

            replay_buffer.add((state, action, reward, next_state, done))  # 将经验添加到经验回放缓冲区

            states.append(state)  # 记录状态
            actions.append(action)  # 记录动作
            rewards.append(reward)  # 记录奖励
            episode_reward += reward  # 累积本回合的奖励
            state = next_state  # 将状态更新为下一步的状态

        rewards_.append(episode_reward)  # 记录每轮的奖励
        total_rewards += episode_reward  # 累积总奖励
        avg_reward = total_rewards / (i + 1)  # 计算平均奖励

        # 如果经验回放缓冲区中的经验数量大于等于32，则从缓冲区中采样经验进行训练
        if replay_buffer.size() >= 32:
            experiences = replay_buffer.sample(32)  # 从缓冲区随机采样32个经验
            states, actions, rewards, next_states, dones = zip(*experiences)  # 解压采样的经验
            states = torch.FloatTensor(states)  # 将状态转换为Tensor
            actions = torch.LongTensor(actions)  # 将动作转换为Tensor
            rewards = torch.FloatTensor(rewards)  # 将奖励转换为Tensor
            next_states = torch.FloatTensor(next_states)  # 将下一个状态转换为Tensor
            dones = torch.FloatTensor(dones)  # 将结束标志转换为Tensor

            # 计算策略网络的损失并进行优化
            action_probs = policy_net(states)  # 使用策略网络计算动作概率
            action_probs = action_probs.gather(1, actions.unsqueeze(1)).squeeze()  # 选择采取的动作的概率
            loss = -torch.mean(torch.log(action_probs) * rewards)  # 计算损失
            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

        avg_reward_history.append(avg_reward)  # 记录平均奖励
        print(f'Episode {i + 1}={episode_reward}, Total Reward: {total_rewards}, Average Reward: {avg_reward}')  # 打印每轮的信息

        # 检查是否收敛
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            no_improvement_count = 0  # 重置计数
            torch.save(policy_net.state_dict(), 'policy_net_model.pth')  # 保存模型
            print(f'Model saved with average reward: {avg_reward}')  # 打印保存模型的信息
        else:
            no_improvement_count += 1  # 增加没有改进的轮数

        # 如果在500轮内没有改进，则停止训练
        if no_improvement_count >= 5000:
            print("No improvement in average reward. Stopping training.")  # 打印停止训练的信息
            break

    return rewards_  # 返回每轮的奖励

episodes = 10000  # 设置训练轮数
qq = train_policy(episodes)  # 训练策略网络

# 关闭环境
env.close()  # 关闭gym环境

# 保存模型权重到文件中，以便后续加载或评估
torch.save(policy_net.state_dict(), 'policy_net_model.pth')

# 绘制奖励图
plt.figure(figsize=(12, 6))  # 设置绘图大小
plt.plot(qq, label='Episode Rewards')  # 绘制每轮的奖励
plt.xlabel('Episode')  # 设置x轴标签
plt.ylabel('Total Reward')  # 设置y轴标签
plt.title('Total Reward per Episode')  # 设置图表标题
plt.legend()  # 显示图例
plt.grid()  # 显示网格
plt.show()  # 显示图表
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

env = gym.make('CartPole-v1')
# 修复 numpy bool8 的问题
if not hasattr(np, 'bool8'):
    np.bool8 = bool  # 将不存在的 numpy.bool8 设为标准 bool
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
        # 得到动作概率分布，dim=-1表示在最后一个维度上应用softmax
        action_probs = torch.softmax(self.fc2(x), dim=-1)
        # 返回动作概率分布
        return action_probs

input_dim = env.observation_space.shape[0]  # 状态空间维度
output_dim = env.action_space.n  # 动作空间大小

policy_net = PolicyNetwork(input_dim, output_dim)
optimizer = optim.Adam(policy_net.parameters(), lr=1e-2)

def evaluate_policy(policy_net, env, episodes=10):
    #评估模型
    total_rewards = 0
    for i in range(episodes):
        # state = env.reset()
        done = False
        state, _ = env.reset()
        episode_reward = 0
        while not done:
            # print(f"State: {state}")  # 调试用，查看state的内容
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_probs = policy_net(state_tensor)
            action = torch.argmax(action_probs).item()
            # next_state, reward, done, _ = env.step(action)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated  # 合并 terminated 和 truncated 以确定回合是否结束

            episode_reward += reward

            state = next_state
        total_rewards += episode_reward

    average_reward = total_rewards / episodes
    return average_reward
episodes=10
# 使用上文定义的PolicyNetwork和初始化的env
average_reward = evaluate_policy(policy_net, env)
print(f"Average reward over {episodes} episodes: {average_reward}")

# 保存模型
torch.save(policy_net.state_dict(), 'policy_net_model.pth')

# # 加载模型
# loaded_policy_net = PolicyNetwork(input_dim, output_dim)
# loaded_policy_net.load_state_dict(torch.load('policy_net_model.pth'))

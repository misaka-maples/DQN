
def train_policy(episodes, convergence_threshold=500, patience=10):
    best_avg_reward = -float('inf')
    avg_reward_history = []
    no_improvement_count = 0

    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        rewards = []
        states = []
        actions = []

        while not done:
            action = select_action(state)
            next_state, reward, done, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            episode_reward += reward
            state = next_state

        total_reward = sum(rewards)
        avg_reward = total_reward / len(rewards)  # 计算平均奖励

        # 更新策略网络
        discounted_rewards = []
        for t in range(len(rewards)):
            Gt = sum(rewards[t:])
            discounted_rewards.append(Gt)

        discounted_rewards_tensor = torch.FloatTensor(np.array(discounted_rewards))
        actions_tensor = torch.LongTensor(np.array(actions))

        action_probs = policy_net(torch.FloatTensor(np.array(states)))
        action_probs = action_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze()
        loss = -torch.mean(torch.log(action_probs) * discounted_rewards_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_reward_history.append(avg_reward)
        print(f'Episode {episode + 1}, Total Reward: {total_reward}, Average Reward: {avg_reward}')

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
        if no_improvement_count >= patience:
            print("No improvement in average reward. Stopping training.")
            break
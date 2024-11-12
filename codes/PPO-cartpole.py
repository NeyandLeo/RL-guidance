import gym
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.policy_net = PolicyNetwork(state_dim, action_dim)
        self.value_net = ValueNetwork(state_dim)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=1e-3)
        self.memory = []

        # PPO参数
        self.clip_param = 0.1
        self.gamma = 0.99
        self.batch_size = 64
        self.update_steps = 10

    def select_action(self, state):
        state = torch.FloatTensor(state)
        with torch.no_grad():
            probs = self.policy_net(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(),dist.probs[action.item()].item()

    def store_transition(self, state,prob, action, reward, next_state, done):
        self.memory.append((state,prob, action, reward, next_state, done))

    def update(self):
        states,old_probs, actions, rewards, next_states, dones = zip(*self.memory)
        old_probs = torch.FloatTensor(old_probs).unsqueeze(1)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # 计算优势函数
        advantages, returns = self.compute_advantages(rewards, dones, states, next_states)

        for _ in range(self.update_steps):
            # 计算新策略的概率
            probs = self.policy_net(states).gather(1, actions.unsqueeze(1))
            # 计算概率比率
            ratios = (probs / old_probs).squeeze()

            # 计算熵损失
            entropy_loss = (probs * torch.log(probs)).mean()

            # 计算策略损失
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_param, 1 + self.clip_param) * advantages
            policy_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy_loss

            # 计算价值损失
            value_loss = nn.MSELoss()(self.value_net(states).squeeze(), returns)

            # 更新策略网络
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            # 更新价值网络
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

        # 清空记忆
        self.memory = []

    def compute_advantages(self, rewards, dones, states, next_states):
        # 计算目标价值
        with torch.no_grad():
            values = self.value_net(states).squeeze() #计算V(s)
            next_values = self.value_net(next_states).squeeze() #计算V(s')
            td_targets = rewards + self.gamma * next_values * (1 - dones) #计算TD目标，即Q(s,a)=r+γV(s')
            advantages = td_targets - values #计算优势函数，即A(s,a)=Q(s,a)-V(s)
        returns = td_targets
        return advantages.detach(), returns.detach()

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)

class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.fc(x)
def main():
    env = gym.make('CartPole-v0')
    agent = PPOAgent(state_dim=4, action_dim=2)
    max_episodes = 500
    reward_history = []

    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # 环境渲染（可选）
            env.render()

            action,prob = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.store_transition(state,prob, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            if done:
                agent.update()
                reward_history.append(episode_reward)
                print(f"Episode {episode + 1}: Total Reward = {episode_reward} Episode Length = {len(reward_history)}")
                break

        # 可视化累积奖励
        if (episode + 1) % 1000 == 0:
            plt.plot(reward_history)
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.title('Training Progress')
            plt.show()

    env.close()

if __name__ == "__main__":
    main()


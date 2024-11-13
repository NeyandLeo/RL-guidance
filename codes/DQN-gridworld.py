import numpy as np
import random
from collections import deque
# 导入必要的深度学习库
import torch
import torch.nn as nn
import torch.optim as optim

#environment
class GridWorld:
    def __init__(self):
        self.size = 5
        self.reset()
        self.actions = [0, 1, 2, 3]  # 上，下，左，右

    def reset(self):
        self.grid = np.zeros((self.size, self.size))
        self.position = [0, 0]
        self.goal = [4, 4]
        self.obstacles = [[1,1], [2,2], [3,3]]
        for obs in self.obstacles:
            self.grid[obs[0], obs[1]] = -1
        self.grid[self.goal[0], self.goal[1]] = 1
        return self.position

    def step(self, action):
        x, y = self.position
        if action == 0 and x > 0:
            x -= 1  # 上
        elif action == 1 and x < self.size -1:
            x += 1  # 下
        elif action == 2 and y > 0:
            y -= 1  # 左
        elif action == 3 and y < self.size -1:
            y += 1  # 右

        self.position = [x, y]

        reward = -1
        done = False

        if self.position == self.goal:
            reward = 10
            done = True
        elif self.position in self.obstacles:
            reward = -10
            done = True

        return self.position, reward, done

    def render(self):
        grid = np.zeros((self.size, self.size), dtype=str)
        grid[:] = '.'
        for obs in self.obstacles:
            grid[obs[0], obs[1]] = 'X'
        grid[self.goal[0], self.goal[1]] = 'G'
        grid[self.position[0], self.position[1]] = 'A'
        print('\n'.join(' '.join(row) for row in grid))
        print()

class duelingNet(nn.Module):
    def __init__(self,in_features=2,out_features=4):
        super(duelingNet, self).__init__()
        self.fc1 = nn.Linear(in_features, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, 12)
        self.fc4 = nn.Linear(12,out_features)
        self.fc5 = nn.Linear(24, 12)
        self.fc6 = nn.Linear(12, 1)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x1 = torch.relu(self.fc3(x))
        x2 = torch.relu(self.fc5(x))
        x1 = self.fc4(x1)
        x2 = self.fc6(x2)
        x = x1 + x2
        return x


#DQN Agent
class DQNAgent:
    def __init__(self,type=1):
        self.memory = deque(maxlen=5000)
        self.gamma = 0.95    # 折扣因子
        self.epsilon = 1.0   # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0005
        if type == 1:
            self.model = self._build_model()
            self.targetnet = self._build_model()
        elif type == 2:
            self.model = self._build_dueling_model()
            self.targetnet = self._build_dueling_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.update_target()  # 将主网络的参数复制到目标网络
        # 冻结目标网络的参数
        for param in self.targetnet.parameters():
            param.requires_grad = False

    def _build_model(self):
        # 请在此处补全神经网络模型的代码
        # 提示：使用PyTorch构建一个简单的全连接网络
        model = nn.Sequential(
            nn.Linear(2, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, 4)
        )
        return model

    def _build_dueling_model(self):
        return duelingNet()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(4)
        state = torch.FloatTensor(state)
        act_values = self.model(state)
        return torch.argmax(act_values).item()

    def act_greedy(self, state):
        state = torch.FloatTensor(state)
        act_values = self.model(state)
        return torch.argmax(act_values).item()

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = torch.FloatTensor(next_state)
                target = (reward + self.gamma *
                          torch.max(self.targetnet(next_state)).item()) # R(s,a) + γmaxQ(s',a')
            state = torch.FloatTensor(state)
            target_f = self.model(state) # Q(s,a)
            target_f = target_f.detach().clone()
            target_f[action] = target # Q(s,a) = R(s,a) + γmaxQ(s',a')

            # 请在此处补全训练过程的代码
            # 提示：计算损失并执行反向传播
            self.optimizer.zero_grad()
            output = self.model(state)
            #注意：output与target_f的其他分量相同，只有target_f[action]不同,且差值即为(R(s,a) + γmaxQ(s',a') - Q(s,a))^2
            loss = self.criterion(output, target_f)
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target(self):
        self.targetnet.load_state_dict(self.model.state_dict())

if __name__ == "__main__":
    env = GridWorld()
    type = int(input("Please input the type of network you want to use: 1 for simple network, 2 for dueling network"))
    agent = DQNAgent(type)
    episodes = 100
    update_target_freq = 5  # 每隔10个回合更新一次目标网络

    for e in range(episodes):
        state = env.reset()
        state = np.array(state)
        re = 0
        for time_t in range(100):
            # env.render()
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            re+=reward
            next_state = np.array(next_state)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print(f"episode: {e}/{episodes}, time: {time_t}, e: {agent.epsilon:.2}, reward: {re}")
                break
            if len(agent.memory) > 128:
                agent.replay(128)
        if e % update_target_freq == 0:
            agent.update_target()
            print("target_net updated")

    # 测试模型
    state = env.reset()
    state = np.array(state)
    for _ in range(100):
        env.render()
        action = agent.act_greedy(state) #using greedy strategy instead of epsilon greedy strategy when testing
        next_state, reward, done = env.step(action)
        next_state = np.array(next_state)
        state = next_state
        if done:
            break
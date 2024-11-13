# GridWorld DQN Implementation

## Introduction

This project implements a simple reinforcement learning example based on the DQN (Deep Q-Network) algorithm, named **GridWorld**. In this 5x5 grid world, the agent needs to move from the starting point to the goal while avoiding obstacles. Through training, the agent will learn to reach the target location via the optimal path.

## Algorithm Principles

### Reinforcement Learning

Reinforcement learning is a type of machine learning where an agent learns a policy by interacting with the environment to maximize cumulative rewards. At each time step, the agent takes an action, and the environment returns a new state and a reward.

### Q-Learning

Q-Learning is a value-based reinforcement learning algorithm that estimates the value (Q-value) of each state-action pair. The update formula is:

Q(s, a) = Q(s, a) + α * [r + γ * max Q(s', a') - Q(s, a)]

where:

- `Q(s, a)`: The estimated value of taking action `a` in state `s`, provided by the Q-network.
- `max Q(s', a')`: The maximum Q-value among all possible actions in the new state `s'`, representing the optimal expected value for the next step, given by the target network.
- `α`: The learning rate, which controls the step size of updates.
- `r`: Immediate reward.
- `γ`: Discount factor, balancing the impact of immediate rewards and future rewards.
- `s'`: The new state reached after taking action `a`.
- `a'`: Possible actions in the new state `s'`.

### Deep Q-Network (DQN)

DQN uses a neural network to approximate the Q-value function, enabling it to handle high-dimensional, continuous state spaces. Key features include:

- **Neural Network Approximation**: Uses a neural network to estimate Q-values.
- **Experience Replay**: Stores experiences (state, action, reward, next state) in a replay buffer, and samples randomly during training to break data correlation.
- **Target Network**: A target network is used to calculate the target Q-values, which is updated periodically to increase training stability.

## Implementation Details

### Environment Design

- **Grid Size**: 5x5.
- **Starting Point**: Coordinates (0, 0).
- **Goal**: Coordinates (4, 4).
- **Obstacles**: Located at (1,1), (2,2), (3,3).
- **Actions**: Up (0), Down (1), Left (2), Right (3).
- **Rewards**:
  - Each move: -0.1.
  - Reaching the goal: +1.
  - Hitting an obstacle: -1, and the episode ends.

### Agent Design
#### Neural Network Design
Two neural network architectures are used to estimate Q(s, a). One is a simple neural network, fully connected:

- **Simple Neural Network Architecture**:
  - Input layer: 2 nodes (the agent's position coordinates).
  - Hidden layers: Two fully connected layers with 24 neurons each, ReLU activation function.
  - Output layer: 4 nodes (Q-values for 4 actions).

Another structure uses Dueling DQN, decomposing Q(s, a) into V(s) and A(s, a), representing the state value and action advantage, respectively:

- **Dueling DQN Structure**:
  - Input layer: 2 nodes (the agent's position coordinates).
  - Hidden layers: Two fully connected layers with 24 neurons each, ReLU activation function.
  - Output layer: Two fully connected layers that output state value V(s) and action advantage A(s, a). Q-value is calculated by Q(s,a) = V(s) + A(s,a).

- **Policy**: ε-greedy policy, where with probability ε a random action is chosen, and with probability 1 - ε the current optimal action is chosen. ε decreases gradually during training.

- **Hyperparameters**:
  - Learning rate: 0.0005
  - Discount factor γ: 0.95
  - Exploration rate ε: initial value 1.0, minimum value 0.01, decay rate 0.995
  - Replay buffer size: 5000
  - Batch size: 128

### Training Process

1. **Initialize the environment and agent**.
2. **Repeat the following steps until the specified number of episodes is reached**:
   - Reset the environment and obtain the initial state.
   - At each time step:
     - Choose an action based on the current state.
     - Execute the action and receive the next state, reward, and whether it’s done.
     - Store the experience in the replay buffer.
     - When the replay buffer has sufficient experiences, start training:
       - Sample a batch of experiences from the replay buffer.
       - Compute target Q-values:
         - If it’s a terminal state, the target Q-value equals the immediate reward `r`.
         - If it’s not a terminal state, the target Q-value equals `r + γ * max Q_target(s', a')`.
       - Compute the loss between predicted Q-values and target Q-values (mean squared error).
       - Backpropagate to update neural network parameters.
     - Update the current state.
     - If a terminal state is reached, exit the loop.
   - Decrease the exploration rate ε.
   - Periodically update the target network by copying parameters from the main network.

### Additional Notes
During training, Q-network training may exhibit significant fluctuations due to large target network updates causing substantial policy changes. To mitigate this, it is recommended to reduce the target network update frequency, e.g., updating every 10 episodes, use a smaller learning rate, such as 0.0001, or apply a soft update strategy for the target network. For soft updates, only a portion of the parameters, e.g., 0.1, are updated each time (not implemented here; you may implement it as needed).

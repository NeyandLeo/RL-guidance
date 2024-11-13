# Introduction to Proximal Policy Optimization (PPO) and its Implementation in the CartPole Environment

## Table of Contents

- [Introduction](#Introduction)
- [Basic Principles of the PPO Algorithm](#Basic-Principles-of-the-PPO-Algorithm)
  - [Policy Gradient Methods](#Policy-Gradient-Methods)
  - [Clipped Objective](#Clipped-Objective)
  - [Advantage Estimation](#Advantage-Estimation)
- [Implementation Details of PPO in the CartPole Environment](#Implementation-Details-of-PPO-in-the-CartPole-Environment)
  - [Environment Introduction](#Environment-Introduction)
  - [Network Architecture](#Network-Architecture)
  - [Training Process](#Training-Process)
  - [Hyperparameter Settings](#Hyperparameter-Settings)
- [Conclusion](#Conclusion)

## Introduction

Proximal Policy Optimization (PPO) is an advanced reinforcement learning algorithm introduced by OpenAI. Building upon policy gradient methods, PPO enhances training stability and efficiency by introducing a clipping mechanism. PPO performs exceptionally well across various control tasks and is particularly well-suited for classic control environments like CartPole.

## Basic Principles of the PPO Algorithm

### Policy Gradient Methods

Policy gradient methods directly optimize the policy function so that actions taken in a given state yield a higher expected return. PPO belongs to this category of methods, updating policy parameters by maximizing the following objective function:

$$
L(\theta) = \mathbb{E}\left[\frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)} A^{\pi_{\theta_{\text{old}}}}(s, a)\right]
$$

where $\pi_\theta(a|s)$ is the current policy, $\pi_{\theta_{\text{old}}}(a|s)$ is the old policy, and $A$ is the advantage function.

### Clipped Objective

To prevent excessive policy updates, PPO introduces a clipping mechanism that limits the range of policy ratio changes. The clipped objective function is defined as:

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}\left[\min\left(r(\theta) A, \text{clip}(r(\theta), 1 - \epsilon, 1 + \epsilon) A\right)\right]
$$

where $r(\theta) = \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}$, and $\epsilon$ is a small hyperparameter (e.g., 0.2).

### Entropy Loss Term

Additionally, to encourage exploration, an entropy loss term is included. By increasing the target entropy, we explicitly enhance the probability of the policy exploring other actions. Specifically, we update the policy parameters by maximizing the following objective function:

$$
L^{\text{S}}(\theta) = \mathbb{E}\left[ H(\pi_{\theta_{\text{}}}(·|s_{t})) \right]
$$

where $\pi_\theta(·|s)$ is the current policy.

### Advantage Estimation

The advantage function $A(s, a)$ is used to evaluate the effectiveness of an action. Common methods include Generalized Advantage Estimation (GAE), which combines temporal difference error and Monte Carlo estimation to provide a lower-variance advantage estimate. For detailed proofs, refer to relevant literature. Here is the conclusion: when calculating gradients, the advantage function and action value function share the same gradient, and the $V(s)$ term does not introduce additional gradients.

## Implementation Details of PPO in the CartPole Environment

### Environment Introduction

[CartPole](https://gym.openai.com/envs/CartPole-v1/) is a classic control task where the goal is to keep a pole upright by moving a cart. The environment state includes the cart's position, velocity, pole angle, and angular velocity.

### Network Architecture

PPO typically uses two neural networks:

- **Policy Network (Actor)**: Outputs the probability distribution of actions given a state.
- **Value Network (Critic)**: Estimates the value of a given state.

# DRL-Lib

A Deep Reinforcement Learning library implementing Soft Actor-Critic (SAC) and other algorithms.

## Installation

```bash
pip install drl-lib
```

## Quick Start

```python
import gymnasium as gym
from drl_lib.agents.SAC.agent_sac import SACAgent

# Create environment
env = gym.make('Pendulum-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

# Initialize SAC agent
agent = SACAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    hidden_dims=[256, 256],
    alpha=0.2,
    gamma=0.99,
    tau=0.005,
    actor_lr=3e-4,
    critic_lr=3e-4,
    value_lr=3e-4,
    buffer_size=1000000,
    batch_size=256,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    action_bound=(-2, 2)
)

# Training loop
for episode in range(1000):
    state, _ = env.reset()
    episode_reward = 0
    done = False
    
    while not done:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.store_experience(state, action, reward, next_state, done)
        agent.update()
        state = next_state
        episode_reward += reward
    
    print(f"Episode {episode}, Reward: {episode_reward}")
```

## Features

- Soft Actor-Critic (SAC) implementation
- Gradient monitoring and debugging tools
- Replay buffer for experience storage
- Support for continuous action spaces
- Automatic GPU support

## Documentation

For detailed documentation, please visit [documentation link].

## License

This project is licensed under the MIT License - see the LICENSE file for details.



# Soft Actor-Critic (SAC) Implementation

## Key Points

- SAC is a reinforcement learning algorithm for continuous action spaces, balancing exploration and exploitation via entropy.
- It uses an Actor for policy, two Q-Networks for value estimation, a Value Network for state values, and a Replay Buffer for experience.
- Mathematics includes Gaussian sampling, Q-value updates, and entropy regularization, crucial for training stability.
- Research suggests SAC performs well on benchmarks like Pendulum-v1, but results vary by environment.

## Introduction to SAC

Soft Actor-Critic (SAC) is an advanced reinforcement learning (RL) method designed for continuous control tasks, such as robotic arm movement or pendulum balancing. Imagine an agent exploring a vast landscape, needing to decide actions like how far to swing a pendulum. SAC helps by not just chasing rewards but also encouraging exploration, ensuring the agent doesn't get stuck in one place.

## Components and Mathematics

SAC's strength lies in its components, each with a mathematical backbone:

- **Actor (Policy Network)**: Outputs a Gaussian distribution, sampling actions for exploration. It uses $\mu(s)$ (mean) and $\sigma(s)$ (standard deviation), transformed by $\tanh$ for bounds.

- **Q-Networks (Critics)**: Two networks estimate $Q(s, a)$, the value of taking action $a$ in state $s$. They minimize $(Q(s, a) - y)^2$, where $y = r + \gamma (1 - d) \cdot V(s')$.

- **Value Network**: Estimates $V(s)$, the state value, updated to match $\min(Q_1, Q_2) - \alpha \log \pi(a|s)$, stabilizing training.

- **Replay Buffer**: Stores experiences $(s, a, r, s', d)$, sampled randomly for off-policy learning.

## Testing and Application

SAC shines on benchmarks like Pendulum-v1, where it learns to balance a pendulum. Training involves episodes of actions, updates, and reward tracking, showing its adaptability.

## Survey Note: Comprehensive Analysis of Soft Actor-Critic (SAC)

### Introduction and Background

Soft Actor-Critic (SAC) is a state-of-the-art reinforcement learning (RL) algorithm, particularly effective for continuous action spaces, as seen in tasks like robotic control or pendulum balancing. Developed to address the limitations of earlier methods like Deep Deterministic Policy Gradients (DDPG), SAC introduces entropy regularization to balance exploration and exploitation, enhancing stability and sample efficiency. This survey note, informed by our detailed exploration, aims to provide a thorough understanding of SAC's components, mathematics, and practical application, drawing on online references and addressing common queries.

### SAC Components and Their Roles

SAC comprises five key components, each with a specific role in the learning process:

1. **Actor (Policy Network)**: The Actor defines the policy $\pi(a|s)$, outputting a Gaussian distribution over actions to enable stochastic behavior. This is crucial for exploration, as it allows the agent to try diverse actions rather than sticking to deterministic choices. The policy is parameterized by mean $\mu(s)$ and standard deviation $\sigma(s)$, often represented as log standard deviation for numerical stability.

2. **Q-Networks (Critics)**: SAC employs two Q-Networks, $Q_1(s, a)$ and $Q_2(s, a)$, to estimate the action-value function. The use of two networks mitigates overestimation bias, a common issue in Q-learning, by taking the minimum of their predictions. This dual-critic approach enhances training stability, especially in continuous spaces.

3. **Value Network**: The Value Network estimates the state-value function $V(s)$, which represents the expected return starting from state $s$ under the current policy, including an entropy term. It plays a pivotal role in providing stable targets for the Q-Networks, reducing variance in updates and ensuring consistent learning.

4. **Replay Buffer**: Essential for off-policy learning, the Replay Buffer stores experiences $(s, a, r, s', d)$ and allows random sampling. This breaks temporal correlations, improving data efficiency and training stability by reusing past experiences.

5. **SAC Agent**: The agent integrates all components, managing environment interactions, experience storage, and network updates. It orchestrates the training loop, ensuring coordinated updates to the Actor, Critics, and Value Network, with soft updates to the target Value Network for additional stability.

### Mathematical Foundations

The mathematics of SAC is central to its operation, and we'll explore each component's equations:

#### Actor's Sampling and Log Probability:
- The Actor outputs $\mu(s)$ and $\log \sigma(s)$, computing $\sigma(s) = e^{\log \sigma(s)}$.
- Actions are sampled from $\mathcal{N}(\mu(s), \sigma(s)^2)$, then transformed: $a = \tanh(u)$, where $u \sim \mathcal{N}(\mu, \sigma)$.
- The log probability is adjusted for the $\tanh$ transformation:
  $\log \pi(a|s) = \log \pi(u|s) - \sum_{i=1}^{d} \log(1 - a_i^2 + \epsilon)$
  where $\log \pi(u|s) = -\frac{1}{2} \sum_{i=1}^{d} \left[ \left(\frac{u_i - \mu_i}{\sigma_i}\right)^2 + 2 \log \sigma_i + \log 2\pi \right]$, and $\epsilon = 10^{-6}$ prevents numerical issues.

#### Q-Networks' Update:
- Trained to minimize the Bellman error:
  $L_Q(\theta) = \mathbb{E} \left[ \left( Q_\theta(s, a) - y \right)^2 \right]$
  where the target $y = r + \gamma (1 - d) \cdot V(s')$, with $\gamma$ as the discount factor and $d$ as the done flag.

#### Value Network's Update:
- Updated to match:
  $V(s) = \mathbb{E}_{a \sim \pi} \left[ \min(Q_1(s, a), Q_2(s, a)) - \alpha \log \pi(a|s) \right]$
  This incorporates entropy regularization, controlled by $\alpha$, ensuring exploration.

#### Actor's Update:
- The Actor maximizes:
  $J(\pi) = \mathbb{E}_{s \sim \mathcal{D}, a \sim \pi} \left[ Q(s, a) - \alpha \log \pi(a|s) \right]$
  This encourages actions with high Q-values while maintaining entropy for exploration.

### Implementation Highlights

To illustrate, here are key code snippets for critical components:

#### Actor's Sampling:
```python
def sample(self, state, deterministic=False):
    mu, log_std = self.forward(state)
    std = torch.exp(log_std)
    if deterministic:
        action = torch.tanh(mu)
    else:
        dist = torch.distributions.Normal(mu, std)
        u = dist.rsample()
        action = torch.tanh(u)
        log_prob = dist.log_prob(u) - torch.log(1 - action.pow(2) + 1e-6)
    return action, log_prob
```
This shows action sampling with the reparameterization trick and log probability adjustment.

#### Q-Network Forward Pass:
```python
def forward(self, state, action):
    x = torch.cat([state, action], dim=-1)
    return self.q_net(x)
```
Demonstrates how Q-Networks process concatenated state-action pairs.

#### Replay Buffer Sampling:
```python
def sample(self, batch_size=None):
    batch_size = batch_size or self.batch_size
    indices = np.random.choice(self.size, size=batch_size, replace=False)
    return (
        torch.tensor(self.states[indices], dtype=torch.float32, device=self.device),
        torch.tensor(self.actions[indices], dtype=torch.float32, device=self.device),
        torch.tensor(self.rewards[indices], dtype=torch.float32, device=self.device),
        torch.tensor(self.next_states[indices], dtype=torch.float32, device=self.device),
        torch.tensor(self.dones[indices], dtype=torch.float32, device=self.device)
    )
```
Highlights random sampling for off-policy learning.

### Anticipated Questions and Answers

During teaching, students often ask:

1. **Why two Q-Networks?**  
   Using two reduces overestimation bias by taking the minimum, enhancing stability. It's like having two advisors to cross-check decisions.

2. **What is entropy regularization and why is it important?**  
   It encourages exploration by rewarding random actions, preventing the agent from getting stuck. It's like ensuring the agent doesn't always take the same path, fostering discovery.

3. **How does the replay buffer help in training?**  
   It stores past experiences, allowing off-policy learning. Random sampling breaks temporal correlations, stabilizing updates, like learning from a diverse history.

4. **What is the reparameterization trick and why is it used?**  
   It rewrites sampling as $u = \mu + \sigma \cdot \epsilon$, where $\epsilon \sim \mathcal{N}(0, 1)$, enabling gradient flow through $\mu$ and $\sigma$, essential for policy optimization.

### Testing and Application

To validate SAC, we tested it on Pendulum-v1, a benchmark from Gym, where the agent learns to balance a pendulum. The setup involved:

- State dimension: 3 (angle, angular velocity, etc.).
- Action dimension: 1 (torque).
- Training for 1000 episodes, monitoring cumulative rewards.

This practical application demonstrated SAC's ability to adapt, with rewards improving over time, showcasing its effectiveness in continuous control.


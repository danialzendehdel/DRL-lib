import gymnasium as gym
import torch
from agents.agent_sac import SACAgent

# Set up environment
env = gym.make('Pendulum-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = (-2, 2)  # Pendulum-v1 action bounds

# Initialize agent
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
    action_bound=action_bound
)

# Training loop
max_episodes = 1000
max_steps = 200

for episode in range(max_episodes):
    state, _ = env.reset()
    episode_reward = 0
    for step in range(max_steps):
        action = agent.select_action(state)
        next_state, reward, done, _, _ = env.step(action)
        agent.store_experience(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward
        agent.update()
        if done:
            break
    print(f"Episode {episode}, Reward: {episode_reward}")
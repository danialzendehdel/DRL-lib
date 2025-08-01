import optuna
import gymnasium as gym
import numpy as np
import torch
from drl_lib.agents.SAC.agent_sac import SACAgent

def objective(trial):
    """
    Objective function for Optuna to optimize.
    """
    # Hyperparameter search space
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    tau = trial.suggest_float("tau", 0.001, 0.1)
    gamma = trial.suggest_float("gamma", 0.9, 0.999)

    # Fixed hyperparameters
    env_id = "Pendulum-v1"
    hidden_dims = [256, 256]
    buffer_size = 100000
    batch_size = 256
    total_timesteps = 10000
    device = 'cpu'

    env = gym.make(env_id)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = (env.action_space.low, env.action_space.high)

    # Create and train the agent
    agent = SACAgent(
        state_dim=state_dim, action_dim=action_dim, hidden_dims=hidden_dims, alpha=0.2, gamma=gamma, tau=tau,
        actor_lr=learning_rate, critic_lr=learning_rate, buffer_size=buffer_size,
        batch_size=batch_size, device=device, action_bound=action_bound, use_tanh=True
    )

    # Training loop
    obs, _ = env.reset(seed=42)
    episode_rewards = []
    current_episode_reward = 0
    for step in range(total_timesteps):
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.store_experience(obs, action, reward, next_obs, done)
        if step > batch_size:
            agent.update()
        current_episode_reward += reward
        obs = next_obs
        if done:
            episode_rewards.append(current_episode_reward)
            obs, _ = env.reset()
            current_episode_reward = 0

    # Return the mean of the last 10 episode rewards
    return np.mean(episode_rewards[-10:]) if episode_rewards else -np.inf

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

import unittest
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC as SB3_SAC
from drl_lib.agents.SAC.agent_sac import SACAgent

# Helper function to set seeds for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

class TestSACComparison(unittest.TestCase):

    def _train_agent(self, agent, env, total_timesteps):
        """Helper function to train an agent and return average rewards."""
        obs, _ = env.reset(seed=42)
        episode_rewards = []
        current_episode_reward = 0

        for step in range(total_timesteps):
            if isinstance(agent, SACAgent): # Our agent
                action = agent.select_action(obs)
            else: # SB3 agent
                action, _ = agent.predict(obs, deterministic=False)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if isinstance(agent, SACAgent):
                agent.store_experience(obs, action, reward, next_obs, done)
                agent.update()
            # SB3 agent learns via `learn` method, so we will call it outside the loop.

            current_episode_reward += reward
            obs = next_obs

            if done:
                episode_rewards.append(current_episode_reward)
                obs, _ = env.reset()
                current_episode_reward = 0

        # In case the last episode doesn't finish
        if not done:
            episode_rewards.append(current_episode_reward)

        return np.mean(episode_rewards) if episode_rewards else 0

    def test_sac_learning_comparison(self):
        """
        Compare the learning performance of our SAC agent with Stable Baselines3's SAC.
        """
        set_seed(0)

        # 1. Create the environment
        env_id = "Pendulum-v1"
        env = gym.make(env_id)

        # Hyperparameters for both agents
        hidden_dims = [64, 64]
        learning_rate = 1e-3
        buffer_size = 10000
        batch_size = 64
        tau = 0.005
        gamma = 0.99
        alpha = 0.2
        total_timesteps = 2000 # Short training run for a basic check

        # 2. Instantiate our SAC agent
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_bound = (env.action_space.low, env.action_space.high)
        device = 'cpu'

        our_agent = SACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            alpha=alpha,
            gamma=gamma,
            tau=tau,
            actor_lr=learning_rate,
            critic_lr=learning_rate,
            value_lr=learning_rate,
            buffer_size=buffer_size,
            batch_size=batch_size,
            device=device,
            action_bound=action_bound,
            use_tanh=True # Default SAC behavior
        )

        # 3. Instantiate Stable Baselines3 SAC agent
        sb3_agent = SB3_SAC(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=(1, "step"),
            policy_kwargs=dict(net_arch=hidden_dims),
            verbose=0,
            seed=0,
            use_sde=False, # To keep it comparable to our non-SDE implementation
            learning_starts=batch_size # Start learning after one batch is collected
        )

        # 4. Train both agents

        # Training loop for our agent
        our_rewards = []
        obs, _ = env.reset(seed=42)
        for i in range(total_timesteps):
            action = our_agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            our_agent.store_experience(obs, action, reward, next_obs, done)
            if i > batch_size:
                our_agent.update()
            obs = next_obs
            if done:
                obs, _ = env.reset()

        # Evaluate our agent
        obs, _ = env.reset(seed=123)
        our_episode_reward = 0
        for _ in range(500):
            action = our_agent.select_action(obs, deterministic=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            our_episode_reward += reward
            obs = next_obs
            if terminated or truncated:
                break

        # Train SB3 agent
        sb3_agent.learn(total_timesteps=total_timesteps, log_interval=-1)

        # Evaluate SB3 agent
        obs, _ = env.reset(seed=123)
        sb3_episode_reward = 0
        for _ in range(500):
            action, _ = sb3_agent.predict(obs, deterministic=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            sb3_episode_reward += reward
            obs = next_obs
            if terminated or truncated:
                break

        print(f"Our SAC Agent Final Reward: {our_episode_reward}")
        print(f"SB3 SAC Agent Final Reward: {sb3_episode_reward}")

        # 5. Compare results
        # We expect the rewards to be in a similar ballpark.
        # Due to implementation differences, they won't be identical.
        # A 50% tolerance is quite loose, but reasonable for a short run.
        self.assertAlmostEqual(our_episode_reward, sb3_episode_reward, delta=abs(sb3_episode_reward) * 0.5,
                             msg="Our SAC agent's performance is not close to SB3's.")

if __name__ == '__main__':
    unittest.main()

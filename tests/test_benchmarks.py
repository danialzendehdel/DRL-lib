import gymnasium as gym
import torch
import numpy as np
from drl_lib.agents.SAC.agent_sac import SACAgent
from drl_lib.debugging.journaling import Journal
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

def run_benchmark(env_name, max_episodes=1000, max_steps=1000, seed=42):
    """
    Run SAC on a benchmark environment and return training statistics.
    
    Args:
        env_name (str): Name of the Gymnasium environment
        max_episodes (int): Maximum number of training episodes
        max_steps (int): Maximum steps per episode
        seed (int): Random seed for reproducibility
    
    Returns:
        dict: Training statistics including rewards and times
    """
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create environment
    env = gym.make(env_name)
    env.reset(seed=seed)
    
    # Get environment dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Get action bounds
    if hasattr(env.action_space, 'low') and hasattr(env.action_space, 'high'):
        action_bound = (env.action_space.low[0], env.action_space.high[0])
    else:
        action_bound = (-1, 1)  # Default bounds
    
    # Initialize Journal for monitoring
    journal = Journal(
        directory=f"logs/{env_name}",
        action_bounds=action_bound,
        experiment_name=f"sac_{env_name}"
    )
    
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
        action_bound=action_bound
    )
    
    # Training statistics
    episode_rewards = []
    episode_times = []
    best_reward = float('-inf')
    
    # Training loop
    for episode in tqdm(range(max_episodes), desc=f"Training on {env_name}"):
        state, _ = env.reset()
        episode_reward = 0
        start_time = time.time()
        
        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.store_experience(state, action, reward, next_state, done)
            agent.update()
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        # Record statistics
        episode_time = time.time() - start_time
        episode_rewards.append(episode_reward)
        episode_times.append(episode_time)
        
        # Update best reward
        if episode_reward > best_reward:
            best_reward = episode_reward
        
        # Log progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            journal.log(f"Episode {episode + 1}, Average Reward: {avg_reward:.2f}, Best Reward: {best_reward:.2f}")
    
    env.close()
    
    return {
        'episode_rewards': episode_rewards,
        'episode_times': episode_times,
        'best_reward': best_reward,
        'avg_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards)
    }

def plot_results(results, env_name):
    """Plot training results."""
    plt.figure(figsize=(12, 5))
    
    # Plot rewards
    plt.subplot(1, 2, 1)
    plt.plot(results['episode_rewards'])
    plt.title(f'Episode Rewards - {env_name}')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    # Plot moving average
    window = 10
    moving_avg = np.convolve(results['episode_rewards'], np.ones(window)/window, mode='valid')
    plt.plot(range(window-1, len(results['episode_rewards'])), moving_avg, 'r-', label='Moving Average')
    plt.legend()
    
    # Plot times
    plt.subplot(1, 2, 2)
    plt.plot(results['episode_times'])
    plt.title(f'Episode Times - {env_name}')
    plt.xlabel('Episode')
    plt.ylabel('Time (s)')
    
    plt.tight_layout()
    plt.savefig(f'results/{env_name}_training.png')
    plt.close()

def main():
    # Benchmark environments
    environments = [
        'Pendulum-v1',
        'MountainCarContinuous-v0', 
    ]
    
    # Create results directory
    import os
    os.makedirs('results', exist_ok=True)
    
    # Run benchmarks
    for env_name in environments:
        print(f"\nRunning benchmark on {env_name}")
        results = run_benchmark(env_name)
        
        # Print summary
        print(f"\nResults for {env_name}:")
        print(f"Best Reward: {results['best_reward']:.2f}")
        print(f"Average Reward: {results['avg_reward']:.2f} ± {results['std_reward']:.2f}")
        
        # Plot results
        plot_results(results, env_name)

if __name__ == "__main__":
    main() 
import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size, batch_size, device):

        self.device = device
        self.capacity = max_size
        self.states = np.zeros((max_size, state_dim))
        self.actions = np.zeros((max_size, action_dim))
        self.rewards = np.zeros((max_size, 1))
        self.next_states = np.zeros((max_size, state_dim))
        self.dones = np.zeros((max_size, 1))

        self.batch_size = batch_size
        self.pointer = 0  # Current index to write to
        self.size = 0  # Current number of stored transitions

    def store(self, state, action, reward, next_state, done):
        self.states[self.pointer] = state
        self.actions[self.pointer] = action
        self.rewards[self.pointer] = reward
        self.next_states[self.pointer] = next_state
        self.dones[self.pointer] = done

        self.pointer = (self.pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        if batch_size > self.size:
            raise ValueError(f"Cannot sample {batch_size} transitions; only {self.size} available.")
        

        indices = np.random.choice(self.size, size=batch_size, replace=False)

        # Extract transitions using the sampled indices
        states = torch.tensor(self.states[indices], dtype=torch.float32, device=self.device)
        actions = torch.tensor(self.actions[indices], dtype=torch.float32, device=self.device)
        rewards = torch.tensor(self.rewards[indices], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(self.next_states[indices], dtype=torch.float32, device=self.device)
        dones = torch.tensor(self.dones[indices], dtype=torch.float32, device=self.device)

        return states, actions, rewards, next_states, dones
        
# Example usage
if __name__ == "__main__":
    # Initialize a Replay Buffer
    buffer = ReplayBuffer(state_dim=3, action_dim=1, max_size=1000, batch_size=4, device='cpu')
    
    # Store some dummy transitions
    for i in range(10):
        state = np.array([i, i+1, i+2])
        action = np.array([i])
        reward = i * 0.1
        next_state = np.array([i+1, i+2, i+3])
        done = 0 if i < 9 else 1
        buffer.store(state, action, reward, next_state, done)
    
    # Sample a batch
    states, actions, rewards, next_states, dones = buffer.sample(batch_size=4)
    print("Sampled states:\n", states)
    print("Sampled actions:\n", actions)
    print("Sampled rewards:\n", rewards)
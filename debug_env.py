import gymnasium as gym

print("Attempting to create BipedalWalker-v3 environment...")
try:
    env = gym.make("BipedalWalker-v3")
    print("Environment created successfully.")
    env.close()
except Exception as e:
    print(f"An error occurred: {e}")

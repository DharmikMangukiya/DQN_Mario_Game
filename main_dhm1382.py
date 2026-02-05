# NOTE: Code adapted from MinimalRL (URL: https://github.com/seungeunrho/minimalRL/blob/master/dqn.py)
# This version uses the env_dhm1382.py environment (9x9 grid with walls, obstacles, Super Mario)

# Imports:
# --------
import torch
import numpy as np
import gymnasium as gym
from DQN_model import Qnet
import torch.optim as optim
import matplotlib.pyplot as plt
from utils import ReplayBuffer, train
from env_dhm1382 import PadmEnv  # Import the dhm1382 environment

# User definitions:
# -----------------
train_dqn = True  # Set to True to retrain the model
test_dqn = False  # Set to False during training
render = True     # Set to True to show pygame window during training

# Define env attributes (environment specific)
no_actions = 4  # Number of actions (up, down, left, right)
no_states = 2   # State space (x, y coordinates on the grid)

# Hyperparameters:
# ----------------
learning_rate = 0.01
gamma = 0.98
buffer_limit = 50_000
batch_size = 32
num_episodes = 3_000  # More episodes for larger environment
max_steps = 200  # More steps for larger 9x9 grid
render_interval = 5  # Only render every N steps (faster training while still showing game)

# Environment configuration (from env_dhm1382.py)
goal_coordinates = (7, 8)
obstacle_states = [(4, 2), (1, 4), (3, 6), (6, 5)]
wall_states = [(0, 0), (0, 8), (2, 0), (1, 8), (2, 8), (3, 0), (3, 8), (4, 0), (4, 8), 
               (5, 0), (5, 8), (6, 0), (6, 8), (7, 0), (8, 0), (8, 8), (0, 1), (8, 1), 
               (0, 2), (8, 2), (0, 3), (8, 3), (0, 4), (8, 4), (0, 5), (8, 5), (0, 6), 
               (8, 6), (0, 7), (8, 7), (6, 2), (5, 6), (3, 4)]
hell_states = [(4, 4)]
food_states = [(2, 2), (6, 7), (7, 1), (1, 7)]


def create_env():
    """Create and configure the dhm1382 environment"""
    env = PadmEnv(goal_coordinates=goal_coordinates)
    
    for hell in hell_states:
        env.add_hell_states(hell)
    
    for wall in wall_states:
        env.add_wall_states(wall)
    
    for obstacle in obstacle_states:
        env.add_obstacle_states(obstacle)
    
    for food in food_states:
        env.add_food_states(food)
    
    return env


if train_dqn:
    env = create_env()

    # Initialize the Q Net and the Q Target Net
    q_net = Qnet(no_actions=no_actions, no_states=no_states)
    q_target = Qnet(no_actions=no_actions, no_states=no_states)
    q_target.load_state_dict(q_net.state_dict())

    # Initialize the Replay Buffer
    memory = ReplayBuffer(buffer_limit=buffer_limit)

    print_interval = 10
    episode_reward = 0.0
    optimizer = optim.Adam(q_net.parameters(), lr=learning_rate)

    rewards = []

    for n_epi in range(num_episodes):
        # Epsilon decay
        epsilon = max(0.01, 1 - 0.99*(n_epi/num_episodes))

        s, _ = env.reset()
        done = False

        # Define maximum steps per episode
        for step in range(max_steps):
            # Only render every N steps for faster training
            if render and step % render_interval == 0:
                env.render()
                
            # Choose an action (Exploration vs. Exploitation)
            a = q_net.sample_action(torch.from_numpy(s).float(), epsilon)
            s_prime, r, done, info = env.step(a)

            done_mask = 0.0 if done else 1.0

            # Save the trajectories
            memory.put((s, a, r, s_prime, done_mask))
            s = s_prime

            episode_reward += r

            if done:
                break

        if memory.size() > 500:
            train(q_net, q_target, memory, optimizer, batch_size, gamma)

        if n_epi % print_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q_net.state_dict())
            print(
                f"n_episode :{n_epi}, Episode reward : {episode_reward:.3f}, n_buffer : {memory.size()}, eps : {epsilon:.3f}")

        rewards.append(episode_reward)
        episode_reward = 0.0

        # Define a stopping condition for the game:
        # Commented out - this environment has different reward structure
        # if len(rewards) >= 10 and np.mean(rewards[-10:]) > 50:
        #     print("Converged!")
        #     break

    env.close()

    # Save the trained Q-net
    torch.save(q_net.state_dict(), "dqn_dhm1382.pth")

    # Plot the training curve
    plt.plot(rewards, alpha=0.3, label='Reward per Episode')
    
    # Smooth the reward curve for better visualization
    window = 50
    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards)), smoothed, label='Smoothed Reward (50 ep)')
    
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.legend()
    plt.savefig("training_curve_dhm1382.png")
    plt.show()

# Test:
if test_dqn:
    print("Testing the trained DQN: ")
    env = create_env()

    dqn = Qnet(no_actions=no_actions, no_states=no_states)
    dqn.load_state_dict(torch.load("dqn_dhm1382.pth"))

    for _ in range(10):
        s, _ = env.reset()
        episode_reward = 0

        for _ in range(max_steps):
            if render:
                env.render()

            # Completely exploit
            action = dqn(torch.from_numpy(s).float())
            s_prime, reward, done, info = env.step(action.argmax().item())
            s = s_prime

            episode_reward += reward

            if done:
                break
        print(f"Episode reward: {episode_reward:.3f}")

    env.close()

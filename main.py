# NOTE: Code adapted from MinimalRL (URL: https://github.com/seungeunrho/minimalRL/blob/master/dqn.py)

# Imports:
# --------
import torch
import numpy as np
import gymnasium as gym
from DQN_model import Qnet
import torch.optim as optim
import matplotlib.pyplot as plt
from utils import ReplayBuffer, train
from padm_env import create_env  # Import your custom environment

# User definitions:
# -----------------
train_dqn = True  # Set to True to retrain the model
test_dqn = False # Set to False during training
render = True   

# Define env attributes (environment specific)
no_actions = 4  # Number of actions (up, down, left, right)
no_states = 2   # State space (x, y coordinates on the grid)

# Hyperparameters:
# ----------------
learning_rate = 0.01
gamma = 0.98
buffer_limit = 50_000
batch_size = 32
num_episodes = 3000  # You can adjust this based on your requirements
max_steps = 100  # Adjust based on your custom environment's requirements

# Create custom environment
goal_coordinates = (5, 5)
hell_state_coordinates = [(2,2),(0,5),(4,1),(5,3)]
food_state_coordinates = [(0,2),(2,4),(5,0)]

if train_dqn:
    env = create_env(goal_coordinates, hell_state_coordinates, food_state_coordinates)

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
        # Epsilon decay (Please come up with your own logic)
        epsilon = max(0.01, 1 - 0.99*(n_epi/num_episodes))  # Linear annealing from 8% to 1%

        s, _ = env.reset()
        done = False

        # Define maximum steps per episode, here 1,000
        for _ in range(max_steps):
            if render:
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
        if len(rewards) >= 10 and np.mean(rewards[-10:]) > 90:
            print("Converged!")
            break

    env.close()

    # Save the trained Q-net
    torch.save(q_net.state_dict(), "dqn.pth")

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
    plt.savefig("training_curve.png")
    plt.show()

# Test:
if test_dqn:
    print("Testing the trained DQN: ")
    env = create_env(goal_coordinates, hell_state_coordinates, food_state_coordinates)

    dqn = Qnet(no_actions=no_actions, no_states=no_states)
    dqn.load_state_dict(torch.load("dqn.pth"))

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
        print(f"Episode reward: {episode_reward}")

    env.close()

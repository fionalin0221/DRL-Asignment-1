import os
import matplotlib.pyplot as plt
import imageio
import numpy as np
import random
import pickle
from IPython.display import Image
from costom_env_v2 import CostomEnv
import time

def get_stations(obs):
    stations = np.zeros((4, 2))
    _, _, stations[0][0],stations[0][1] ,stations[1][0],stations[1][1],stations[2][0],stations[2][1],stations[3][0],stations[3][1], _, _, _, _, _, _ = obs

    return [tuple(station) for station in stations]

def get_state(obs, stations, passenger_loc, destination_loc, get_passenger):
    taxi_row, taxi_col, _,_,_,_,_,_,_,_, obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = obs
    
    possible_positions = [
        (taxi_row - 1, taxi_col),  # Up
        (taxi_row + 1, taxi_col),  # Down
        (taxi_row, taxi_col - 1),  # Left
        (taxi_row, taxi_col + 1),  # Right
        (taxi_row, taxi_col)       # Same position
    ]
    
    if passenger_look:
        for pos in possible_positions:
            if pos in stations:
                passenger_loc = pos
                # passenger_loc = (passenger_loc[0] - taxi_row, passenger_loc[1]-taxi_col)

    if destination_look:
        for pos in possible_positions:
            if pos in stations:
                destination_loc = pos
                # destination_loc = (destination_loc[0] - taxi_row, destination_loc[1]-taxi_col)    
    
    state = (taxi_row, taxi_col, passenger_loc, destination_loc, get_passenger, obstacle_north, obstacle_south, obstacle_east, obstacle_west)

    return state

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()


def save_policy_table(policy_table, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(policy_table, file)
    print(f"policy table saved to {file_path}")


def load_policy_table(file_path):
    with open(file_path, 'rb') as file:
        policy_table = pickle.load(file)
    print(f"policy table loaded from {file_path}")
    return policy_table


def train_policy_table(env, episodes=5000, alpha=5e-5, gamma=0.99):
    policy_table = {}

    rewards_per_episode = []
    max_step = 5000

    for episode in range(episodes):
        obs, _ = env.reset()
        stations = get_stations(obs)
        get_passenger = False
        
        state = get_state(obs, stations, (-1, -1), (-1, -1), get_passenger)
        passenger_loc = state[2]
        destination_loc = state[3]

        done = False
        total_reward = 0
        trajectory = []
        step = 0

        if episode % 1000 == 0:
            alpha = alpha * 0.8

        while not done:
            if state not in policy_table:
                policy_table[state] = [0] * env.action_space.n

            action_probs = softmax(policy_table[state])
            action = np.random.choice(range(env.action_space.n), p=action_probs)

            next_obs, reward, terminant, truncated, _ = env.step(action)
            done = terminant or truncated

            if (not get_passenger) and (next_obs[0], next_obs[1]) == passenger_loc and action == 4:
                get_passenger = True
            if get_passenger and action == 5:
                get_passenger = False

            next_state = get_state(next_obs, stations, passenger_loc, destination_loc, get_passenger)
            total_reward += reward

            trajectory.append((state, action, reward))

            state = next_state
            passenger_loc = state[2]
            destination_loc = state[3]

            step += 1
            if step >= max_step:
                break

        rewards_per_episode.append(total_reward)

        # ✅ **Policy Update (REINFORCE-like)**
        G = 0
        for t in reversed(range(len(trajectory))):
            if step >= max_step and t < 0.1*len(trajectory):
                continue
            state, action, reward = trajectory[t]
            G = reward + gamma * G

            policy = softmax(policy_table[state])
            grad = - np.array(policy)
            grad[action] += 1

            policy_table[state] += alpha * G * grad

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_per_episode[-100:])
            print(f"Episode {episode + 1}/{episodes}, Avg Reward: {avg_reward:.4f}")
            save_policy_table(policy_table, "policy_table_Q4.pkl")

    return rewards_per_episode

if __name__ == "__main__":
    env = CostomEnv()
    rewards = train_policy_table(env, episodes=200000)

    plt.plot(rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Training Progress")
    plt.show()


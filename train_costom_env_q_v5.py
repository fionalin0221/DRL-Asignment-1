import os
import matplotlib.pyplot as plt
import imageio
import numpy as np
import random
import pickle
from IPython.display import Image
from costom_env_v3 import CostomEnv
import time

def get_stations(obs):
    stations = np.zeros((4, 2))
    _, _, stations[0][0],stations[0][1] ,stations[1][0],stations[1][1],stations[2][0],stations[2][1],stations[3][0],stations[3][1], _, _, _, _, _, _ = obs

    return [tuple(station) for station in stations]

def get_state(pre_obs, obs, stations, passenger_loc, destination_loc, get_passenger):
    taxi_row, taxi_col, _,_,_,_,_,_,_,_, obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = obs
    
    possible_positions = [
        (taxi_row - 1, taxi_col),  # Up
        (taxi_row + 1, taxi_col),  # Down
        (taxi_row, taxi_col - 1),  # Left
        (taxi_row, taxi_col + 1),  # Right
        (taxi_row, taxi_col)       # Same position
    ]

    if pre_obs != None:
        rel_pose = (obs[0]-pre_obs[0], obs[1]-pre_obs[1])
    else:
        rel_pose = (0, 0)

    if get_passenger:
        passenger_loc = (0, 0)

    elif passenger_look and passenger_loc == None:
        for pos in possible_positions:
            if pos in stations:
                passenger_loc = pos
                passenger_loc = (passenger_loc[0] - taxi_row, passenger_loc[1]-taxi_col)
    
    elif passenger_loc != None:
        passenger_loc = (passenger_loc[0]-rel_pose[0], passenger_loc[1]-rel_pose[1])

    if destination_look and destination_loc == None:
        for pos in possible_positions:
            if pos in stations:
                destination_loc = pos
                destination_loc = (destination_loc[0] - taxi_row, destination_loc[1]-taxi_col)

    elif destination_loc != None:
        destination_loc = (destination_loc[0]-rel_pose[0], destination_loc[1]-rel_pose[1])

    rel_stations = []
    for i in range(len(stations)):
        rel_stations.append((stations[i][0] - taxi_row, stations[i][1] - taxi_col))

    state = (passenger_loc, destination_loc, get_passenger, obstacle_north, obstacle_south, obstacle_east, obstacle_west) #, rel_stations[0], rel_stations[1], rel_stations[2], rel_stations[3]

    return state

def softmax(x, T):
    x = np.array(x)
    exp_x = np.exp(x/T - np.max(x/T))
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


def train_q_table(env, episodes=10000, alpha=0.2, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, decay_rate=0.9996):

    q_table = {}
    rewards_per_episode = []
    # epsilon = epsilon_start
    # epsilon_eposides = 0.95 * episodes

    for episode in range(episodes):
        obs, _  = env.reset()
        stations = get_stations(obs)
        get_passenger = False

        state = get_state(None, obs, stations, None, None, get_passenger)
        passenger_loc = state[0]
        destination_loc = state[1]

        # print(state)
        # env.render_env()

        done = False
        total_reward = 0
        step_count = 0
        
        while not done:
            if state not in q_table:
                q_table[state] = [0] * env.action_space.n

            # prob = random.random()
            # if prob < epsilon:
            #     action = random.randint(0, env.action_space.n - 1)
            # else:
            #     action = np.argmax(q_table[state])

            action_probs = softmax(q_table[state], T=1)
            action = np.random.choice(range(env.action_space.n), p=action_probs)

            next_obs, reward, terminant, truncated, info = env.step(action)
            done = terminant or truncated

            if (not get_passenger) and passenger_loc == (0, 0) and action == 4:
                get_passenger = True
            if get_passenger and action == 5:
                get_passenger = False

            next_state = get_state(obs, next_obs, stations, passenger_loc, destination_loc, get_passenger)
            # print(next_state)
            # env.render_env()

            total_reward += reward

            if next_state not in q_table:
                q_table[next_state] = [0] * env.action_space.n

            q_value = q_table[state][action]
            q_table[state][action] = q_value + alpha * (reward + (1-done) * gamma * max(q_table[next_state]) - q_value)

            step_count += 1
            obs = next_obs
            state = next_state
            passenger_loc = state[0]
            destination_loc = state[1]

        rewards_per_episode.append(total_reward)

        # epsilon = max(epsilon * decay_rate, epsilon_end)
        # epsilon = max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * episode / epsilon_eposides)

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_per_episode[-100:])
            print(f"Episode {episode + 1}/{episodes}, Avg Reward: {avg_reward:.4f}")
            
            with open("q_table_Q4_random_station_3.pkl", 'wb') as file:
                pickle.dump(q_table, file)
            print(f"Q-table saved to q_table_Q4_random_station_3.pkl")

    return rewards_per_episode

if __name__ == "__main__":
    env = CostomEnv()
    rewards = train_q_table(env, episodes=100000)

    plt.plot(rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Training Progress")
    plt.show()


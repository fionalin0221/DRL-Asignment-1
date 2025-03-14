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
    """
    Computes the current state representation for the taxi environment.

    Parameters:
        previous_obs (tuple or None): The previous observation of the environment.
        obs (tuple): The current observation of the environment.
        station_positions (list of tuples): The known locations of stations.
        passenger_position (tuple or None): The current known location of the passenger (relative to the taxi).
        destination_position (tuple or None): The current known location of the destination (relative to the taxi).
        has_passenger (bool): Whether the taxi currently has the passenger.

    Returns:
        tuple: The updated state representation.
    """

    # Unpack relevant features from the current observation
    taxi_row, taxi_col, _,_,_,_,_,_,_,_, obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = obs
    
    # Define possible positions where the taxi can observe objects
    possible_positions = [
        (taxi_row - 1, taxi_col),  # Up
        (taxi_row + 1, taxi_col),  # Down
        (taxi_row, taxi_col - 1),  # Left
        (taxi_row, taxi_col + 1),  # Right
        (taxi_row, taxi_col)       # Same position
    ]

    # Compute the taxi's movement relative to the previous state
    if pre_obs != None:
        movement_offset = (obs[0]-pre_obs[0], obs[1]-pre_obs[1])
    else:
        movement_offset = (0, 0)

    # If the passenger is in the taxi, its position is fixed relative to the taxi
    if get_passenger:
        passenger_loc = (0, 0)

    # If the passenger's location is unknown but visible, determine its position relative to the taxi
    elif passenger_look and passenger_loc == None:
        for pos in possible_positions:
            if pos in stations:
                passenger_loc = pos # absolute passenger location
                passenger_loc = (passenger_loc[0] - taxi_row, passenger_loc[1] - taxi_col)
    
    # If the passenger's location is known, update it based on the taxi's movement
    elif passenger_loc != None:
        passenger_loc = (passenger_loc[0]-movement_offset[0], passenger_loc[1]-movement_offset[1])

    # If the destination's location is unknown but visible, determine its position relative to the taxi
    if destination_look and destination_loc == None:
        for pos in possible_positions:
            if pos in stations:
                destination_loc = pos
                destination_loc = (destination_loc[0] - taxi_row, destination_loc[1]-taxi_col)

    # If the destination's location is known, update it based on the taxi's movement
    elif destination_loc != None:
        destination_loc = (destination_loc[0]-movement_offset[0], destination_loc[1]-movement_offset[1])

    # Compute the relative positions of the stations
    relative_stations = []
    for i in range(len(stations)):
        relative_stations.append((stations[i][0] - taxi_row, stations[i][1] - taxi_col))

    # Construct the state representation
    state = (passenger_loc, destination_loc, get_passenger, obstacle_north, obstacle_south, obstacle_east, obstacle_west) #, relative_stations[0], relative_stations[1], relative_stations[2], relative_stations[3])

    return state

def softmax(x, T):
    x = np.array(x)
    exp_x = np.exp(x/T - np.max(x/T))
    return exp_x / exp_x.sum()


def save_table(table, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(table, file)
    print(f"table saved to {file_path}")


def load_table(file_path):
    with open(file_path, 'rb') as file:
        table = pickle.load(file)
    print(f"table loaded from {file_path}")
    return table


def train_q_table(env, episodes=10000, alpha=0.2, gamma=0.99):

    q_table = {}
    rewards_per_episode = []

    for episode in range(episodes):
        obs, _  = env.reset()
        stations = get_stations(obs)
        get_passenger = False

        state = get_state(None, obs, stations, None, None, get_passenger)
        passenger_loc = state[0]
        destination_loc = state[1]

        done = False
        total_reward = 0
        step_count = 0
        
        while not done:
            # env.render_env()
            # print(state)

            if state not in q_table:
                q_table[state] = [0] * env.action_space.n

            # Choose an action using the softmax policy
            action_probs = softmax(q_table[state], T=1)
            action = np.random.choice(range(env.action_space.n), p=action_probs)

            next_obs, reward, terminant, truncated, info = env.step(action)
            done = terminant or truncated

            # Update passenger status
            if (not get_passenger) and passenger_loc == (0, 0) and action == 4:
                get_passenger = True
            if get_passenger and action == 5:
                get_passenger = False

            next_state = get_state(obs, next_obs, stations, passenger_loc, destination_loc, get_passenger)

            total_reward += reward

            if next_state not in q_table:
                q_table[next_state] = [0] * env.action_space.n

            # Q-learning update rule
            q_value = q_table[state][action]
            q_table[state][action] = q_value + alpha * (reward + (1-done) * gamma * max(q_table[next_state]) - q_value)

            step_count += 1
            obs = next_obs
            state = next_state
            passenger_loc = state[0]
            destination_loc = state[1]

        rewards_per_episode.append(total_reward)

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_per_episode[-100:])
            print(f"Episode {episode + 1}/{episodes}, Avg Reward: {avg_reward:.4f}")
            
            save_table(q_table, 'q_table_Q4_size5.pkl')

    return rewards_per_episode

if __name__ == "__main__":
    env = CostomEnv()
    rewards = train_q_table(env, episodes=100000)

    plt.plot(rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Training Progress")
    plt.show()


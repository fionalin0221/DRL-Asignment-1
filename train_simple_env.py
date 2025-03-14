import gym
import gym_minigrid
import os
import matplotlib.pyplot as plt
import imageio
import numpy as np
import random
import pickle
from IPython.display import Image
from simple_custom_taxi_env import SimpleTaxiEnv
import time

def get_state(obs):
    taxi_row, taxi_col, _,_,_,_,_,_,_,_,obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look, pass_idx, dest_idx = obs

    state = (taxi_row, taxi_col, passenger_look, destination_look, pass_idx,dest_idx)

    return state

def explore_all_walls(env):
    """Systematically explore all states to find walls (between two adjacent states)."""
    walls = set()
    
    for state in range(env.observation_space.n):  # Loop through all possible states
        env.unwrapped.s = state  # Manually set the state
        taxi_row, taxi_col, _, _ = env.custom_decode()

        for action in [0, 1, 2, 3]:  # Actions: [0: Down, 1: Up, 2: Right, 3: Left]
            env_copy = env  # Clone environment if needed
            next_state, reward, done, truncated, _ = env_copy.step(action)
            next_taxi_row, next_taxi_col, _, _ = env.custom_decode()

            # If position didn't change, it's a wall (between two states)
            if (taxi_row, taxi_col) == (next_taxi_row, next_taxi_col):
                # Create a pair of states that represent a wall
                if action == 0:  # South (down)
                    wall = ((taxi_row, taxi_col), (taxi_row + 1, taxi_col))
                elif action == 1:  # North (up)
                    wall = ((taxi_row, taxi_col), (taxi_row - 1, taxi_col))
                elif action == 2:  # East (right)
                    wall = ((taxi_row, taxi_col), (taxi_row, taxi_col + 1))
                elif action == 3:  # West (left)
                    wall = ((taxi_row, taxi_col), (taxi_row, taxi_col - 1))
                
                walls.add(tuple(sorted(wall)))  # Add the wall pair in sorted order to avoid duplicates

    return walls



def train(env, episodes=10000, alpha=0.2, gamma=0.99,
                       epsilon_start=1.0, epsilon_end=0.1, decay_rate=0.9995):

    q_table = {}
    rewards_per_episode = []
    epsilon = epsilon_start

    for episode in range(episodes):
        obs, _  = env.reset()
        
        # if (episode + 1) % 100 == 0:
        #     walls = explore_all_walls(env)
        #     print(walls)
        
        state = get_state(obs)
        
        done = False
        total_reward = 0
        step_count = 0
        
        # if episode == episodes -1:
        #     print(state, next_state)
        #     env.render_env((state[0], state[1]),
        #                 action=None, step=step_count, fuel=env.current_fuel)
        #     time.sleep(0.2)

        while not done:
            if state not in q_table:
                q_table[state] = [0] * env.action_space.n

            prob = random.random()
            if prob < epsilon:
                action = random.randint(0, env.action_space.n - 1)
            else:
                action = np.argmax(q_table[state])

            next_obs, reward, terminant, truncated, info = env.step(action)
            done = terminant or truncated
            next_state = get_state(next_obs)
            # env.render_env((state[0], state[1]), action=action, step=step_count, fuel=env.current_fuel)
            # print(state, next_state, action)

            total_reward += reward

            if next_state not in q_table:
                q_table[next_state] = [0] * env.action_space.n

            q_value = q_table[state][action]
            q_table[state][action] = q_value + alpha * (reward + (1-done) * gamma * max(q_table[next_state]) - q_value)

            step_count += 1
            state = next_state
            # if episode == episodes -1:
            #     # print(state, next_state, reward)
            #     env.render_env((state[0], state[1]),
            #                 action=action, step=step_count, fuel=env.current_fuel)
                # if state[2] != 0:
                #     print("-----------debug-----------")
                #     print(pass_idx, dest_idx)
                #     print(state)
                    # break
                # time.sleep(0.2)

            # if done:
            #     print("Exit")
            # if truncated:
            #     print("Fail")

        rewards_per_episode.append(total_reward)

        epsilon = max(epsilon * decay_rate, epsilon_end)

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_per_episode[-100:])
            print(f"Episode {episode + 1}/{episodes}, Avg Reward: {avg_reward:.4f}, Epsilon: {epsilon:.3f}")
            
            with open("q_table_Q4.pkl", 'wb') as file:
                pickle.dump(q_table, file)
            print(f"Q-table saved to q_table_Q4.pkl")
        if (episode + 1) % 1000 == 0:
            print(q_table)
    
    env.close()

    return rewards_per_episode


if __name__ == "__main__":
    env_config = {
        "fuel_limit": 5000
    }

    env = SimpleTaxiEnv(**env_config)
    
    rewards = train(env)
    
    plt.plot(rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Training Progress")
    plt.show()
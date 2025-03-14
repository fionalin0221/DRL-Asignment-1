import gym
from gymnasium import spaces
import numpy as np
import random
import time
import importlib.util

class CostomEnv:
    def __init__(self, size=10):
        super(CostomEnv, self).__init__()
        self.size = size

        # Number of obstacles in the environment
        self.num_obstacles = size*size // 10
        
        # Define the action space: 6 discrete actions (Move in 4 directions + Pickup + Dropoff)
        self.action_space = spaces.Discrete(6)  # South, North, East, West, Pick Up, Drop Off 
        
        # Observation space: Taxi's row and column (2D position)
        self.observation_space = spaces.Box(low=0, high=size - 1, shape=(2,), dtype=np.int32)
        
        # Define the maximum number of steps per episode
        self.max_steps = 5000
        self.current_steps = 0

        # Define all possible grid positions
        self.all_positions = {(i, j) for i in range(self.size) for j in range(self.size)}
        
        # Initialize key attributes
        self.stations = None #set([(0, 0), (0, self.size - 1), (self.size - 1, 0), (self.size - 1, self.size - 1)])
        self.obstacles = None #set([(1, 1), (3, 1), (0, 3)])
        self.agent_pos = None
        self.passenger_loc = None
        self.destination = None
        self.passenger_picked_up = False
        self.pre_action = None

    def _generate_grid(self):
        # Randomly generate 4 station locations, ensuring they are at least 2 Manhattan distance apart
        self.stations = set()
        while len(self.stations) < 4:
            candidate = random.choice(list(self.all_positions))
            if all(np.linalg.norm(np.array(candidate) - np.array(s), ord=1) >= 3 for s in self.stations):
                self.stations.add(candidate)

        # Assign obstacles randomly, ensuring they don't overlap with stations
        available_positions = self.all_positions - self.stations
        self.obstacles = set(random.sample(available_positions, self.num_obstacles))
        available_positions = available_positions - self.obstacles
        
        # Place taxi randomly in an available position
        self.agent_pos = random.sample(available_positions, 1)[0]

        # Select passenger and destination locations from the stations
        pass_idx, dest_idx = random.sample(range(len(self.stations)), 2)
        self.passenger_loc = list(self.stations)[pass_idx] 
        self.destination = list(self.stations)[dest_idx] 


    def reset(self):
        self.current_steps = 0
        self.passenger_picked_up = False

        self._generate_grid()
        return self.get_state(), {}
    
    def get_state(self):
        taxi_row, taxi_col = self.agent_pos
        stations = list(self.stations)

        # Identify obstacles around the taxi
        obstacle_north = ((taxi_row-1, taxi_col) in self.obstacles) or (taxi_row == 0)
        obstacle_south = ((taxi_row+1, taxi_col) in self.obstacles) or (taxi_row == self.size-1)
        obstacle_east = ((taxi_row, taxi_col+1) in self.obstacles) or (taxi_col == self.size-1)
        obstacle_west = ((taxi_row, taxi_col-1) in self.obstacles) or (taxi_col == 0)

        # Check if passenger or destination is in view
        passenger_look = ((taxi_row-1, taxi_col) == self.passenger_loc) or ((taxi_row+1, taxi_col) == self.passenger_loc) or ((taxi_row, taxi_col-1) == self.passenger_loc) or ((taxi_row, taxi_col+1) == self.passenger_loc) or ((taxi_row, taxi_col) == self.passenger_loc)
        destination_look = ((taxi_row-1, taxi_col) == self.destination) or ((taxi_row+1, taxi_col) == self.destination) or ((taxi_row, taxi_col-1) == self.destination) or ((taxi_row, taxi_col+1) == self.destination) or ((taxi_row, taxi_col) == self.destination)

        # Return state as a tuple
        state = (taxi_row, taxi_col, stations[0][0],stations[0][1] ,stations[1][0],stations[1][1],stations[2][0],stations[2][1],stations[3][0],stations[3][1]
                 ,obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look)
        
        return state
    
    def step(self, action):
        self.current_steps += 1
        self.pre_action = action

        taxi_row, taxi_col = self.agent_pos
        next_row, next_col = taxi_row, taxi_col

        # Move the taxi based on action
        if action == 0 :  # Move South
            next_row += 1
        elif action == 1:  # Move North
            next_row -= 1
        elif action == 2 :  # Move East
            next_col += 1
        elif action == 3 :  # Move West
            next_col -= 1

        reward = 0
        if action in [0, 1, 2, 3]: # Check movement validity
            if (not (0 <= next_row < self.size and 0 <= next_col < self.size)) or ((next_row, next_col) in self.obstacles):
                if self.current_steps >= self.max_steps:
                    return self.get_state(), -15, True, False, {}
                return self.get_state(), -5, False, False, {}
            else:
                self.agent_pos = (next_row, next_col)

        elif action == 4: # Pickup passenger
            if (not self.passenger_picked_up) and ((next_row, next_col) == self.passenger_loc):
                self.passenger_picked_up = True
                reward = 10
            else:
                reward = -10 # Invalid pickup attempt

        elif action == 5: # Drop-off passenger
            if self.passenger_picked_up and ((next_row, next_col) == self.destination):
                reward = 50
                return self.get_state(), reward, True, False, {} # Success
            else:
                self.passenger_picked_up = False
                reward = -10 # Invalid drop-off attempt
        
        reward -= 0.1 # Small penalty for each action

        if self.passenger_picked_up:
            self.passenger_loc = (next_row, next_col)

        if self.current_steps >= self.max_steps:
            return self.get_state(), reward -10, True, False, {} # Timeout penalty
        else:
            return self.get_state(), reward, False, False, {}
        
    def render_env(self):
        grid = [['.' for _ in range(self.size)] for _ in range(self.size)]

        # Mark obstacles
        obstacles = list(self.obstacles)
        for obstacle in obstacles:
            grid[obstacle[0]][obstacle[1]] = 'X'
        
        # Mark stations
        stations = list(self.stations)
        grid[stations[0][0]][stations[0][1]] = 'R'
        grid[stations[1][0]][stations[1][1]] = 'G'
        grid[stations[2][0]][stations[2][1]] = 'Y'
        grid[stations[3][0]][stations[3][1]] = 'B'

        # Mark agent position
        agent_pos = list(self.agent_pos)
        grid[agent_pos[0]][agent_pos[1]] = '🚖'

        # Print step info
        print(f"\nStep: {self.current_steps}")
        print(f"Taxi Position: ({self.agent_pos[0]}, {self.agent_pos[1]})")
        print(f"Passenger Position: {self.passenger_loc}, Destination: {self.destination}")
        print(f"Last Action: {self.get_action_name(self.pre_action)}")
        print(f"Get Passenger: {self.passenger_picked_up}\n")

        # Display the grid
        for row in grid:
            print(" ".join(row))
        print("\n")

        time.sleep(0.5)

    def get_action_name(self, action):
        actions = ["Move South", "Move North", "Move East", "Move West", "Pick Up", "Drop Off"]
        return actions[action] if action is not None else "None"

def run_agent(agent_file):
    spec = importlib.util.spec_from_file_location("student_agent", agent_file)
    student_agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(student_agent)

    size = random.choice([5, 6, 7, 8, 9, 10])
    env = CostomEnv(size)

    obs, _ = env.reset()
    total_reward = 0
    done = False
    step_count = 0

    # env.render_env()

    while not done:
        action = student_agent.get_action(obs)
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        step_count += 1
        # env.render_env()

    print(f"Agent Finished in {step_count} steps, Score: {total_reward}")
    return total_reward

if __name__ == "__main__":

    agent_score = run_agent("student_agent.py")
    print(f"Final Score: {agent_score}")
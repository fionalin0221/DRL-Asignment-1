import gym
from gymnasium import spaces
import numpy as np
import random
import time
import importlib.util
from IPython.display import clear_output
import curses

# Define actions
ACTIONS = [
    (0, 1),  # Right
    (0, -1), # Left
    (1, 0),  # Down
    (-1, 0)  # Up
]

# Map keys to actions
KEY_MAP = {
    curses.KEY_RIGHT: 0,
    curses.KEY_LEFT: 1,
    curses.KEY_DOWN: 2,
    curses.KEY_UP: 3,
}

class CostomEnv:
    def __init__(self, size=5):
        super(CostomEnv, self).__init__()
        self.size = size
        self.num_obstacles = 3
        self.action_space = spaces.Discrete(6)  # South, North, East, West, Pick Up, Drop Off 
        self.observation_space = spaces.Box(low=0, high=size - 1, shape=(2,), dtype=np.int32)
        self.max_steps = 5000
        self.current_steps = 0

        self.all_positions = {(i, j) for i in range(self.size) for j in range(self.size)}
        self.stations = None
        self.obstacles = None
        self.agent_pos = None
        self.passenger_loc = None
        self.destination = None
        self.passenger_picked_up = False
        self.pre_action = None

    def _generate_grid(self):
        # self.stations = set(random.sample(list(self.all_positions), 4))
        min_distance = 3
        self.stations = set()
        while len(self.stations) < 4:
            candidate = random.choice(list(self.all_positions))
            if all(np.linalg.norm(np.array(candidate) - np.array(s), ord=1) >= min_distance for s in self.stations):
                self.stations.add(candidate)
        
        available_positions = self.all_positions - self.stations
        self.obstacles = set(random.sample(available_positions, self.num_obstacles))
        
        available_positions = available_positions - self.obstacles
        self.agent_pos = random.sample(available_positions, 1)[0]

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

        obstacle_north = ((taxi_row-1, taxi_col) in self.obstacles) or (taxi_row == 0)
        obstacle_south = ((taxi_row+1, taxi_col) in self.obstacles) or (taxi_row == self.size-1)
        obstacle_east = ((taxi_row, taxi_col+1) in self.obstacles) or (taxi_col == self.size-1)
        obstacle_west = ((taxi_row, taxi_col-1) in self.obstacles) or (taxi_col == 0)

        passenger_look = ((taxi_row-1, taxi_col) == self.passenger_loc) or ((taxi_row+1, taxi_col) == self.passenger_loc) or ((taxi_row, taxi_col-1) == self.passenger_loc) or ((taxi_row, taxi_col+1) == self.passenger_loc) or ((taxi_row, taxi_col) == self.passenger_loc)
        destination_look = ((taxi_row-1, taxi_col) == self.destination) or ((taxi_row+1, taxi_col) == self.destination) or ((taxi_row, taxi_col-1) == self.destination) or ((taxi_row, taxi_col+1) == self.destination) or ((taxi_row, taxi_col) == self.destination)

        state = (taxi_row, taxi_col, stations[0][0],stations[0][1] ,stations[1][0],stations[1][1],stations[2][0],stations[2][1],stations[3][0],stations[3][1]
                 ,obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look)
        
        return state
    
    def step(self, action):
        self.current_steps += 1
        self.pre_action = action

        taxi_row, taxi_col = self.agent_pos
        next_row, next_col = taxi_row, taxi_col

        if action == 0 :  # Move South
            next_row += 1
        elif action == 1:  # Move North
            next_row -= 1
        elif action == 2 :  # Move East
            next_col += 1
        elif action == 3 :  # Move West
            next_col -= 1

        if action in [0, 1, 2, 3]:
            if (not (0 <= next_row < self.size and 0 <= next_col < self.size)) or ((next_row, next_col) in self.obstacles):
                if self.current_steps >= self.max_steps:
                    return self.get_state(), -15, True, False, {}
                return self.get_state(), -5, False, False, {}
            else:
                reward = -0.1
                self.agent_pos = (next_row, next_col)

        elif action == 4:
            if (not self.passenger_picked_up) and ((next_row, next_col) == self.passenger_loc):
                self.passenger_picked_up = True
                reward = 10
            else:
                reward = -10

        elif action == 5:
            if self.passenger_picked_up and ((next_row, next_col) == self.destination):
                reward = 50
                return self.get_state(), reward, True, False, {}
            else:
                self.passenger_picked_up = False
                reward = -10

        if self.passenger_picked_up:
            self.passenger_loc = (taxi_row, taxi_col)

        if self.current_steps >= self.max_steps:
            return self.get_state(), reward -10, True, False, {}
        else:
            return self.get_state(), reward, False, False, {}
        
    def render_env(self):
        grid = [['.' for _ in range(self.size)] for _ in range(self.size)]

        obstacles = list(self.obstacles)
        for obstacle in obstacles:
            grid[obstacle[0]][obstacle[1]] = 'X'
        
        stations = list(self.stations)
        grid[stations[0][0]][stations[0][1]] = 'R'
        grid[stations[1][0]][stations[1][1]] = 'G'
        grid[stations[2][0]][stations[2][1]] = 'Y'
        grid[stations[3][0]][stations[3][1]] = 'B'

        agent_pos = list(self.agent_pos)
        grid[agent_pos[0]][agent_pos[1]] = '🚖'

        # Print step info
        print(f"\nStep: {self.current_steps}")
        print(f"Taxi Position: ({self.agent_pos[0]}, {self.agent_pos[1]})")
        print(f"Passenger Position: {self.passenger_loc}, Destination: {self.destination}")
        print(f"Last Action: {self.get_action_name(self.pre_action)}\n")

        # Print grid
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

    env = CostomEnv(size=5)
    obs, _ = env.reset()
    total_reward = 0
    done = False
    step_count = 0

    env.render_env()

    while not done:
        action = student_agent.get_action(obs)
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        step_count += 1
        env.render_env()

    print(f"Agent Finished in {step_count} steps, Score: {total_reward}")
    return total_reward

# def play_manual(env):
#     def get_key(stdscr):
#         stdscr.nodelay(True)
#         stdscr.timeout(100)
#         while True:
#             key = stdscr.getch()
#             if key != -1:
#                 return key

#     action_map = {
#         curses.KEY_DOWN: 0,  # Move South
#         curses.KEY_UP: 1,    # Move North
#         curses.KEY_RIGHT: 2, # Move East
#         curses.KEY_LEFT: 3,  # Move West
#         ord('p'): 4,         # Pick Up
#         ord('d'): 5          # Drop Off
#     }

#     obs, _ = env.reset()
#     done = False
#     step_count = 0
#     total_reward = 0

#     def game_loop(stdscr):
#         nonlocal obs, done, step_count, total_reward
#         while not done:
#             env.render_env()
#             key = get_key(stdscr)
#             action = action_map.get(key, None)

#             if action is not None:
#                 obs, reward, done, _, _ = env.step(action)
#                 total_reward += reward
#                 step_count += 1

#         print(f"Game Over! Total Steps: {step_count}, Score: {total_reward}")

#     curses.wrapper(game_loop)


def get_key(stdscr):
    stdscr.nodelay(True)
    key = stdscr.getch()
    return key if key in KEY_MAP else None

def game_loop(stdscr, env):
    env.reset()
    done = False
    total_reward = 0
    step_count = 0
    
    while not done:
        env.render_env()
        key = get_key(stdscr)
        
        if key is not None:
            action = KEY_MAP[key]
            _, reward, done, _, _ = env.step(action)
            total_reward += reward
            step_count += 1
    
    stdscr.addstr(f"Game Over! Total Steps: {step_count}, Score: {total_reward}\n")
    stdscr.refresh()
    stdscr.getch()

def play_manual(env):
    curses.wrapper(game_loop, env)

if __name__ == "__main__":

    # env = CostomEnv(size=5)
    # obs, _ = env.reset()
    # env.render_env()
    # play_manual(env)

    agent_score = run_agent("student_agent.py")
    # print(f"Final Score: {agent_score}")
# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
import time

with open('q_table_Q4_size_all_3.pkl', 'rb') as file:
    q_table = pickle.load(file)

def softmax(x, T):
    x = np.array(x)
    exp_x = np.exp(x/T - np.max(x/T))
    return exp_x / exp_x.sum()

def get_state(obs):

    stations = np.zeros((4, 2))
    # Unpack relevant features from the current observation
    taxi_row, taxi_col, stations[0][0],stations[0][1],stations[1][0],stations[1][1],stations[2][0],stations[2][1],stations[3][0],stations[3][1], obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = obs
    # Convert stations to tuple format
    stations = [tuple(station) for station in stations]

    # Define possible positions where the taxi can observe objects
    possible_positions = [
        (taxi_row - 1, taxi_col),  # Up
        (taxi_row + 1, taxi_col),  # Down
        (taxi_row, taxi_col - 1),  # Left
        (taxi_row, taxi_col + 1),  # Right
        (taxi_row, taxi_col)       # Same position
    ]

    # Compute the taxi's movement relative to the previous state
    if get_action.pre_obs != None:
        movement_offset = (obs[0]-get_action.pre_obs[0], obs[1]-get_action.pre_obs[1])
    else:
        movement_offset = (0, 0)

    # If the passenger is in the taxi, its position is fixed relative to the taxi
    if get_action.get_passenger:
        get_action.passenger_loc = (0, 0)

    # If the passenger's location is unknown but visible, determine its position relative to the taxi
    elif passenger_look and get_action.passenger_loc == None:
        count_view = 0
        for pos in possible_positions:
            if pos in stations:
                count_view += 1
                get_action.passenger_loc = pos
                get_action.passenger_loc = (get_action.passenger_loc[0] - taxi_row, get_action.passenger_loc[1]-taxi_col)
        if count_view > 1:
            get_action.passenger_loc = None
    
    # If the passenger's location is known, update it based on the taxi's movement
    elif get_action.passenger_loc != None:
        get_action.passenger_loc = (get_action.passenger_loc[0]-movement_offset[0], get_action.passenger_loc[1]-movement_offset[1])

    # If the destination's location is unknown but visible, determine its position relative to the taxi
    if destination_look and get_action.destination_loc == None:
        count_view = 0
        for pos in possible_positions:
            if pos in stations:
                count_view += 1
                get_action.destination_loc = pos
                get_action.destination_loc = (get_action.destination_loc[0] - taxi_row, get_action.destination_loc[1]-taxi_col)
        if count_view > 1:
            get_action.destination_loc = None
    
    # If the destination's location is known, update it based on the taxi's movement
    elif get_action.destination_loc != None:
        get_action.destination_loc = (get_action.destination_loc[0]-movement_offset[0], get_action.destination_loc[1]-movement_offset[1])

    # Compute the relative positions of the stations
    relative_stations = []
    for i in range(len(stations)):
        relative_stations.append((stations[i][0] - taxi_row, stations[i][1] - taxi_col))

    # Construct the state representation
    state = (get_action.passenger_loc, get_action.destination_loc, get_action.get_passenger, obstacle_north, obstacle_south, obstacle_east, obstacle_west) #, relative_stations[0], relative_stations[1], relative_stations[2], relative_stations[3])

    return state

def get_action(obs):
    """
    Selects an action for the taxi agent based on the current observation using a pre-trained Q-table.

    Parameters:
        obs (tuple): The observation received from the environment.

    Returns:
        int: The chosen action (0-5).
    
    Action Mapping:
        0: Move South
        1: Move North
        2: Move East
        3: Move West
        4: Pickup Passenger
        5: Drop-off Passenger
    """

    # TODO: Train your own agent
    # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
    # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
    #       To prevent crashes, implement a fallback strategy for missing keys. 
    #       Otherwise, even if your agent performs well in training, it may fail during testing.
    
    # Initialize static variables for tracking state across function calls
    if not hasattr(get_action, 'pre_obs'):
        get_action.pre_obs = None  # Previous observation
        get_action.passenger_loc = None  # Passenger location (relative)
        get_action.destination_loc = None  # Destination location (relative)
        get_action.get_passenger = False  # Whether the passenger is in the taxi
        get_action.env = np.zeros((11, 11))

    # Compute the current state representation
    state = get_state(obs)
    get_action.pre_obs = obs
    get_action.passenger_loc = state[0]
    get_action.destination_loc = state[1]
    

    # Choose an action based on Q-table or a random action if the state is unknown
    if state not in q_table:
        action = np.random.choice(range(6))

    else:
        policy = softmax(q_table[state], T=1)
        action = np.random.choice(range(6), p=policy)

    # Update passenger status based on action
    if (not get_action.get_passenger) and get_action.passenger_loc == (0, 0) and action == 4:
        get_action.get_passenger = True

    if get_action.get_passenger and action == 5:
        get_action.get_passenger = False

    # print("taxi position" ,obs[0], obs[1])
    # print(f"stations: {obs[2],obs[3]}, {obs[4],obs[5]}, {obs[6],obs[7]}, {obs[8],obs[9]}")
    # print(f"obstacle: {obs[10], obs[11], obs[12], obs[13]}")
    # print(f"Passenger and Destination: {obs[14], obs[15]}")
    # print(obs)
    # print(state, action)
    # # time.sleep(0.5)

    # taxi_row, taxi_col = obs[0], obs[1]
    # s1_row, s1_col = obs[2], obs[3]
    # s2_row, s2_col = obs[4], obs[5]
    # s3_row, s3_col = obs[6], obs[7]
    # s4_row, s4_col = obs[8], obs[9]

    # # free-space: 1, obstacle: 2
    # # station: 3

    # print(taxi_row, taxi_col)
    # if taxi_row+1<=10:
    #     get_action.env[taxi_row+1][taxi_col] = obs[11] + 1
    # if taxi_row-1>=0:
    #     get_action.env[taxi_row-1][taxi_col] = obs[10] + 1
    # if taxi_col+1<=10:
    #     get_action.env[taxi_row][taxi_col+1] = obs[12] + 1
    # if taxi_col-1>=0:
    #     get_action.env[taxi_row][taxi_col-1] = obs[13] + 1

    # get_action.env[s1_row][s1_col] = 3
    # get_action.env[s2_row][s2_col] = 3
    # get_action.env[s3_row][s3_col] = 3
    # get_action.env[s4_row][s4_col] = 3

    # print(get_action.env)
    # time.sleep(0.5)

    return action

    # return random.choice([0, 1, 2, 3, 4, 5]) # Choose a random action
    # You can submit this random agent to evaluate the performance of a purely random strategy.
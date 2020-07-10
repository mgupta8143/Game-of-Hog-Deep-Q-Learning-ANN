import numpy as np
import random

def is_swap(player_score, opponent_score):
  return abs(player_score % 10 - opponent_score % 10) == opponent_score // 10 % 10

def give_reward(player_score, opponent_score):
  if player_score >= 100 and not is_swap(player_score, opponent_score):
    return 700 - player_score
  elif player_score >= 100 and is_swap(player_score, opponent_score):
    return -200
  if opponent_score >= 100 and is_swap(player_score, opponent_score):
    return 100
  elif opponent_score >= 100 and not is_swap(player_score, opponent_score):
    return -100
  if not is_swap(player_score, opponent_score):
      return player_score - opponent_score
  else:
    return (opponent_score - player_score)

environment_player_states = 160
environment_opponent_states = 160

q_values = np.zeros((environment_player_states, environment_opponent_states, 11))
actions = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]

rewards = np.full((environment_player_states, environment_opponent_states), -100.)
for x in range(environment_player_states):
    for y in range(environment_opponent_states):
        rewards[x,y] = give_reward(x,y)

def is_terminal_state(player_state, opponent_state):
  if player_state >= 100 or opponent_state >= 100:
    return True
  else:
    return False

def get_next_action(current_player_state, current_opponent_state, epsilon):
  if np.random.random() < epsilon:
    return np.argmax(q_values[current_player_state, current_opponent_state])
  else:
    return np.random.randint(4)


def get_next_location(current_player_state, current_opponent_state, action_index, previousScorePointsEarned0, previousScorePointsEarned1, feral_hogs = True):

  next_player_state = current_player_state
  next_opponent_state = current_opponent_state
  
  def is_feral(numRolls, previousScoreEarned):
      return abs(numRolls - previousScoreEarned) == 2

  if action_index == 0:
    if feral_hogs and is_feral(action_index, previousScorePointsEarned0):
       next_player_state += 3
    previousScorePointsEarned0 = 10 - (current_opponent_state % 10) + (current_opponent_state // 10)
    next_player_state += previousScorePointsEarned0
    if is_swap(next_player_state, next_opponent_state):
        temp = next_player_state
        next_player_state = next_opponent_state
        next_opponent_state = temp
  elif action_index >= 1 and action_index <= 10:
    if feral_hogs and is_feral(action_index, previousScorePointsEarned0):
       next_player_state += 3
    previousScorePointsEarned0 = random.randint(1, action_index * 6)
    if random.randint(1,100) <= int((1 - (5/6)**action_index) * 100):
      previousScorePointsEarned0 = 1
    next_player_state += previousScorePointsEarned0
    if is_swap(next_player_state, next_opponent_state):
        temp = next_player_state
        next_player_state = next_opponent_state
        next_opponent_state = temp

  opponent_roll = random.randint(0, 10)
  if opponent_roll >= 0 and opponent_roll < 6:
    opponent_roll = random.randint(0, 4)
  elif opponent_roll >= 6:
    opponent_roll = random.randint(5,10)

  if opponent_roll == 0:
    if feral_hogs and is_feral(opponent_roll, previousScorePointsEarned1):
       next_opponent_state += 3
    previousScorePointsEarned1 = 10 - (next_player_state % 10) + (next_player_state // 10)
    next_opponent_state += previousScorePointsEarned1
    if is_swap(next_opponent_state, next_player_state):
        temp = next_player_state
        next_player_state = next_opponent_state
        next_opponent_state = temp
  elif opponent_roll >= 1 and opponent_roll <= 10:
    if feral_hogs and is_feral(opponent_roll, previousScorePointsEarned1):
       next_opponent_state += 3
    previousScorePointsEarned1 = random.randint(1, opponent_roll * 6)
    if random.randint(1,100) <= int((1 - (5/6)**opponent_roll) * 100):
      previousScorePointsEarned1 = 1
    next_opponent_state += previousScorePointsEarned1
    if is_swap(next_opponent_state, next_player_state):
        temp = next_player_state
        next_player_state = next_opponent_state
        next_opponent_state = temp
    
  if next_player_state >= 160:
    next_player_state = 159
  if next_opponent_state >= 160:
    next_opponent_state = 159
    
  return next_player_state, next_opponent_state, previousScoreEarned0, previousScoreEarned1
  #Will be implemented using a random dice on both sides 

def get_shortest_path(start_row_index, start_column_index):
  #return immediately if this is an invalid starting location
  if is_terminal_state(start_row_index, start_column_index):
    return []
  else: #if this is a 'legal' starting location
    current_row_index, current_column_index = start_row_index, start_column_index
    shortest_path = []
    shortest_path.append([current_row_index, current_column_index])
    #continue moving along the path until we reach the goal (i.e., the item packaging location)
    previousScoreEarned0, previousScoreEarned1 = 0,0
    while not is_terminal_state(current_row_index, current_column_index):
      #get the best action to take
      action_index = get_next_action(current_row_index, current_column_index, 1.)
      #move to the next location on the path, and add the new location to the list
      current_row_index, current_column_index, previousScoreEarned0, previousScoreEarned1 = get_next_location(current_row_index, current_column_index, action_index, previousScoreEarned0, previousScoreEarned1)  #replace with the play function
      shortest_path.append([current_row_index, current_column_index])
    return shortest_path
    
epsilon = 1 #the percentage of time when we should take the best action (instead of a random action)
discount_factor = 0.4 #discount factor for future rewards
learning_rate = 0.4 #the rate at which the AI agent should learn
num_episodes = 30000
#run through 1000 training episodes
for episode in range(num_episodes):
  #get the starting location for this episode
  row_index, column_index = 0,0
  previousScoreEarned0, previousScoreEarned1 = 0,0

  #continue taking actions (i.e., moving) until we reach a terminal state
  while not is_terminal_state(row_index, column_index):
    #choose which action to take (i.e., where to move next)
    action_index = get_next_action(row_index, column_index, epsilon)
    #perform the chosen action, and transition to the next state (i.e., move to the next location)
    old_row_index, old_column_index = row_index, column_index #store the old row and column indexes
    row_index, column_index, previousScoreEarned0, previousScoreEarned1 = get_next_location(row_index, column_index, action_index, previousScoreEarned0, previousScoreEarned1)
    
    #receive the reward for moving to the new state, and calculate the temporal difference
    reward = rewards[row_index, column_index]
    old_q_value = q_values[old_row_index, old_column_index, action_index]
    temporal_difference = reward + (discount_factor * np.max(q_values[row_index, column_index])) - old_q_value

    #update the Q-value for the previous state and action pair
    new_q_value = old_q_value + (learning_rate * temporal_difference)
    q_values[old_row_index, old_column_index, action_index] = new_q_value
  
  print("Episode {} Done...".format(episode))

print('Training complete!')



wins = 0
total_games = 0
for i in range(10000):
  tempList = get_shortest_path(0,0)
  if tempList[len(tempList) - 1][0] >= 100:
    wins += 1
  total_games += 1

print(get_shortest_path(0,0))
print("Win Percentage: {}%".format(wins/total_games))


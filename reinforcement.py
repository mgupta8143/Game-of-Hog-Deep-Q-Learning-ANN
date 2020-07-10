#Dependencies
import os
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Dropout, MaxPooling1D, Conv1D, Activation
from keras.optimizers import Adam

"""
Objective: To learn about and create a Deep Q-Learning ANN that can play the Game of Hog. 

Game: The rules of Game of Hog are quite simple. There are two players that roll die and essentially every turn both players can roll 0 to 10 die. 
The sum on the faces of the die are added to each player's score. First player to get 100 wins. The catch is that if you roll a 1 on any of the die
you only get 1 point for that whole roll. More rolls equals more risk per turn essentially. Also, there are certain rules like Feral Hogs and Swine Swap
which can be observed in the code below that change gameplay.

Results: Using a rather simple ANN, we saw about a 59-60% win rate which quite frankly wasn't too good but it wasn't horrible either. This makes
sense considering the game is based on randomness, however, it is alsoq uite possible that the model was just simply not the right model to use

Possible Improvements: 
A convolutional neural network might improve the results by introducing "shared" weights
A better rewarding system might be necessary
More fine-tuning of hyperparameters and training epsiodes accordingly
Traditional Q-Learning may prove beneficial???
"""

state_size = 4 #As in 2 (Player Score, Opponent Score, Previous Player Score Earned, Previous Opponent Score Earned)
action_size = 10 #Because we can roll 0 to 10 die


batch_size = 32 #HYPERPARAMETER THAT CAN BE VARIED BY POWERS OF TWO
num_epsiodes = 9000 #HYPERPARAMTER #number of games

class DQNAgent:
  def __init__(self, state_size, action_size):
    self.state_size = state_size
    self.action_size = action_size

    self.memory = deque(maxlen = 3000)
    self.gamma = 0.9 #initial hyperparameter discount_factor

    self.epsilon = 1. #initial epsilon right now fully explore, but we will decay it over time
    self.epsilon_decay = 0.997 #we make the epsilon go down, so we can exploit more over time
    self.epsilon_min = 0.01 #minimum epsilon (1% of time will explore even at worst)  explore --> exploitation

    self.learning_rate = 0.001 #this is for the Adam Optimizer

    self.model = self._build_model()

  def _build_model(self):
    model = Sequential()
    #model.add(ConvD(256, (3), input_shape = (state_size)))
    
    #model.add(Flatten())
    #model.add(Dense(64))
    #model.add(Dense(self.action_size, activation = "linear"))
    #model.compile(loss = "mse", optimizer = Adam(lr = self.learning_rate))
    model.add(Dense(7, activation = "relu"))
    model.add(Dense(self.action_size, activation="linear"))
    model.compile(loss = "mse", optimizer = Adam(lr = self.learning_rate))
    return model

  def remember(self, state, action, reward, next_state, done):
    state =np.reshape(state, [1, state_size])
    self.memory.append((state, action, reward, next_state, done))

  def act(self, state):
    if np.random.rand() <= self.epsilon:
      return random.randrange(self.action_size)
    act_values = self.model.predict(np.reshape(state, [1, state_size]))
    return np.argmax(act_values[0])

  def replay(self, batch_size):
    minibatch = random.sample(self.memory, batch_size)

    for state, action, reward, next_state, done in minibatch:
      target = reward
      if not done:
        target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
      target_f = self.model.predict(state)
      target_f[0][action] = target

      self.model.fit(state, target_f, epochs = 1, verbose = 0)

    if self.epsilon > self.epsilon_min:
      self.epsilon *= self.epsilon_decay

  def load(self, name):
    self.model.load_weights(name)
  
  def save(self, name):
    self.model.save_weights(name)

def is_terminal_state(state):
  if state[0][0] >= 100 or state[0][1] >= 100:
    return True
  return False

def is_swap(player_score, opponent_score):
  return abs(player_score % 10 - opponent_score % 10) == opponent_score // 10 % 10

def give_reward(state):
  player_score = state[0][0]
  opponent_score = state[0][1]
  if not is_swap(player_score, opponent_score):
      return player_score - opponent_score
  else:
      return (opponent_score - player_score)

def get_next_location(state, action_index, feral_hogs = True):
  previousScorePointsEarned0 = state[0][2]
  previousScorePointsEarned1 = state[0][3]
  if previousScorePointsEarned0 == 0:
    previousScorePointsEarned0 = 0
  if previousScorePointsEarned1 == 0:
    previousScorePointsEarned1 = 0

  current_player_state = state[0][0]
  current_opponent_state = state[0][1]
  next_player_state = current_player_state
  next_opponent_state = current_opponent_state
  
  def is_feral(numRolls, previousScoreEarned):
      return abs(numRolls - previousScoreEarned) == 2

  if action_index == 0:
    if feral_hogs and is_feral(action_index, previousScorePointsEarned0):
       next_player_state += 3
    previousScorePointsEarned0 = 10 - (next_opponent_state % 10) + ((next_opponent_state // 10) % 10)
    next_player_state += previousScorePointsEarned0
    if is_swap(next_player_state, next_opponent_state):
        temp = next_player_state
        next_player_state = next_opponent_state
        next_opponent_state = temp
  elif action_index >= 1 and action_index <= 10:
    if feral_hogs and is_feral(action_index, previousScorePointsEarned0):
        next_player_state += 3
    if random.randint(1,100) <= int((1 - (5/6)**action_index) * 100):
        previousScorePointsEarned0 = 1
    else:
      previousScorePointsEarned0 = random.randint(1, action_index * 6)
    next_player_state += previousScorePointsEarned0
    if is_swap(next_player_state, next_opponent_state):
        temp = next_player_state
        next_player_state = next_opponent_state
        next_opponent_state = temp

  opponent_roll = random.randint(0, 10)

  if opponent_roll == 0:
    if feral_hogs and is_feral(opponent_roll, previousScorePointsEarned1):
       next_opponent_state += 3
    previousScorePointsEarned1 = 10 - (next_player_state % 10) + ((next_player_state // 10) % 10)
    next_opponent_state += previousScorePointsEarned1
    if is_swap(next_opponent_state, next_player_state):
        temp = next_player_state
        next_player_state = next_opponent_state
        next_opponent_state = temp
  elif opponent_roll >= 1 and opponent_roll <= 10:
    if feral_hogs and is_feral(opponent_roll, previousScorePointsEarned1):
       next_opponent_state += 3
    if random.randint(1,100)/100 <= 1 - (5/6)**opponent_roll:
      previousScorePointsEarned1 = 1
    else:
      previousScorePointsEarned1 = random.randint(1, opponent_roll * 6)
    next_opponent_state += previousScorePointsEarned1
    if is_swap(next_opponent_state, next_player_state):
        temp = next_player_state
        next_player_state = next_opponent_state
        next_opponent_state = temp
    
  if next_player_state >= 160:
    next_player_state = 159
  if next_opponent_state >= 160:
    next_opponent_state = 159
    
  result = [[next_player_state, next_opponent_state, previousScorePointsEarned0, previousScorePointsEarned1]]
  result_reward = 0

  if next_player_state - current_player_state == 1:
    result_reward = give_reward(result) #
  else:
    result_reward = give_reward(result)
  result = np.reshape(result, [1, state_size])
  return result, result_reward, is_terminal_state(result)

def get_shortest_path(start_row_index = 0, start_column_index = 0,  previousScorePointsEarned0 = 0,   previousScorePointsEarned1  = 0):
  if is_terminal_state([[start_row_index, start_column_index, previousScorePointsEarned0, previousScorePointsEarned1]]):
    return []
  else: 
    current_row_index, current_column_index = start_row_index, start_column_index
    state = [[current_row_index, current_column_index, previousScorePointsEarned0, previousScorePointsEarned1]]
    shortest_path = []
    shortest_path.append([state[0][0], state[0][1]])
    while not is_terminal_state(state):
      action_index = agent.act(state)
      state, reward, done = get_next_location(state, action_index)  #replace with the play function
      shortest_path.append([state[0][0], state[0][1]])
    return shortest_path

agent = DQNAgent(state_size, action_size)

#INTERACT WITH ENVIRONMENT
done = False
wins = 0
total_games = 0
previousScorePointsEarned0 = 0
previousScorePointsEarned1 = 0
for e in range(num_epsiodes):
  state = [[0,0,0,0]]
  did_win = False
  while not is_terminal_state(state):
    action = agent.act(state)
    next_state, reward, done = get_next_location(state, action)
    agent.remember(state, action, reward, next_state, done)
    state = next_state

    if done:
      if state[0][0] > state[0][1]:
        wins += 1
      total_games += 1
      if e % 500 == 0:
        print("Episode: {}/{}, Win: {} e: {:.2}".format(e, num_epsiodes, round(wins/total_games*100)/100, agent.epsilon))
        wins = 0
        total_games = 0
      break
  if len(agent.memory) > batch_size:
    agent.replay(batch_size)

###Testing Model###
wins = 0
total_games = 0
for i in range(10000):
  tempList = get_shortest_path(0,0)
  if tempList[len(tempList) - 1][0] >= 100:
    wins += 1
  total_games += 1

print(get_shortest_path(0,0))
print("Win Percentage: {}%".format(wins/total_games))

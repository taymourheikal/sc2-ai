import random
import math

import numpy as np
import pandas as pd

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features
from pysc2.env import sc2_env
from pysc2.env import environment

import cPickle as pickle
import os.path
filename="Q_table.txt"

KILL_UNIT_REWARD = 0.2
KILL_BUILDING_REWARD = 0.5
BUILD_ARMY_REWARD = 0.8
LOSE_ARMY_REWARD = -0.4
EXCESS_MINERALS_REWARD = -0.4
UP_MINERAL_RATE_REWARD = 0.4
DOWN_MINERAL_RATE_REWARD = -0.6
SPENT_MINERALS_REWARD = 0.4

## To run this:
## python -m pysc2.bin.agent --map Simple64 --agent my_smart_agent-v1.SmartAgent --agent_race T --max_agent_steps 0 --norender

## This Class stolen from https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow
## Simple q learning -- Does not incorporate a NN
class QLearningTable:
	def __init__(self, allactions, learning_rate=0.1, reward_decay=0.9, e_greedy=0.95):
		self.allactions = allactions  # a list
		self.lr = learning_rate
		self.gamma = reward_decay
		self.epsilon = e_greedy
		if os.path.exists(filename):
			self.q_table = pickle.load( open(filename, 'rb'))
			print "Opened"
		else:
			self.q_table = pd.DataFrame(columns=self.allactions)

	def choose_action(self, observation, availableactions):
		self.check_state_exist(observation)
        
		if np.random.uniform() < self.epsilon:
			# choose best action
			current_q_table = self.q_table.iloc[:, availableactions]
			#print current_q_table
			state_action = current_q_table.ix[observation, :]

			# some actions have the same value
			state_action = state_action.reindex(np.random.permutation(state_action.index))
            
			action = state_action.argmax()
		else:
			# choose random action
			action = np.random.choice(availableactions)
            
		return action

	def learn(self, s, a, r, s_):
		self.check_state_exist(s_)
		self.check_state_exist(s)

		q_predict = self.q_table.ix[s, a]
		q_target = r + self.gamma * self.q_table.ix[s_, :].max()
		
		# update
		self.q_table.ix[s, a] += self.lr * (q_target - q_predict)

		return self.q_table

		#print "Pickled"
		#pickle.dump( self.q_table, open(filename, 'wb'))
		#self.q_table.to_csv(filename, sep='\t', index_label='State')

	def check_state_exist(self, state):
		if state not in self.q_table.index:
			# append new state to q table
			self.q_table = self.q_table.append(pd.Series([0] * len(self.allactions), index=self.q_table.columns, name=state))

class SmartAgent(base_agent.BaseAgent):
	def __init__(self):

		self.qlearn = QLearningTable(allactions=list(range(524)))
		self.q_table = self.qlearn.q_table

		self.previous_killed_unit_score = 0
		self.previous_killed_building_score = 0
		self.previous_army_sumpply = 0
        
		self.previous_action = None
		self.previous_state = None

		self.episodes = 0
		self.steps = 0
		self.reward = 0
        
	def transformLocation(self, x, x_distance, y, y_distance):
		if not self.base_top_left:
			return [x - x_distance, y - y_distance]

		return [x + x_distance, y + y_distance]

	def step(self, obs):
		super(SmartAgent, self).step(obs)
		#print self.qlearn.q_table

		## Can these be incorporated logically as a state?
		killed_unit_score = obs.observation['score_cumulative'][5]
		killed_building_score = obs.observation['score_cumulative'][6]

		supply_limit = obs.observation['player'][4]
		army_supply = obs.observation['player'][5]
		mineral_count = obs.observation['player'][2]
		mineral_rate = obs.observation['score_cumulative'][9]
		mineral_spent = obs.observation['score_cumulative'][11]

		## This can also take CNN in later iterations
		current_state = [
			supply_limit,
			army_supply,
			mineral_count,
			mineral_rate,
			mineral_spent
		]

		if self.previous_action is not None:
			reward = 0
			if killed_unit_score > self.previous_killed_unit_score:
				reward += KILL_UNIT_REWARD

			if killed_building_score > self.previous_killed_building_score:
				reward += KILL_BUILDING_REWARD

			if army_supply > self.previous_army_sumpply:
				reward += BUILD_ARMY_REWARD

			if army_supply < self.previous_army_sumpply:
				reward += LOSE_ARMY_REWARD

			if mineral_count > 300:
				reward += EXCESS_MINERALS_REWARD

			if mineral_rate > self.previous_mineral_rate:
				reward += UP_MINERAL_RATE_REWARD

			if mineral_spent > self.previous_mineral_spent:
				reward += SPENT_MINERALS_REWARD

			if mineral_rate <=self.previous_mineral_rate:
				reward += DOWN_MINERAL_RATE_REWARD

			self.q_table = self.qlearn.learn(str(self.previous_state), self.previous_action, reward, str(current_state))

		if self.steps % 1000 == 0:
			print "Pickling"
			with open(filename, 'wb') as f:
				pickle.dump(self.q_table, f)

		## How to only choose available actions (as referred to in paper)
		#print list(obs.observation['available_actions'])
		rl_action = self.qlearn.choose_action(str(current_state), list(obs.observation['available_actions']))
		#print "rl_actions is: ", rl_action

		self.previous_killed_unit_score = killed_unit_score
		self.previous_killed_building_score = killed_building_score

		## Adjust state
		self.previous_supply_limit = supply_limit
		self.previous_army_sumpply = army_supply
		self.previous_mineral_count = mineral_count
		self.previous_mineral_rate = mineral_rate
		self.previous_mineral_spent = mineral_spent

		self.previous_state = current_state
		self.previous_action = rl_action

		#	pickle.dump( qtable, open(filename, 'wb'))

		args = [[np.random.randint(0, size) for size in arg.sizes]
			for arg in self.action_spec.functions[rl_action].args]

		return actions.FunctionCall(rl_action, args)

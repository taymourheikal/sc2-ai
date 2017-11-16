import random
import math

import numpy as np
import pandas as pd
import tensorflow as tf

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features
from pysc2.env import sc2_env
from pysc2.env import environment

from dqn import DeepQNetwork

## To run this:
## python -m pysc2.bin.agent --map Simple64 --agent dqn_smart_agent.SmartAgent --agent_race T --max_agent_steps 0 --norender


class SmartAgent(base_agent.BaseAgent):
	def __init__(self):
		self.dqn = DeepQNetwork(n_actions=524, n_features=13)

		self.previous_action = None
		self.previous_state = None

		self.episodes = 0
		self.steps = 0
		self.reward = 0

		self.reward_weights = np.array([
			.2,##blizz_score
			.2,.2,##total_unit_value, total_structure_value
			.2,.3,##killed_unit_value, killed_building_value
			.2,.2,##mineral_rate, mineral_spent
			.2,.1,##supply_used, supply_limit
			.3,.3,##army_supply,worker_supply
			.3#army_count
			])	
        
	def transformLocation(self, x, x_distance, y, y_distance): ## Revisit how this is evaluated
		if not self.base_top_left:
			return [x - x_distance, y - y_distance]

		return [x + x_distance, y + y_distance]

	def step(self, obs):
		super(SmartAgent, self).step(obs)

		blizz_score = obs.observation['score_cumulative'][0]
		total_unit_value = obs.observation['score_cumulative'][3]
		total_structure_value = obs.observation['score_cumulative'][4]
		killed_unit_value = obs.observation['score_cumulative'][5]
		killed_building_value = obs.observation['score_cumulative'][6]
		mineral_rate = obs.observation['score_cumulative'][9]
		mineral_spent = obs.observation['score_cumulative'][11]

		mineral_count = obs.observation['player'][1] ##7th
		supply_used = obs.observation['player'][3]
		supply_limit = obs.observation['player'][4]
		army_supply = obs.observation['player'][5]
		worker_supply = obs.observation['player'][6]
		army_count = obs.observation['player'][8]

		## This should also take feature layers
		current_state = np.array([
			blizz_score,
			total_unit_value,total_structure_value,
			killed_unit_value,killed_building_value,
			mineral_rate,mineral_spent,
			mineral_count,
			supply_used,supply_limit,
			army_supply,worker_supply,
			army_count
		]) ## New state? 0 or 1 based on position?

		## Choose action
		rl_action = self.dqn.choose_action(current_state, list(obs.observation['available_actions'])) 

		reward = 0
		if self.steps > 1:
			reward = np.delete(current_state, 7) - np.delete(self.previous_state, 7)
			reward = (reward > 0).astype(int)
			reward = np.sum(np.dot(reward, self.reward_weights))
			#print reward

			## Store transition
			self.dqn.store_transition(self.previous_state, self.previous_action, reward, current_state)

			## Learn
			self.dqn.learn()

		self.previous_state = current_state
		self.previous_action = rl_action

		args = [[np.random.randint(0, size) for size in arg.sizes]
			for arg in self.action_spec.functions[rl_action].args]

		return actions.FunctionCall(rl_action, args)

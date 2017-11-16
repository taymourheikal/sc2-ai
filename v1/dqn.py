
import numpy as np
import tensorflow as tf

# Adapted from https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow

class DeepQNetwork:
	def __init__(self, n_actions, n_features, learning_rate=0.01, gamma=0.9, e_greedy=0.9, e_greedy_increment=0.0001, epsilon_max=0.975, replace_target_iter=100,memory_size=1000, batch_size=64):
		self.n_features=n_features
		self.n_actions=n_actions
		self.learning_rate=learning_rate
		self.gamma=gamma
		self.e_greedy=e_greedy
		self.memory_size=memory_size
		self.batch_size=batch_size
		self.e_greedy_increment=e_greedy_increment
		self.epsilon_max=epsilon_max
		self.replace_target_iter=replace_target_iter

		self.relu_neurons=30

		self.learn_step_counter = 0

		self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

		self._build_net()

		t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
		e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

		with tf.variable_scope('soft_replacement'):
			self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

		self.sess = tf.Session()
		self.saver = tf.train.Saver()

		self.sess.run(tf.global_variables_initializer())

	def _build_net(self):
		self.s = tf.placeholder(tf.float32, name="s", shape=[None,self.n_features])
		self.s_ = tf.placeholder(tf.float32, name="s_", shape=[None,self.n_features])
		self.r = tf.placeholder(tf.float32, name="r", shape=[None, ])
		self.a = tf.placeholder(tf.int32, name="a", shape=[None, ])

		init_net_weights = tf.random_normal_initializer(0.,0.3)
		init_bias_weights = tf.constant_initializer(0.1)

		with tf.variable_scope("eval_net"):
			e1 = tf.layers.dense(self.s, units=self.relu_neurons, activation=tf.nn.relu, kernel_initializer=init_net_weights,
				bias_initializer=init_bias_weights, name="e1")
			self.q_eval = tf.layers.dense(e1, units=self.n_actions, kernel_initializer=init_net_weights,
				bias_initializer=init_bias_weights, name="q")

		with tf.variable_scope("target_net"):
			t1 = tf.layers.dense(self.s_, units=self.relu_neurons, activation=tf.nn.relu, kernel_initializer=init_net_weights,
				bias_initializer=init_bias_weights, name="t1")
			self.q_next = tf.layers.dense(t1, units=self.n_actions, kernel_initializer=init_net_weights,
				bias_initializer=init_bias_weights, name="t2")

		with tf.variable_scope("q_target"):
			q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name="Qmax_s_")
			self.q_target = tf.stop_gradient(q_target) ## Why stop_gradient?!?! Because he replaces target params in function
		with tf.variable_scope("q_eval"):
			a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
			## Lost here
			self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)
		with tf.variable_scope("loss"):
			self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name="TD_error"))
		with tf.variable_scope("train"):
			self._train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

	def store_transition(self, s, a, r, s_):
		if not hasattr(self, 'memory_counter'):
			self.memory_counter = 0
		transition = np.hstack((s, [a, r], s_))
		index = self.memory_counter % self.memory_size
		self.memory[index, :] = transition
		self.memory_counter +=1

	def choose_action(self, observation, avail_actions_indices): ##Indices given as list
		observation = observation[np.newaxis, :]

		if np.random.uniform() < self.e_greedy:
			actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation}).tolist()[0]
			available_actions_value = [-1 if actions_value.index(x) not in avail_actions_indices else x for x in actions_value]
			action = np.argmax(available_actions_value)
		else:
			action = int(np.random.choice(avail_actions_indices, size=1))
		return action

	def learn(self):
		if self.learn_step_counter % self.replace_target_iter == 0:
			self.sess.run(self.target_replace_op)
			save_path = self.saver.save(self.sess, "dqn_model.ckpt")

		if self.memory_counter > self.memory_size:
			sample_index = np.random.choice(self.memory_size, size=self.batch_size, replace=False)
		else:
			sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
		batch_memory = self.memory[sample_index, :]

		_, cost = self.sess.run(
			[self._train_op, self.loss],
			feed_dict={
			self.s: batch_memory[:, :self.n_features],
			self.a: batch_memory[:, self.n_features],
			self.r: batch_memory[:, self.n_features + 1],
			self.s_: batch_memory[:, -self.n_features:],
			})

		self.learn_step_counter +=1
		if self.e_greedy < self.epsilon_max:
			self.e_greedy = self.e_greedy + self.e_greedy_increment



from policies import base_policy as bp
import numpy as np
import tensorflow as tf
from collections import deque
import random
import time
import pickle

class SARSA(bp.Policy):
    def cast_string_args(self, policy_args):
        policy_args['patch_radius'] = int(policy_args['patch_radius']) if 'patch_radius' in policy_args else 5
        policy_args['start_epsilon'] = float(policy_args['start_epsilon']) if 'start_epsilon' in policy_args else 0.1
        policy_args['end_epsilon'] = float(policy_args['end_epsilon']) if 'end_epsilon' in policy_args else 0.001
        policy_args['gamma'] = float(policy_args['gamma']) if 'gamma' in policy_args else 0.98
        policy_args['eta'] = float(policy_args['eta']) if 'eta' in policy_args else 1e-4
        policy_args['batch_size'] = int(policy_args['batch_size']) if 'batch_size' in policy_args else 32
        policy_args['observe_num'] = int(policy_args['observe_num']) if 'observe_num' in policy_args else 1000
        policy_args['explore_num'] = int(policy_args['explore_num']) if 'explore_num' in policy_args else 100000
        policy_args['replay_num'] = int(policy_args['replay_num']) if 'replay_num' in policy_args else 10000
        policy_args['head_shift'] = int(policy_args['head_shift']) if 'head_shift' in policy_args else 0
        policy_args['suicide_prob'] = float(policy_args['suicide_prob']) if 'suicide_prob' in policy_args else 1
        policy_args['eat_prob'] = float(policy_args['eat_prob']) if 'eat_prob' in policy_args else 0.025
        policy_args['kill_terminals'] = bool(policy_args['kill_terminals']) if 'kill_terminals' in policy_args else True
        policy_args['verbose'] = bool(policy_args['verbose']) if 'verbose' in policy_args else True
        return policy_args

    def init_run(self):
        # running score sum
        self.r_sum = 0
        # dicts that save transitions and rewards for each iteration. key: t, val: transition(state,action,state2)/reward
        self.transition_dict = {}
        self.reward_dict = {}
        # dicts that map value in the game to a reward, and saving the minimal reward we saw (should be the kill reward hopefully)
        self.value_reward_dict = {}
        self.minimal_reward = -0.000001
        self.iteration_to_value = {}

        # time limit for our act function so we don't add messy data to our batches
        self.max_time = 0.0195
        self.decay_rate = (self.start_epsilon - self.end_epsilon) / self.explore_num
        self.epsilon = self.start_epsilon

        # network initialization and parameters
        self.state_space_sz = (2 * self.patch_radius + 1)

        self.sess = tf.Session()
        self.x = tf.placeholder(tf.float32, shape=[None, self.state_space_sz, self.state_space_sz, 1]) # state-space-size
        self.net_q = self.q_net(self.x)
        self.a = tf.placeholder(tf.float32, [None, len(bp.Policy.ACTIONS)])
        self.y = tf.placeholder(tf.float32, shape=[None])
        self.action = tf.reduce_sum(tf.multiply(self.net_q, self.a), reduction_indices=1)
        self.loss = tf.reduce_mean(tf.square(self.y - self.action))
        self.train_step = tf.train.AdamOptimizer(self.eta).minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())


        if hasattr(self, 'load_from'):
            with open(self.load_from, 'rb') as f:
                model_from_pickle = pickle.load(f)
                for variable, weight in zip(tf.trainable_variables(), model_from_pickle):
                    self.sess.run(variable.assign(weight))

        # our memory of previous states,actions,rewards,states2,actions2
        self.state_memory = deque()
        self.t1 = None

        self.action_moves= {'CC':[0,-1], 'CW':[0,1], 'CN':[-1,0]}


    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):
        # update reward
        self.t1 = time.time()
        self.reward_dict[round] = reward
        if round % 1000 == 0 and self.verbose:
            self.log(str(self.r_sum), 'SARSA-value')
        self.r_sum = self.r_sum*0.9998 + reward

        # make sure previous values are good for learning
        if round - 2 not in self.transition_dict or round - 1 not in self.transition_dict or round - 1 not in self.reward_dict:
            return

        # get the previous state, action, reward, state2, action2
        state_action1 = self.transition_dict.pop(round - 2)
        state1 = state_action1[0]
        action1 = state_action1[1]
        state_action2 = self.transition_dict[round - 1]
        state2 = state_action2[0]
        action2 = state_action2[1]
        old_reward = self.reward_dict.pop(round - 1)
        if state1[self.patch_radius, self.patch_radius] == state1[self.patch_radius + self.action_moves[action1][0], self.patch_radius + self.action_moves[action1][1]]:
            old_reward = -100
        # input the reward (without the length) into our value-reward dictionary
        self.value_reward_dict[self.iteration_to_value[round - 2]] = old_reward
        self.iteration_to_value.pop(round - 2)

        # update minimal reward (death penalty)
        if old_reward < self.minimal_reward:
            self.minimal_reward = old_reward

        # learn Deep-Q-Net
        self.learnQ(state1, action1, old_reward, state2, action2, round)

    # def act(self, t, state, player_state):
    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):
        # encode state
        board, head = new_state
        head_pos, direction = head

        encoded_state = self.encode_state_generic(board, head_pos, direction, self.patch_radius)

        # choose random action (by epsilon) or choose by q function
        random_action = False
        if np.random.rand() < self.epsilon:
            action = np.random.choice(bp.Policy.ACTIONS)
            random_action = True
        else:
            q = self.getQ(encoded_state[np.newaxis,:, :,np.newaxis])
            action = bp.Policy.ACTIONS[np.argmax(q)]

        # head_pos = player_state['chain'][-1]
        # print(head_pos.move(bp.Policy.TURNS[direction][action]))
        # print(exit())
        cord = head_pos.move(bp.Policy.TURNS[direction][action])
        r, c = np.mod((cord[0], cord[1]), board.shape)
        # r, c = head_pos.move(bp.Policy.TURNS[direction][action])[0], head_pos.move(bp.Policy.TURNS[direction][action])[1]
        # r, c = np.mod((r, c), board.shape)

        # try not to kill yourself, in probability (won't get in here with default values)
        if np.random.rand() > self.suicide_prob and self.value_reward_dict.get(board[r, c], np.inf) <= self.minimal_reward:
            if not random_action:
                action = bp.Policy.ACTIONS[np.argsort(q)[0][1]]
            else:
                for act in bp.Policy.ACTIONS:
                    r, c = head_pos.move(bp.Policy.TURNS[direction][act]) % board.shape
                    if self.value_reward_dict.get(board[r, c], np.inf) > self.minimal_reward:
                        action = act
                        break
        else:
            # try to improve action, in probability (low probability) for smart exploring
            pred_value = self.value_reward_dict.get(board[r, c], None)
            if pred_value is not None and np.random.rand() < self.eat_prob:
                for act in bp.Policy.ACTIONS:
                    r, c = head_pos.move(bp.Policy.TURNS[direction][act]) % board.shape
                    if self.value_reward_dict.get(board[r, c], -np.inf) > pred_value:
                        action = act
                        break

        # if we didn't make it on time, don't save action
        # if (time.time() - self.t1) > self.max_time:
        #     if self.verbose:
        #         self.log('elapsed: %f' % (time.time() - self.t1))
        # else:
        #     # save action to dictionary
        #     cord = head_pos.move(bp.Policy.TURNS[direction][action])
        #     r, c = np.mod((cord[0], cord[1]), board.shape)
        #     self.iteration_to_value[round] = board[r, c]
        #     self.transition_dict[round] = (encoded_state, action)
        #
        #     # reduce epsilon
        #     if self.epsilon > self.end_epsilon and round > self.observe_num:
        #         self.epsilon -= self.decay_rate
        cord = head_pos.move(bp.Policy.TURNS[direction][action])
        r, c = np.mod((cord[0], cord[1]), board.shape)
        self.iteration_to_value[round] = board[r, c]
        self.transition_dict[round] = (encoded_state, action)

        # reduce epsilon
        if self.epsilon > self.end_epsilon and round > self.observe_num:
            self.epsilon -= self.decay_rate
        return action

    def get_state(self):
        return self.sess.run(tf.trainable_variables())

    def getQ(self, state):
        return self.net_q.eval(feed_dict = {self.x: state}, session=self.sess)

    def learnQ(self, state, action, reward, state2, action2, t):

        # remove item from replay memory, if it is full
        if len(self.state_memory) > self.replay_num:
            self.state_memory.popleft()

        # add new item to replay memory
        self.state_memory.append((state[:,:,np.newaxis], self.action_to_vec(action), reward, state2[:,:,np.newaxis],  self.action_to_vec(action2)))

        # learn only after seeing X examples
        if t > self.observe_num:
            # get minibatch
            minibatch = random.sample(self.state_memory, self.batch_size)

            # get minibatch variables
            minibatch_states = [i[0] for i in minibatch]
            minibatch_actions = [i[1] for i in minibatch]
            minibatch_rewards = [i[2] for i in minibatch]
            minibatch_states2 = [i[3] for i in minibatch]
            minibatch_actions2 = [i[4] for i in minibatch]

            # predict next state Q
            next_state_q = self.getQ(minibatch_states2)
            rwrd = np.array(minibatch_rewards)
            # calculate labels based on if it is a terminal or not
            if self.kill_terminals:
                y_ = rwrd + np.where(rwrd==-100, 0, self.gamma * next_state_q[:,np.where(minibatch_actions2 != 0)[0][0]])
            else:
                y_ = rwrd + self.gamma * next_state_q[:,np.where(minibatch_actions2 != 0)[0][0]]

            # train on batch
            _, l, nq= self.sess.run([self.train_step, self.loss, self.net_q], feed_dict = {self.y: y_, self.a: minibatch_actions, self.x: minibatch_states})

            if t % 1000 == 0 and self.verbose:
                self.log('%d %.5f %.7f' % (t, l, self.epsilon))

    # network definition
    def q_net(self, x):
        '''
        Architecture:

        conv layer
        max-pool
        conv layer
        max pool
        fully connected
        fully connected

        '''
        W_conv1 = self.weight_variable([3, 3, 1, 8])
        b_conv1 = self.bias_variable([8])

        h_conv1 = tf.nn.relu(self.conv2d(x, W_conv1) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)

        W_conv2 = self.weight_variable([3, 3, 8, 16])
        b_conv2 = self.bias_variable([16])

        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)

        h_pool_shape = h_pool2.get_shape().as_list()

        W_fc1 = self.weight_variable([h_pool_shape[1] * h_pool_shape[2] * h_pool_shape[3], 10])
        b_fc1 = self.bias_variable([10])

        h_pool2_flat = tf.reshape(h_pool2, [-1, h_pool_shape[1] * h_pool_shape[2] * h_pool_shape[3]])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        W_fc2 = self.weight_variable([10, 3])
        b_fc2 = self.bias_variable([3])

        y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2
        return y_conv

    def action_to_vec(self, action):
        a_vec = np.zeros((len(bp.Policy.ACTIONS)))
        a_vec[bp.Policy.ACTIONS.index(action)] = 1
        return a_vec

    # convert state to KxK matrix
    def encode_state_generic(self, state, head_pos, direction, radius):
        # head_pos = player_state['chain'][-1]
        # direction = player_state['dir']
        # create a KxK window around the coordinate
        window_indices = np.meshgrid(
            np.mod(np.arange(head_pos[0] - radius - self.head_shift, head_pos[0] + radius - self.head_shift + 1), state.shape[0]),
            np.mod(np.arange(head_pos[1] - radius, head_pos[1] + radius + 1), state.shape[1]))
        encoded_state = state[window_indices].T

        if direction == 'E':
            encoded_state = np.rot90(encoded_state, k=1)
        if direction == 'W':
            encoded_state = np.rot90(encoded_state, k=3)
        if direction == 'S':
            encoded_state = np.rot90(encoded_state, k=2)

        return encoded_state

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='VALID')
import random
from collections import namedtuple
from time import time

import numpy as np
from tensorflow.python import keras

from policies import base_policy as bp

Policy = bp.Policy

Transition = namedtuple('Transition', ('prev_state', 'prev_action', 'reward', 'new_state'))

INT_TO_ACTION_MAPPING = {0: 'L', 1: 'R', 2: 'F'}
ACTION_TO_INT_MAPPING = {'L': 0, 'R': 1, 'F': 2}

MIN_BOARD_VALUE, MAX_BOARD_VALUE = -1, 9
NORMALIZE_PTP = np.ptp([MIN_BOARD_VALUE, MAX_BOARD_VALUE])


# bs 8, radius 1, gamma 0.2

DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_CAPACITY = 5000
DEFAULT_RADIUS = 2
DEFAULT_GAMMA = 0.01  # Discount factor
DEFAULT_LR = 0.001
DEFAULT_TRAIN_ON_TERMINAL_STATES = True
DEFAULT_START_EPSILON, DEFAULT_END_EPSILON = 1.0, 0.1
DEFAULT_SYMBOL_REWARD_DECAY = 0.9


DEFAULT_VERBOSE = False
DEFAULT_VERBOSE_PRINT_FREQ = 100


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        random.seed(time())
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Linear(bp.Policy):

    def _normalize_state(self, s):
        return (2 * (s - MIN_BOARD_VALUE) / NORMALIZE_PTP) - 1

    def _break_state(self, state):
        board, head = state
        pos, direction = head
        pos = pos.pos
        return board, direction, pos

    def _preprocess_state(self, state):
        if state is None:
            return None
        board, direction, head_pos = self._break_state(state)
        Hs = np.arange(head_pos[0] - self.window_radius, head_pos[0] + self.window_radius + 1)
        Ws = np.arange(head_pos[1] - self.window_radius, head_pos[1] + self.window_radius + 1)
        indices = np.meshgrid(
            np.mod(Hs, board.shape[0]),
            np.mod(Ws, board.shape[1])
        )
        patch = board[tuple(indices)].T
        if direction == "N":
            pass
        elif direction == "E":
            patch = np.rot90(patch, k=1)
        elif direction == "S":
            patch = np.rot90(patch, k=2)
        elif direction == "W":
            patch = np.rot90(patch, k=3)
        else:
            raise ValueError("Critical error, unknown direction: %s" % direction)
        patch = patch.reshape([1, -1])
        return self._normalize_state(patch)

    def _preprocess_is_final(self, prev_state, new_state):
        if prev_state is None:
            return False
        _, _, prev_head = self._break_state(prev_state)
        _, _, new_head = self._break_state(new_state)
        board_shape = self.board_size
        prev_loc = np.array([prev_head[0], prev_head[1]])
        new_loc = np.array([new_head[0], new_head[1]])
        # We have 4 types of steps:
        # step in 0 axes
        if ((prev_loc[0] + 1) % board_shape[0] == new_loc[0] and prev_loc[1] == new_loc[1]) or (
                (prev_loc[0] - 1) % board_shape[0] == new_loc[0] and prev_loc[1] == new_loc[1]):
            return False
        # step in axis 1
        elif ((prev_loc[1] + 1) % board_shape[1] == new_loc[1] and prev_loc[0] == new_loc[0]) or (
                (prev_loc[1] - 1) % board_shape[1] == new_loc[1] and prev_loc[0] == new_loc[0]):
            return False
        else:
            return True

    def _preprocess_data(self, prev_state, prev_action, reward, new_state):
        is_final = self._preprocess_is_final(prev_state, new_state)
        prev_board = self._preprocess_state(prev_state)
        new_board = self._preprocess_state(new_state)
        if prev_action is not None:
            prev_action = ACTION_TO_INT_MAPPING[prev_action]
            prev_action = np.array([prev_action], dtype=np.uint8)
        if reward is not None:
            reward = np.array([reward], dtype=np.float32)
        return prev_board, prev_action, reward, new_board, is_final

    def cast_string_args(self, policy_args):
        """
        this function casts arguments passed during policy construction to their proper types/names.
        :param policy_args: an arg -> string value map as received in command line.
        :return: A map of string -> value after casting to useful objects, these will be added as members to the policy
        """
        self.gamma = float(policy_args.get('g', DEFAULT_GAMMA))
        self.learning_rate = float(policy_args.get('lr', DEFAULT_LR))
        self.train_on_terminal_states = bool(policy_args.get('tot', DEFAULT_TRAIN_ON_TERMINAL_STATES))
        self.verbose = bool(policy_args.get('v', DEFAULT_VERBOSE))
        self.batch_size = int(policy_args.get('bs', DEFAULT_BATCH_SIZE))
        self.max_capacity = int(policy_args.get('mc', DEFAULT_MAX_CAPACITY))
        self.start_epsilon = float(policy_args.get('se', DEFAULT_START_EPSILON))
        self.end_epsilon = float(policy_args.get('ee', DEFAULT_END_EPSILON))
        self.symbol_reward_decay = float(policy_args.get('srd', DEFAULT_SYMBOL_REWARD_DECAY))
        self.epsilon_decay = (self.start_epsilon - self.end_epsilon) / self.game_duration
        self.eps = self.start_epsilon
        self.prev_round_symbol_for_action = None
        self.prev_round_action = None
        self.window_radius = int(policy_args.get('r', DEFAULT_RADIUS))
        self.window_radius = int(np.floor(min(self.window_radius, self.board_size[0] / 2, self.board_size[1] / 2)))
        self.input_shape = np.power((2 * self.window_radius + 1), 2)
        if self.verbose:
            print("Batch Size:", self.batch_size)
            print("Gamma:", self.gamma)
            print("Learning Rate:", self.learning_rate)
            print("Radius:", self.window_radius)
            print("Eps Decay Start, End:", self.start_epsilon, self.end_epsilon)
        return {}

    def init_run(self):
        """
        this function is called right after the initialization of the agent.
        you may use it to initialize variables that are needed for your policy,
        such as Keras models and so on. you may also use this function
        to load your pickled model and set the variables accordingly, if the
        game uses a saved model and is not a training session.
        """
        self.replay_memory = ReplayMemory(capacity=self.max_capacity)
        self.symbol_reward_map = {}
        self.model = keras.models.Sequential([
            keras.layers.Dense(units=3,
                               input_shape=(self.input_shape,),
                               kernel_initializer=keras.initializers.glorot_normal(),
                               bias_initializer=keras.initializers.zeros(),
                               name='q_values')
        ])
        self.model.compile(optimizer=keras.optimizers.Adam(lr=self.learning_rate), loss='mse')
        # self.model.compile(optimizer=keras.optimizers.Nadam(lr=self.learning_rate), loss='mse')

        # force keras lazy method's to be not lazy
        rand_num = np.random.rand(1, self.input_shape)
        pred_rand = self.model.predict(rand_num)
        self.model.train_on_batch(rand_num, pred_rand)

    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):
        """
        the function for learning and improving the policy. it accepts the
        state-action-reward needed to learn from the final move of the game,
        and from that (and other state-action-rewards saved previously) it
        may improve the policy.
        :param round: the round of the game.
        :param prev_state: the previous state from which the policy acted.
        :param prev_action: the previous action the policy chose.
        :param reward: the reward given by the environment following the previous action.
        :param new_state: the new state that the agent is presented with, following the previous action.
                          This is the final state of the round.
        :param too_slow: true if the game didn't get an action in time from the
                        policy for a few rounds in a row. you may use this to make your
                        computation time smaller (by lowering the batch size for example).
        """
        size_replay_memory = len(self.replay_memory)
        if size_replay_memory == 0:
            return
        batch_size = np.minimum(self.batch_size, size_replay_memory)
        transitions = self.replay_memory.sample(batch_size)
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for detailed explanation).
        batch = Transition(*zip(*transitions))
        state_batch = np.concatenate(batch.prev_state)
        action_batch = np.concatenate(batch.prev_action)
        reward_batch = np.concatenate(batch.reward)
        if self.train_on_terminal_states:
            target = reward_batch
            non_final_indices = np.array([i for i in range(batch_size) if batch.new_state[i] is not None],
                                         dtype=np.uint8)
            if non_final_indices.size != 0:
                non_final_states = np.concatenate([s for s in batch.new_state if s is not None])
                target[non_final_indices] += self.gamma * np.max(self.model.predict(non_final_states), axis=1,
                                                                 keepdims=False)
        else:
            new_state_batch = np.concatenate(batch.new_state)
            target = reward_batch + self.gamma * np.max(self.model.predict(new_state_batch), axis=1, keepdims=False)
        target_f = self.model.predict(state_batch)
        target_f[np.arange(batch_size), action_batch] = target
        loss = self.model.train_on_batch(state_batch, target_f)
        if self.verbose and round % DEFAULT_VERBOSE_PRINT_FREQ == 0:
            print("Round %d, Loss: %3.5f" % (round, loss))

    def _prevent_suicide(self, unmodified_new_state, new_state):
        board, head = unmodified_new_state
        head_pos, direction = head
        for a in self.ACTIONS:
            next_pos = head_pos.move(bp.Policy.TURNS[direction][a])
            if board[next_pos[0], next_pos[1]] == self.id:
                a = ACTION_TO_INT_MAPPING[a]
                a = np.array([a], dtype=np.uint8)
                self.replay_memory.push(new_state, a, np.array([-100.0], dtype=np.float32), None)

    def _smart_explore(self, new_state):
        board, head = new_state
        head_pos, direction = head
        cur_best_reward, cur_best_action = -np.inf, "F"
        for a in np.random.permutation(self.ACTIONS):
            next_pos = head_pos.move(bp.Policy.TURNS[direction][a])
            pos_board_value = board[next_pos[0], next_pos[1]]
            next_pos_reward = self.symbol_reward_map.get(pos_board_value, -np.inf)
            if next_pos_reward > cur_best_reward:
                cur_best_reward = next_pos_reward
                cur_best_action = a
        return cur_best_action

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):
        """
        the function for choosing an action, given current state.
        it accepts the state-action-reward needed to learn from the previous
        move (which it can save in a data structure for future learning), and
        accepts the new state from which it needs to decide how to act.
        :param round: the round of the game.
        :param prev_state: the previous state from which the policy acted.
        :param prev_action: the previous action the policy chose.
        :param reward: the reward given by the environment following the previous action.
        :param new_state: the new state that the agent is presented with, following the previous action.
        :param too_slow: true if the game didn't get an action in time from the
                        policy for a few rounds in a row. use this to make your
                        computation time smaller (by lowering the batch size for example)...
        :return: an action (from Policy.Actions) in response to the new_state.
        """
        random.seed(time())
        board, head = new_state
        head_pos, direction = head
        unmodified_new_state = new_state
        self._update_reward_mapping(reward)
        prev_state, prev_action, reward, new_state, is_final = self._preprocess_data(prev_state=prev_state,
                                                                                     prev_action=prev_action,
                                                                                     reward=reward, new_state=new_state)
        num = random.uniform(0, 1)
        eps = self.eps / 2
        if num < eps:  # Smart explore
            action = self._smart_explore(unmodified_new_state)
        elif eps <= num < 2 * self.eps:  # Random explore
            int_action = random.randint(0, len(self.ACTIONS) - 1)  # a <= N <= b
            action = INT_TO_ACTION_MAPPING[int_action]
        else:
            int_action = self.model.predict(new_state).argmax(axis=1)[0]
            action = INT_TO_ACTION_MAPPING[int_action]
        if round > 0:
            if is_final and self.train_on_terminal_states:
                self.replay_memory.push(prev_state, prev_action, reward, None)
            else:
                self.replay_memory.push(prev_state, prev_action, reward, new_state)

        self._prevent_suicide(unmodified_new_state, new_state)
        # Update epsilon
        self.eps = self.start_epsilon - (self.start_epsilon * round * self.epsilon_decay)
        self.eps = np.clip(self.eps, self.end_epsilon, self.start_epsilon)

        # Update the values and action for next round
        self.prev_round_action = action
        next_pos = head_pos.move(bp.Policy.TURNS[direction][action])
        self.prev_round_symbol_for_action = board[next_pos[0], next_pos[1]]
        # if self.verbose and (round % DEFAULT_VERBOSE_PRINT_FREQ == 0):
        #     print("Epsilon:", self.eps)
        #     print("# Memories:", len(self.replay_memory))
        #     print("SymbolRewardMap:", self.symbol_reward_map)
        return action

    def _update_reward_mapping(self, new_r):
        if self.prev_round_symbol_for_action is not None:
            if self.prev_round_symbol_for_action not in self.symbol_reward_map:
                self.symbol_reward_map[self.prev_round_symbol_for_action] = new_r
            else:
                r = self.symbol_reward_map[self.prev_round_symbol_for_action]
                w = self.symbol_reward_decay
                updated_reward = (1 - w) * r + w * new_r
                self.symbol_reward_map[self.prev_round_symbol_for_action] = updated_reward

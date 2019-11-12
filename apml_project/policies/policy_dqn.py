from policies import base_policy as bp
from time import time
from tensorflow.python import keras
import numpy as np
from collections import namedtuple
import random

Policy = bp.Policy

Transition = namedtuple('Transition', ('prev_state', 'prev_action', 'reward', 'new_state'))

INT_TO_ACTION_MAPPING = {0: 'L', 1: 'R', 2: 'F'}
ACTION_TO_INT_MAPPING = {'L': 0, 'R': 1, 'F': 2}

MIN_BOARD_VALUE, MAX_BOARD_VALUE = -1, 9
NORMALIZE_PTP = np.ptp([MIN_BOARD_VALUE, MAX_BOARD_VALUE])

DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_CAPACITY = 8192 * 2
DEFAULT_RADIUS = 8
DEFAULT_GAMMA = 0.99  # Discount factor
DEFAULT_LR = 0.0001
DEFAULT_SYMBOL_REWARD_DECAY = 0.9
DEFAULT_SAVE_NON_POLICY_MEMORIES = True
DEFAULT_TRAIN_ON_TERMINAL_STATES = True
DEFAULT_START_EPSILON, DEFAULT_END_EPSILON = 1.0, 0.001
DEFAULT_UPDATE_TARGET_ESTIMATOR_FREQ = 250

DEFAULT_VERBOSE = False
DEFAULT_VERBOSE_PRINT_FREQ = 100
DEFAULT_DOUBLE_NETWORK = True


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


def build_model(input_shape):
    state_input = keras.layers.Input(shape=input_shape, name='states')
    nn = keras.layers.Conv2D(filters=4, kernel_size=(3, 3), activation='relu',
                             input_shape=input_shape,
                             kernel_initializer=keras.initializers.glorot_normal(),
                             bias_initializer=keras.initializers.zeros())(state_input)
    nn = keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)(nn)
    nn = keras.layers.Conv2D(filters=8, kernel_size=(3, 3), activation='relu',
                             kernel_initializer=keras.initializers.glorot_normal(),
                             bias_initializer=keras.initializers.zeros())(nn)
    nn = keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)(nn)
    nn = keras.layers.Flatten()(nn)
    nn = keras.layers.Dense(units=64,
                            activation=None,
                            kernel_initializer=keras.initializers.glorot_normal(),
                            bias_initializer=keras.initializers.zeros())(nn)
    nn = keras.layers.Dense(units=len(Policy.ACTIONS),
                            kernel_initializer=keras.initializers.glorot_normal(),
                            bias_initializer=keras.initializers.zeros(),
                            name='q_values')(nn)

    return keras.Model(inputs=state_input, outputs=nn)


class DQN(bp.Policy):

    def normalize_state(self, s):
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
        # patch = patch.T
        # we insert it into a CNN add dpeth
        patch = patch[np.newaxis, :, :, np.newaxis]
        return self.normalize_state(patch)

    def _preprocess_head(self, state):
        board, _, head = self._break_state(state)
        board_shape = self.board_size
        # normalize to [-1,1] with: 2 * ((x - min(x)) / (max(x) - min(x))) - 1
        head_loc = np.array([
            2 * (head[0] / board_shape[0]) - 1,
            2 * (head[1] / board_shape[1]) - 1
        ])
        head_loc = np.round(head_loc, 2)
        return head_loc.reshape([1, -1])

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

    def preprocess_data(self, prev_state, prev_action, reward, new_state):
        # head_loc = self._preprocess_head(new_state)
        is_final = self._preprocess_is_final(prev_state, new_state)  # Before preprocessing the states. it removes head
        prev_state = self._preprocess_state(prev_state)
        new_state = self._preprocess_state(new_state)
        if prev_action is not None:
            prev_action = ACTION_TO_INT_MAPPING[prev_action]
            prev_action = np.array([prev_action], dtype=np.uint8)
        if reward is not None:
            reward = np.array([reward], dtype=np.float32)
        return prev_state, prev_action, reward, new_state, is_final

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
        self.is_double = float(policy_args.get('double', DEFAULT_DOUBLE_NETWORK))
        self.epsilon_decay = (self.start_epsilon - self.end_epsilon) / self.game_duration
        self.eps = self.start_epsilon
        self.symbol_reward_decay = float(policy_args.get('srd', DEFAULT_SYMBOL_REWARD_DECAY))
        self.prev_round_symbol_for_action = None
        self.prev_round_action = None
        self.window_radius = int(policy_args.get('r', DEFAULT_RADIUS))
        self.window_radius = int(np.floor(min(self.window_radius, self.board_size[0] / 2, self.board_size[1] / 2)))
        self.input_shape = (2 * self.window_radius + 1)
        return {}

    def transfer_weights(self):
        """ Transfer Weights from Model to Target at rate Tau
        """
        q_weights = self.model.get_weights()
        self.target_model.set_weights(q_weights)

    def init_run(self):
        """
        this function is called right after the initialization of the agent.
        you may use it to initialize variables that are needed for your policy,
        such as Keras models and so on. you may also use this function
        to load your pickled model and set the variables accordingly, if the
        game uses a saved model and is not a training session.
        """
        """
        State dimension: 
            * (x,y) for snake's head position => 2
            * {S,W,N,E} for snake's head direction => 1 (converted to integer)
            * X*Y for board positions
        """
        self.slow = 0
        self.slow_learn = 0
        self.replay_memory = ReplayMemory(capacity=self.max_capacity)
        self.symbol_reward_map = {}
        self.model = build_model(input_shape=(self.input_shape, self.input_shape, 1))
        self.model.compile(optimizer=keras.optimizers.adam(lr=self.learning_rate), loss='mse')

        # from tensorflow.python.keras.utils import plot_model
        # plot_model(self.model, show_shapes=True, to_file='./dqn.png')

        # This is used to force generate the model
        rand_num = np.random.rand(1, self.input_shape, self.input_shape, 1)
        pred_rand = self.model.predict(rand_num)
        self.model.train_on_batch(rand_num, pred_rand)

        if self.is_double:
            self.target_model = keras.models.clone_model(self.model)
            self.target_model.predict(np.random.rand(1, self.input_shape, self.input_shape, 1))
        else:
            self.target_model = self.model

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
        if round % DEFAULT_UPDATE_TARGET_ESTIMATOR_FREQ == 0 and self.is_double:
            self.transfer_weights()

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
                target[non_final_indices] += self.gamma * np.max(self.target_model.predict(non_final_states), axis=1,
                                                                 keepdims=False)
        else:
            new_state_batch = np.concatenate(batch.new_state)
            target = reward_batch + self.gamma * np.max(self.target_model.predict(new_state_batch), axis=1,
                                                        keepdims=False)
        target_f = self.model.predict(state_batch)
        target_f[np.arange(batch_size), action_batch] = target
        loss = self.model.train_on_batch(state_batch, target_f)
        if self.verbose:
            print("Round %d:"
                  "\n\tLoss: %3.5f" % (round, loss))
            if round >= self.game_duration - 5:
                print("total slow learn: %d" % self.slow_learn)

    def smart_explore(self, new_state):
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
        if too_slow:
            self.slow += 1
        board, head = new_state
        head_pos, direction = head
        unmodified_new_state = new_state
        self.update_reward_mapping(reward)
        prev_state, prev_action, reward, new_state, is_final = self.preprocess_data(prev_state=prev_state,
                                                                                    prev_action=prev_action,
                                                                                    reward=reward, new_state=new_state)
        is_policy_action = False
        num = random.uniform(0, 1)
        eps = self.eps / 2
        if num < eps:  # Smart explore
            action = self.smart_explore(unmodified_new_state)
        elif eps < num <= 2 * eps:  # Random explore
            int_action = random.randint(0, len(self.ACTIONS) - 1)  # a <= N <= b
            action = INT_TO_ACTION_MAPPING[int_action]
        else:
            is_policy_action = True
            prediction_vec = self.model.predict(new_state)[0]
            int_action = prediction_vec.argmax()
            action = INT_TO_ACTION_MAPPING[int_action]
        if round > 0:
            if is_policy_action:  # Obviously save all policy initiated actions
                if is_final and self.train_on_terminal_states:
                    self.replay_memory.push(prev_state, prev_action, reward, None)
                else:
                    self.replay_memory.push(prev_state, prev_action, reward, new_state)
            elif DEFAULT_SAVE_NON_POLICY_MEMORIES and not is_policy_action:
                if is_final and self.train_on_terminal_states:
                    self.replay_memory.push(prev_state, prev_action, reward, None)
                else:
                    self.replay_memory.push(prev_state, prev_action, reward, new_state)

        # Update epsilon
        self.eps = self.start_epsilon - (self.start_epsilon * round * self.epsilon_decay)
        self.eps = np.clip(self.eps, self.end_epsilon, self.start_epsilon)

        # Update the values and action for next round
        self.prev_round_action = action
        next_pos = head_pos.move(bp.Policy.TURNS[direction][action])
        self.prev_round_symbol_for_action = board[next_pos[0], next_pos[1]]

        if self.verbose and (round % DEFAULT_VERBOSE_PRINT_FREQ == 0):
            print("Epsilon:", self.eps)
            print("SymbolRewardMap:", self.symbol_reward_map)

        return action

    def update_reward_mapping(self, new_r):
        if self.prev_round_symbol_for_action is not None:
            if self.prev_round_symbol_for_action not in self.symbol_reward_map:
                self.symbol_reward_map[self.prev_round_symbol_for_action] = new_r
            else:
                r = self.symbol_reward_map[self.prev_round_symbol_for_action]
                w = self.symbol_reward_decay
                updated_reward = (1 - w) * r + w * new_r
                self.symbol_reward_map[self.prev_round_symbol_for_action] = updated_reward

from policies import base_policy as bp
import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from collections import namedtuple
import random


Policy = bp.Policy

NUM_ACTIONS = len(Policy.ACTIONS)


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

BATCH_SIZE = 32
MAX_CAPACITY = 8192
GAMMA = 0.99
LR = 0.001


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
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



def preprocess_state(state):
    board = state[0]
    head = state[1]
    head_pos = head[0]
    head_direction = head[1]
    print(board.shape, type(board))
    print(head_pos[0], head_pos[1])
    print(ACTION_TO_INT[head_direction])
    return np.concatenate((head, board))


class LinearAgent(keras.Model):

    def __init__(self, num_units_state: int, num_actions: int, *args, **kwargs):
        super(LinearAgent, self).__init__(*args, **kwargs)
        self.dense1 = keras.layers.Dense(units=num_units_state // 2)
        self.dense2 = keras.layers.Dense(units=64)
        self.dense3 = keras.layers.Dense(units=32)
        self.dense4 = keras.layers.Dense(units=num_actions, activation=keras.activations.softmax)

    def call(self, inputs, training=None, mask=None):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        return x

    def compute_output_shape(self, input_shape):
        return None, NUM_ACTIONS


class Linear(bp.Policy):


    def cast_string_args(self, policy_args):
        """
        this function casts arguments passed during policy construction to their proper types/names.
        :param policy_args: an arg -> string value map as received in command line.
        :return: A map of string -> value after casting to useful objects, these will be added as members to the policy
        """
        args = {
            'gamma': float(policy_args.get('g', GAMMA)),
            'learning_rate': float(policy_args.get('lr', LR)),
            'batch_size': int(policy_args.get('bs', 32)),
            'max_capacity': int(policy_args.get('mc', MAX_CAPACITY)),
        }
        return args

    def init_run(self):
        """
        this function is called right after the initialization of the agent.
        you may use it to initialize variables that are needed for your policy,
        such as Keras models and so on. you may also use this function
        to load your pickled model and set the variables accordingly, if the
        game uses a saved model and is not a training session.
        """
        board_size = self.board_size
        ACTIONS = Policy.ACTIONS
        NUM_ACTIONS = len(ACTIONS)
        num_units_input = int(np.prod(board_size))
        self.replay_memory = ReplayMemory(capacity=self.max_capacity)
        self.agent = LinearAgent(num_units_state=num_units_input, num_actions=NUM_ACTIONS)
        self.agent.compile(optimizer=keras.optimizers.adam(), loss=keras.losses.mse)


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
        gamma = self.gamma
        learning_rate = self.learning_rate
        batch_size = self.batch_size
        transitions = self.replay_memory.sample(batch_size)
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        batch = Transition(*zip(*transitions))


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
        board, head = new_state
        head_pos, direction = head
        self.replay_memory.push(Transition(state=prev_state, action=prev_action, next_state=new_state, reward=reward))
        q_values = self.agent.predict(board)
        print(q_values)
        action = np.argmax(q_values, axis=0)
        return np.argmax(q_values)

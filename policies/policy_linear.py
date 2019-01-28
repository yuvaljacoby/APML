from policies import base_policy as bp
import numpy as np
from time import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple
import random


Policy = bp.Policy

ACTIONS = Policy.ACTIONS
NUM_ACTIONS = len(ACTIONS)

# ACTION_TO_INT = dict([])


Transition = namedtuple('Transition', ('prev_state', 'prev_action', 'next_state', 'reward'))

BATCH_SIZE = 32
MAX_CAPACITY = 8192
GAMMA = 0.99
LR = 0.001

########################################################################################################################
################################################# POLICIES #############################################################
########################################################################################################################


def random_action():
    random.seed(time())
    rand_int = random.uniform(0, NUM_ACTIONS)
    return ACTIONS[rand_int]


def make_epsilon_greedy_policy(eps_start, eps_end, eps_decay):
    def eps_greedy_policy(t, q_values):
        random.seed(time())
        eps_threshold = eps_end + (eps_start - eps_end) * \
            np.exp(-1. * t / eps_decay)
        if random.uniform(0, 1) > eps_threshold:
            return q_values.max(1)[1].view(1, 1)
        else:
            return random_action()
    return eps_greedy_policy


def make_q_values_policy():
    return lambda t, q_values: F.softmax(q_values, dim=1).multinomial(num_samples=1)


########################################################################################################################
################################################# UTILITIES ############################################################
########################################################################################################################


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, **args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        t = Transition(**args)
        self.memory[self.position] = t
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        random.seed(time())
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

########################################################################################################################
################################################### POLICY #############################################################
########################################################################################################################


class LinearAgent(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(LinearAgent, self).__init__()
        self.fc1 = nn.Linear(in_features=in_dim, out_features=100)
        # self.fc2 = nn.Linear(in_features=24, out_features=24)
        # self.fc2 = nn.Linear(in_features=24, out_features=24)
        # self.fc3 = nn.Linear(in_features=24, out_features=24)
        self.head = nn.Linear(in_features=100, out_features=out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        return self.head(x)


class Linear(bp.Policy):

    def cast_string_args(self, policy_args):
        """
        this function casts arguments passed during policy construction to their proper types/names.
        :param policy_args: an arg -> string value map as received in command line.
        :return: A map of string -> value after casting to useful objects, these will be added as members to the policy
        """
        # args = {
        #     'gamma': float(policy_args.get('g', GAMMA)),
        #     'learning_rate': float(policy_args.get('lr', LR)),
        #     'batch_size': int(policy_args.get('bs', BATCH_SIZE)),
        #     'max_capacity': int(policy_args.get('mc', MAX_CAPACITY)),
        #     # 'policy': make_q_values_policy(),
        # }
        self.gamma = float(policy_args.get('g', GAMMA))
        self.learning_rate = float(policy_args.get('lr', LR))
        self.batch_size = int(policy_args.get('bs', BATCH_SIZE))
        self.max_capacity = int(policy_args.get('mc', MAX_CAPACITY))
        self.policy = make_q_values_policy()
        return {}

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
        # print(torch.cuda.device.idx)
        # exit()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.replay_memory = ReplayMemory(capacity=self.max_capacity)
        self.estimator = LinearAgent(in_dim=num_units_input, out_dim=NUM_ACTIONS).to(device=self.device)
        self.optimizer = optim.Adam(self.estimator.parameters(), lr=self.learning_rate)
        self.augment_data = False

    def get_action(self, step, state, is_deterministic=False):
        q_values = self.estimator.forward(state)
        with torch.no_grad():
            if is_deterministic:
                return F.softmax(q_values, dim=1).argmax().to(self.device)
            else:
                return self.policy(step, q_values).to(self.device)

    def preprocess(self, prev_state, prev_action, reward, new_state):
        prev_state = torch.tensor(prev_state.reshape([1, -1]), dtype=torch.float, device=self.device)

        reward = torch.tensor([reward], dtype=torch.float, device=self.device)
        new_state = torch.tensor(new_state.reshape([1, -1]), dtype=torch.float, device=self.device)
        return prev_state, prev_action, reward, new_state


    def augment(self, prev_state, prev_action, reward, new_state):
        """
        4.6 Using Symmetries To Your Advantage
            If you’re using a relatively large state representation, you may want to think of ways of learning
            faster by taking advantage of certain symmetries in the game. This way, you may turn a single
            state-action-reward triplet into more than one triplet, and in this way you could learn faster and
            “see more states” without actually visiting them.
        :param prev_state: the previous state from which the policy acted.
        :param prev_action: the previous action the policy chose.
        :param reward: the reward given by the environment following the previous action.
        :param new_state: the new state that the agent is presented with, following the previous action.
        :return:
        """
        pass
        # results = []
        # action_idx = ACTIONS.index(prev_action)
        # i = np.arange()
        # while
        #     prev_x = np.rot90(prev_state, k=i)
        #     new_x = np.rot90(new_state, k=i)
        # for x in data:
        #     preprocess(x). add to memory


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
        if self.augment_data:
            self.augment(prev_state=prev_state, prev_action=prev_action, reward=reward, new_state=new_state)
        self.replay_memory.push(prev_state=prev_state, prev_action=prev_action, next_state=new_state, reward=reward)
        return self.get_action(step=round, state=board)

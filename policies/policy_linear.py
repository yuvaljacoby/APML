from policies import base_policy as bp
import numpy as np
from time import time
import torch
from torchvision.transforms import functional as TVF
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple
import random


Policy = bp.Policy

ACTIONS = Policy.ACTIONS
NUM_ACTIONS = len(ACTIONS)

ACTION_TO_INT_MAPPING = {
    'L': 0,
    'R': 1,
    'F': 2,
}
INT_TO_ACTION_MAPPING = {
    0: 'L',
    1: 'R',
    2: 'F',
}
DIRECTION_TO_INT_MAPPING = {
    'N': 0,
    'S': 1,
    'W': 2,
    'E': 3,
}
INT_TO_DIRECTION_MAPPING = {
    0: 'N',
    1: 'S',
    2: 'W',
    3: 'E',
}



Transition = namedtuple('Transition', ('prev_state', 'prev_action', 'reward', 'new_state'))

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
        eps_threshold = eps_end + (eps_start - eps_end) * np.exp(-1. * t / eps_decay)
        random.seed(time())
        if random.uniform(0, 1) > eps_threshold:
            return q_values.max(1)[1].view(1, 1)
        else:
            return random_action()
    return eps_greedy_policy


def make_q_values_policy():
    return lambda t, q_values: F.softmax(q_values, dim=1).multinomial(num_samples=1)



########################################################################################################################
################################################### POLICY #############################################################
########################################################################################################################


class LinearModel(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(LinearModel, self).__init__()
        self.fc1 = nn.Linear(in_features=in_dim, out_features=600)
        self.fc2 = nn.Linear(in_features=600, out_features=300)
        # self.fc3 = nn.Linear(in_features=24, out_features=24)
        self.head = nn.Linear(in_features=300, out_features=out_dim)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.head(x)
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        return x


class Agent(object):
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

    def __init__(self, in_dim, out_dim, gamma, replay_memory_capacity, learning_rate, policy, NNModel, device):
        self._device = device
        self._estimator = NNModel(in_dim=in_dim, out_dim=out_dim).to(self._device)
        self._replay_memory = Agent.ReplayMemory(replay_memory_capacity)
        self._NNModel = NNModel
        self._optimizer = optim.Adam(self._estimator.parameters(), lr=learning_rate)
        self._policy = policy
        self._gamma = gamma

    def add_to_memory(self, *args):
        self._replay_memory.push(*args)

    def reset_weights(self):
        self._estimator.reset_weights()

    def get_weights(self):
        return self._estimator.state_dict()

    def set_weights(self, new_weights):
        t_weights = self._estimator.parameters()
        for old, new in zip(t_weights, new_weights):
            # old.data = torch.from_numpy(new).to(self._device)
            old.data = new

    def predict(self, state):
        pred = self._estimator.forward(state)
        return pred

    def get_action(self, step, state, is_deterministic=False):
        q_values = self.predict(state)
        with torch.no_grad():
            if is_deterministic:
                action = F.softmax(q_values, dim=1).argmax()
            else:
                action = self._policy(step, q_values)
            return int(action.cpu().numpy())

    def train(self, batch_size):
        size_replay_memory = len(self._replay_memory)
        if size_replay_memory == 0:
            return
        batch_size = min(batch_size, size_replay_memory)
        transitions = self._replay_memory.sample(batch_size)
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.prev_state)
        action_batch = torch.cat(batch.prev_action).reshape(-1, 1)
        reward_batch = torch.cat(batch.reward)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.new_state)),
                                      device=self._device, dtype=torch.uint8)
        non_final_new_states = torch.cat([s for s in batch.new_state if s is not None])
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        q_values = self._estimator(state_batch)
        state_action_values = q_values.gather(1, action_batch)
        # Compute V(s_{t+1}) for all next states.
        new_state_values = torch.zeros(batch_size, device=self._device)
        new_state_values[non_final_mask] = self._estimator(non_final_new_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (new_state_values * self._gamma) + reward_batch
        # Compute MSE loss
        loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        # Optimize the model
        self._optimizer.zero_grad()
        loss.backward()
        for param in self._estimator.parameters():
            param.grad.data.clamp_(-1, 1)
        self._optimizer.step()

    def save_model(self, path):
        try:
            torch.save(self._estimator.state_dict(), path)
            return True
        except Exception as e:
            print("Failed to write state_dict")
            print(e)
            return False

    def load_model(self, path):
        try:
            self._estimator.load_state_dict(torch.load(path))
            return True
        except Exception as e:
            print("Failed to read state_dict")
            print(e)
            return False


class Linear(bp.Policy):

    def _preprocess_state(self, state):
        if state is None:
            return None
        board, head = state
        board = board.reshape([1, -1])
        pos, dir = head
        pos = np.array((pos[0], pos[1]), dtype=np.float32).reshape([1, 2])
        dir = np.array([DIRECTION_TO_INT_MAPPING[dir]], dtype=np.float32).reshape([1, 1])
        s = np.concatenate((pos, dir, board), axis=1)
        return torch.tensor(s, dtype=torch.float, device=self._device)

    def preprocess_data(self, prev_state, prev_action, reward, new_state):
        prev_state = self._preprocess_state(prev_state)
        new_state = self._preprocess_state(new_state)
        if prev_action is not None:
            prev_action = ACTION_TO_INT_MAPPING[prev_action]
            prev_action = torch.tensor([prev_action], dtype=torch.long, device=self._device).reshape(1, 1)
        if reward is not None:
            reward = torch.tensor([reward], dtype=torch.float, device=self._device)
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
        if prev_state is None or prev_action is None:
            return
        prev_state = prev_state.cpu().numpy()
        prev_action = int(prev_action.cpu().numpy())
        reward = reward.cpu().numpy()
        new_state = new_state.cpu().numpy()


        if self.board_size[0] == self.board_size[1]:  # additional augmentation
            for i in range(1, 3):  # [1, 2] => 2 more possible rotations TODO Maybe consider mirror for more examples?
                n_p_state, n_p_action, n_reward, n_n_state = self.preprocess_data(prev_state=np.rot90(prev_state, k=i),
                                                                                  prev_action=prev_action,
                                                                                  reward=reward,
                                                                                  new_state=np.rot90(new_state, k=i))
                self.agent.add_to_memory(n_p_state, n_p_action, n_reward, n_n_state)



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
        """
        State dimension: 
            * (x,y) for snake's head position => 2
            * {S,W,N,E} for snake's head direction => 1 (converted to integer)
            * X*Y for board positions
        """
        global NUM_ACTIONS
        state_dim = 3 + int(np.prod(self.board_size))
        self.augment_data = True
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.agent = Agent(in_dim=state_dim, out_dim=NUM_ACTIONS, gamma=self.gamma,
                           replay_memory_capacity=self.max_capacity,
                           learning_rate=self.learning_rate,
                           policy=make_q_values_policy(),
                           NNModel=LinearModel,
                           device=self._device)

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
        self.agent.train(batch_size=self.batch_size)

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
        # board, head = new_state
        # pos, dir = head
        # print(board.shape)
        # print(pos[0], pos[1])
        # print(dir)
        prev_state, prev_action, reward, new_state = self.preprocess_data(prev_state, prev_action, reward, new_state)
        if round != 0:
            self.agent.add_to_memory(prev_state, prev_action, reward, new_state)
        action = self.agent.get_action(round, state=new_state)
        return INT_TO_ACTION_MAPPING[action]

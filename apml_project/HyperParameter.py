import numpy as np
import pandas as pd

import os.path
import time
from Snake import Game, parse_args
from policies import base_policy

BOARD_SIZE = [(25, 3), (30, 30)]
DURATION = [100]
SCORE_DURATION_RATIO = [0.1]


def run_game_multiple(policies, k=5, board_size=None, game_duration=None, score_scope=None, action_time=None,
                      learning_time=None):
    scores = []
    for _ in range(k):
        scores.append(run_game_once(policies, board_size, game_duration, score_scope, action_time, learning_time))
    return np.mean(scores, axis=0)


def run_game_get_scores(policies, board_size=None, game_duration=None, score_scope=None, action_time=None,
                        learning_time=None, init_time=25):
    if not game_duration:
        game_duration = np.random.choice(DURATION)

    if not score_scope:
        score_scope = np.random.choice(SCORE_DURATION_RATIO) * game_duration

    score_scope = int(score_scope)
    args = parse_args()
    arg_d = vars(args)
    arg_d['game_duration'] = game_duration
    arg_d['score_scope'] = score_scope

    if board_size:
        arg_d['board_size'] = board_size

    if learning_time:
        arg_d['policy_learn_time'] = learning_time

    if action_time:
        arg_d['policy_wait_time'] = action_time

    arg_d['player_init_time'] = init_time
    arg_d['policies'] += [base_policy.build(p) for p in policies]
    g = Game(args)
    scores = g.run()
    return scores


def run_game_once(policies, board_size=None, game_duration=None, score_scope=None, action_time=None,
                  learning_time=None):
    scores = run_game_get_scores(policies, board_size, game_duration, score_scope, action_time, learning_time)

    scope = [np.min([len(scores[i]), score_scope]) for i in range(len(policies))]
    # The indices are like this to align with snake.py (they print the result before the last score)
    return [np.mean(scores[i][-scope[i] - 1:-1]) for i in range(len(policies))]


def _build_str_policy(base_policy, gamma=None, lr=None, batch_size=None, max_capacity=None, start_epsilon=None,
                      end_epsilon=None, window_radius=None, train_on_terminal_states=None, double=None):
    if gamma:
        base_policy += "g=" + str(gamma) + ","

    if lr:
        base_policy += "lr=" + str(lr) + ","

    if batch_size:
        base_policy += "bc=" + str(batch_size) + ","

    if max_capacity:
        base_policy += "mc=" + str(max_capacity) + ","

    if start_epsilon:
        base_policy += "se=" + str(start_epsilon) + ","

    if end_epsilon:
        base_policy += "ee=" + str(end_epsilon) + ","

    if window_radius:
        base_policy += "r=" + str(window_radius) + ","

    if train_on_terminal_states:
        base_policy += "tot=" + str(train_on_terminal_states) + ","

    if double is not None:
        base_policy += "double=" + str(double) + ","

    last_comma = base_policy.rfind(",")
    if last_comma > 0:
        return base_policy[:last_comma] + ')'
    else:
        return base_policy + ')'


def build_linear_policy(gamma=None, lr=None, batch_size=None, max_capacity=None, start_epsilon=None,
                        end_epsilon=None, window_radius=None, tot=None):
    return _build_str_policy("SimpleLinear(", gamma, lr, batch_size, max_capacity, start_epsilon, end_epsilon,
                             window_radius)


def build_custom_policy(gamma=None, lr=None, batch_size=None, max_capacity=None, start_epsilon=None,
                        end_epsilon=None, window_radius=None, double=None):
    return _build_str_policy("Custom(", gamma, lr, batch_size, max_capacity, start_epsilon, end_epsilon, window_radius,
                             double=double)


def build_dqn_policy(gamma=None, lr=None, batch_size=None, max_capacity=None, start_epsilon=None,
                     end_epsilon=None, window_radius=None, double=None):
    return _build_str_policy("DQN(", gamma, lr, batch_size, max_capacity, start_epsilon, end_epsilon, window_radius,
                             double=double)


def build_ddqn_policy(gamma=None, lr=None, batch_size=None, max_capacity=None, start_epsilon=None,
                      end_epsilon=None, window_radius=None, double=None):
    return _build_str_policy("DDQN(", gamma, lr, batch_size, max_capacity, start_epsilon, end_epsilon, window_radius,
                             double=double)


def linear_evaluation(gamma=None, lr=None, batch_size=None, max_capacity=None, start_epsilon=None,
                      end_epsilon=None, window_radius=None, train_on_terminal_states=None):
    game_duration = 5000
    scope = 1000
    action_time = 0.005
    learning_time = 0.01
    lp = build_linear_policy(gamma, lr, batch_size, max_capacity, start_epsilon, end_epsilon, window_radius,
                             train_on_terminal_states)

    avoid_str = "Avoid(epsilon=0)"

    policies = [lp, avoid_str, avoid_str]

    return run_game_multiple(policies, game_duration=game_duration, score_scope=scope, action_time=action_time,
                             learning_time=learning_time, board_size=(50, 30))


def custom_avoid_evaluation(gamma=None, lr=None, batch_size=None, max_capacity=None, start_epsilon=None,
                            end_epsilon=None, window_radius=None):
    game_duration = 50000
    scope = 5000
    action_time = 0.005
    learning_time = 0.01
    cp = build_custom_policy(gamma, lr, batch_size, max_capacity, start_epsilon, end_epsilon, window_radius)

    avoid_agents = ["Avoid(epsilon={})".format(str(np.random.uniform(0, 0.5))) for _ in range(4)]

    policies = [cp] + avoid_agents

    return run_game_once(policies, game_duration=game_duration, score_scope=scope, action_time=action_time,
                         learning_time=learning_time)


def random_search(evaluation_func, num_iterations):
    gamma_range = np.arange(0.01, 1, 0.1)
    lr_range = np.arange(0.01, 0.2, 0.05)
    bs_range = np.arange(16, 64, 8)
    max_capacity_range = np.arange(1000, 5000, 1000)
    s_epsilon_range = np.arange(0.001, 0.5, 0.05)
    e_epsilon_range = np.arange(0, 0.1, 0.01)
    raduis_range = np.arange(2, 7, 1)
    scores = []
    file_path = "HyperParameter" + evaluation_func.__name__ + ".csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        max_idx = df.index.max() + 1
    else:
        df = None
        max_idx = 0

    for i in range(max_idx, num_iterations + max_idx):
        start = time.time()
        g = np.random.choice(gamma_range)
        lr = np.random.choice(lr_range)
        bs = np.random.choice(bs_range)
        mc = np.random.choice(max_capacity_range)
        s = np.random.choice(s_epsilon_range)
        e = np.random.choice(e_epsilon_range)
        r = np.random.choice(raduis_range)
        tot = np.random.choice([0, 1])
        score = evaluation_func(gamma=g, lr=lr, batch_size=bs, max_capacity=mc, start_epsilon=s, end_epsilon=e,
                                window_radius=r, train_on_terminal_states=tot)

        score_dict = {
            'gamma': g, 'lr': lr, 'batch_size': bs, 'max_capacity': mc, 'start_epsilon': s, 'end_epsilon': e,
            'radius': r, 'score': score[0], 'other_scores': score[1:], 'index': i, 'train_on_terminal_states': tot
        }
        print("##########################################################################################\n\\n\n\n")
        print("iteration: %d took:" % i, time.time() - start)
        print(score_dict)
        print("##########################################################################################\n\\n\n\n")
        scores.append(score_dict)

    df2 = pd.DataFrame(scores)
    if df is not None:
        df = df.append(df2, ignore_index=True)
    else:
        df = df2
    df.to_csv("HyperParameter" + evaluation_func.__name__ + ".csv")
    return df


if __name__ == "__main__":
    # print("linear_evaluation:", linear_evaluation(0.4, 0.1, 64, 3500, 0.05, 0.005, 5))
    # linear params
    print(random_search(linear_evaluation, 5))

    # custom params
    # print(random_search(custom_avoid_evaluation, 2))

import pickle
import time

import matplotlib.pyplot as plt
import numpy as np

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
    return _build_str_policy("Linear(", gamma, lr, batch_size, max_capacity, start_epsilon, end_epsilon,
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


def compare_model():
    start = time.time()
    duration = 50000
    scope = 5000

    policies = []
    policies.append(build_dqn_policy(double=1))
    policies.append(build_dqn_policy(double=0))
    policies.append(build_ddqn_policy(double=1))
    policies.append(build_ddqn_policy(double=0))

    scores = run_game_get_scores(policies, game_duration=duration, score_scope=scope, action_time=0.02,
                                 learning_time=0.1, init_time=15)

    pickle.dump(scores, open('scores.temp', 'wb'))
    for i, mod in enumerate(['double_dqn', 'dqn', 'double_ddqn', 'ddqn']):
        cur_score = scores[i]
        score_to_plot = []
        for s in range(scope, duration):
            score_to_plot.append(np.mean(cur_score[s - scope: s]))
        plt.plot(range(scope, duration), score_to_plot, label=mod)
    plt.legend()
    plt.title("compare models")
    plt.show()
    print("experiment took:", time.time() - start)


def radius_size():
    duration = 50000
    scope = 5000
    options = [7, 9]
    scores = []
    for opt in options:
        policies = ['Avoid(epsilon=0)', 'Avoid(epsilon=0)']
        policies.append(build_ddqn_policy(window_radius=opt, double=1))

        scores.append(run_game_get_scores(policies, game_duration=duration, score_scope=scope, action_time=0.02,
                                          learning_time=0.1))

    pickle.dump(scores, open('scores.temp', 'wb'))
    for i, opt in enumerate(options):
        cur_score = scores[i][1]
        score_to_plot = []
        for s in range(scope, duration, scope):
            score_to_plot.append(np.mean(cur_score[s - scope: s]))
        plt.plot(range(scope, duration, scope), score_to_plot, label=opt)
    plt.legend()
    plt.title("compare scores as a function of window size")
    plt.show()


if __name__ == '__main__':
    # radius_size()
    compare_model()
    # linear_evaluation()

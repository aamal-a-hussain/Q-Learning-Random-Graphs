import numpy as np
from networkgames import get_payoffs


def check_var(traj, window_size, tol):
    return np.mean(np.var(traj[:, -window_size:], axis=1)) <= tol


def check_rel_diff(traj, window_size, tol):
    rel_diff = np.max(traj[:, -window_size:], axis=1) - \
        np.min(traj[:, -window_size:], axis=1)
    rel_diff /= np.max(traj[:, -window_size:], axis=1)

    return np.all(rel_diff <= tol)


def init_strats(n_agents, n_actions):
    return np.random.dirichlet(np.ones(n_actions), size=n_agents)

# SMALL T
# def stable_softmax(x):
#    # Subtract max along axis 1 (actions) while keeping dimensions
#    z = x - np.max(x, axis=1)[:, np.newaxis]
#    numerator = np.exp(z)
#    denominator = np.sum(numerator, axis=1)[:, np.newaxis]
#    softmax = numerator/denominator
#    return softmax

# LARGE T


def logsoftmax(x, T=1.0):  # Added T parameter for temperature scaling
    # Clip x/T to prevent overflow/underflow
    scaled_x = np.clip(x/T, -709.78, 709.78)  # log(float.max) ≈ 709.78

    # LogSoftMax Implementation
    max_x = np.max(scaled_x, axis=1)[:, np.newaxis]
    exp_x = np.exp(scaled_x - max_x)
    sum_exp_x = np.sum(exp_x, axis=1)[:, np.newaxis]
    log_sum_exp_x = np.log(sum_exp_x + 1e-12)  # Added small epsilon
    max_plus_log_sum_exp_x = max_x + log_sum_exp_x
    log_probs = scaled_x - max_plus_log_sum_exp_x

    # Recover probs
    exp_log_probs = np.exp(log_probs)
    sum_log_probs = np.sum(exp_log_probs, axis=1)[:, np.newaxis]
    probs = exp_log_probs / (sum_log_probs + 1e-12)  # Added small epsilon

    return probs


def ql_step(P, Q, T, alpha):
    Q = (1 - alpha) * Q + alpha * P
    x = logsoftmax(Q, T)  # Pass T to logsoftmax
    return x, Q


def run_ql(x, Q, n_iter, edgeset, G, n_agents, n_actions, T):

    alpha = 0.1
    allx = np.zeros((n_iter, n_agents * n_actions))

    for c_iter in range(n_iter):
        P = get_payoffs(x, edgeset, G, n_agents, n_actions)
        x, Q = ql_step(P, Q, T, alpha)
        allx[c_iter] = x.reshape((n_agents*n_actions))

    return x, Q, allx

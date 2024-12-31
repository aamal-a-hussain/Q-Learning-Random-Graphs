import numpy as np
from networkgames import get_payoffs

def check_var(traj, window_size, tol):
    return np.mean(np.var(traj[:, -window_size:], axis=1)) <= tol


def check_rel_diff(traj, window_size, tol):
    rel_diff = np.max(traj[:, -window_size:], axis=1) - np.min(traj[:, -window_size:], axis=1)
    rel_diff /= np.max(traj[:, -window_size:], axis=1)

    return np.all(rel_diff <= tol)

def init_strats(n_agents, n_actions):
    return np.random.dirichlet(np.ones(n_actions), size=n_agents)

def ql_step(P, Q, T, alpha):
    Q = (1 - alpha) * Q + alpha * P
    x = np.exp(Q / T) / np.sum(np.exp(Q / T), axis=1)[:, np.newaxis]
    return x, Q

def run_ql(x, Q, n_iter, edgeset, G, n_agents, n_actions, T):

    alpha = 0.1
    allx = np.zeros((n_iter, n_agents * n_actions))

    for c_iter in range(n_iter):
        P = get_payoffs(x, edgeset, G, n_agents, n_actions)
        x, Q = ql_step(P, Q, T, alpha)
        allx[c_iter] = x.reshape((n_agents*n_actions))

    return x, Q, allx
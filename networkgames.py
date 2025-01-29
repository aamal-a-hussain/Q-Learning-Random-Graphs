import numpy as np


def mismatching(M=2, n_edges=3):
    A = np.dstack([np.array([[0, 1], [M, 0]])] * n_edges)
    B = np.zeros((2, 2, n_edges))

    return A, B


def chakraborty(S=7.0, T=8.5, n_edges=3):
    A = np.dstack([np.array([[1, S], [T, 0]])] * n_edges)
    B = np.zeros((2, 2, n_edges))

    return A, B


def shapley(beta=0.2, n_edges=3):
    A = np.dstack(
        [np.array([[1, 0, beta], [beta, 1, 0], [0, beta, 1]])]*n_edges)
    B = np.dstack(
        [np.array([[-beta, 1, 0], [0, -beta, 1], [1, 0, -beta]])]*n_edges)

    return A, B


def sato(ex=0.5, ey=-0.3, n_edges=3):
    A = np.dstack(
        [np.array([[ex, -1, 1], [1, ex, -1], [-1, 1, ex]])] * n_edges)
    B = np.dstack(
        [np.array([[ey, -1, 1], [1, ey, -1], [-1, 1, ey]])] * n_edges)

    return A, B


def generate_edgeset(network):
    return list(network.edges())


def conflict(edgeset, n_edges, n_agents=5, n_actions=2):
    A = np.zeros((n_actions, n_actions, n_edges))
    B = np.zeros((n_actions, n_actions, n_edges))
    v = np.random.uniform(0, 1, size=n_agents)
    for e, edge in enumerate(edgeset):
        c_1 = np.random.uniform(-1, 1, size=(n_actions, 1))
        c_2 = np.random.uniform(-1, 1, size=(1, n_actions))
        P = np.random.uniform(-5, 5, size=(n_actions, n_actions))
        A_kl = v[edge[0]] * P + c_1
        A_lk = v[edge[1]] * (1 - P) + c_2
        A[:, :, e] = A_kl
        B[:, :, e] = A_lk
    return A, B


def get_payoffs(x, edgeset, G, n_agents, n_actions):
    P = np.zeros((n_agents, n_actions))
    A, B = G
    for cI, e in enumerate(edgeset):
        P[e[0]] += np.einsum('ij,j->i', A[:, :, cI], x[e[1]])
        P[e[1]] += np.einsum('ij,j->i', B[:, :, cI], x[e[0]])

    return P

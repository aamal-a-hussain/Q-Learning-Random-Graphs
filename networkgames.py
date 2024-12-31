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
    A = np.dstack([np.array([[1, 0, beta], [beta, 1, 0], [0, beta, 1]])]*n_edges)
    B = np.dstack([np.array([[-beta, 1, 0], [0, -beta, 1], [1, 0, -beta]])]*n_edges)

    return A, B

def sato(ex=0.1,ey=-0.05,n_edges=3):
    A = np.dstack([np.array([[ex, -1, 1], [1, ex, -1], [-1, 1, ex]])] * n_edges)
    B = np.dstack([np.array([[ey, -1, 1], [1, ey, -1], [-1, 1, ey]])] * n_edges)

    return A, B

def generate_edgeset(network):
    return list(network.edges())

def get_payoffs(x, edgeset, G, n_agents, n_actions):
    P = np.zeros((n_agents, n_actions))
    A, B = G
    for cI, e in enumerate(edgeset):
        P[e[0]] += np.einsum('ij,j->i', A[:, :, cI], x[e[1]])
        P[e[1]] += np.einsum('ij,j->i', B[:, :, cI], x[e[0]])

    return P
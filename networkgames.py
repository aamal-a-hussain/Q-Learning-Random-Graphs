import numpy as np
import networkx as nx

from parameters import NetworkParameters


def mismatching(M=2, n_edges=3):
    A = np.dstack([np.array([[0, 1], [M, 0]])] * n_edges)
    B = np.zeros((2, 2, n_edges))

    return A, B


def chakraborty(S=7.0, T=8.5, n_edges=3):
    A = np.dstack([np.array([[1, S], [T, 0]])] * n_edges)
    B = np.zeros((2, 2, n_edges))

    return A, B


def shapley(beta=0.2, n_edges=3):
    A = np.dstack([np.array([[1, 0, beta], [beta, 1, 0], [0, beta, 1]])] * n_edges)
    B = np.dstack([np.array([[-beta, 1, 0], [0, -beta, 1], [1, 0, -beta]])] * n_edges)

    return A, B


def sato(ex=0.1, ey=-0.05, n_edges=3):
    A = np.dstack([np.array([[ex, -1, 1], [1, ex, -1], [-1, 1, ex]])] * n_edges)
    B = np.dstack([np.array([[ey, -1, 1], [1, ey, -1], [-1, 1, ey]])] * n_edges)

    return A, B


def generate_edgeset(network):
    return list(network.edges())


def conflict(edgeset, n_edges, n_agents=5, n_actions=2):
    A = np.zeros((n_actions, n_actions, n_edges))
    B = np.zeros((n_actions, n_actions, n_edges))
    v = np.random.uniform(0, 1, size=n_agents)
    P = np.random.uniform(-5, 5, size=(n_edges, n_actions, n_actions))
    for e, edge in enumerate(edgeset):
        c_1 = np.random.uniform(-1, 1, size=(n_actions, 1))
        c_2 = np.random.uniform(-1, 1, size=(1, n_actions))
        A_kl = v[edge[0]] * P[e] + c_1
        A_lk = v[edge[1]] * (1 - P[e]) + c_2
        A[:, :, e] = A_kl
        B[:, :, e] = A_lk
    return A, B


def get_payoffs(x, edgeset, G, n_agents, n_actions):
    P = np.zeros((n_agents, n_actions))
    A, B = G
    for cI, e in enumerate(edgeset):
        P[e[0]] += np.einsum("ij,j->i", A[:, :, cI], x[e[1]])
        P[e[1]] += np.einsum("ij,j->i", B[:, :, cI], x[e[0]])

    return P


def generate_sbm_network(n_agents: int, params: NetworkParameters):
    if params.network_type != "sbm":
        raise ValueError("Network Type must be 'sbm' to initialise an SBM network")

    if params.p is None or (params.p_min is None and params.p_max is None):
        raise ValueError("Either p or p_min, p_max must be set for SBM experiments")

    if params.p is not None and (params.p_min is not None or params.p_max is not None):
        raise ValueError("Cannot set both p and p_min,p_max")

    community_size = n_agents // params.n_blocks
    sizes = [community_size] * params.n_blocks

    if params.p is not None:
        block_probability_matrix = params.q * np.ones(
            (params.n_blocks, params.n_blocks)
        ) + (params.p - params.q) * np.eye(params.n_blocks)

    else:
        Ps = np.linspace(params.p_min, params.p_max, num=params.n_blocks)
        block_probability_matrix = (
            params.q * np.ones((params.n_blocks, params.n_blocks))
            - params.q * np.eye(params.n_blocks)
            + np.diag(Ps)
        )

    block_probabilities = [list(p) for p in block_probability_matrix]

    return nx.stochastic_block_model(sizes, block_probabilities)


def generate_er_network(n_agents, params: NetworkParameters):
    if params.network_type != "er":
        raise ValueError(
            "Network Type must be 'er' to initialise an Erdos-Renyi network"
        )

    if params.p is None:
        raise ValueError("p must be set for Erdos-Renyi experiments")
    return nx.erdos_renyi_graph(n_agents, params.p)

#import jax.numpy as jnp
import numpy as np
from jax import jit


class NetworkGame:
    n_agents: int
    n_actions: int
    payoff_A: np.ndarray | None = None
    payoff_B: np.ndarray | None = None

    def get_payoffs(self, x, edgeset):
        assert self.payoff_A is not None and self.payoff_B is not None

        assert len(edgeset) == self.payoff_A.shape[-1]
        assert len(edgeset) == self.payoff_B.shape[-1]

        x = np.array(x)
        edges = np.array(edgeset)
        return _get_payoffs_general(#_get_jit_payoffs(
            x, edges, self.payoff_A, self.payoff_B, self.n_agents, self.n_actions
        )


class ShapleyGame(NetworkGame):
    name = "shapley"
    n_actions = 3

    def __init__(self, n_agents, beta=0.2):
        assert n_agents > 2
        self.n_agents = n_agents
        self.payoff_A = np.array([[1, 0, beta], [beta, 1, 0], [0, beta, 1]])
        self.payoff_B = np.array([[-beta, 1, 0], [0, -beta, 1], [1, 0, -beta]])

    def get_payoffs(self, x, edgeset):
        assert self.payoff_A is not None and self.payoff_B is not None

        x = np.array(x)
        edges = np.array(edgeset)
        return _get_payoffs_no_edges(#_get_jit_payoffs_no_edges(
            x, edges, self.payoff_A, self.payoff_B, self.n_agents, self.n_actions
        )


class SatoGame(NetworkGame):
    name = "sato"
    n_actions = 3

    def __init__(self, n_agents, ex=0.1, ey= -0.05):
        assert n_agents > 2
        self.n_agents = n_agents
        self.payoff_A = np.array([[ex, -1, 1], [1, ex, -1], [-1, 1, ex]])
        self.payoff_B = np.array([[ey, -1, 1], [1, ey, -1], [-1, 1, ey]])

    def get_payoffs(self, x, edgeset):
        assert self.payoff_A is not None and self.payoff_B is not None

        x = np.array(x)
        edges = np.array(edgeset)
        return _get_payoffs_no_edges(
            x, edges, self.payoff_A, self.payoff_B, self.n_agents, self.n_actions
        )


class ConflictGame(NetworkGame):
    name = "conflict"

    def __init__(self, n_agents, n_actions, edgeset):
        assert n_agents > 2
        assert len(edgeset) > 1
        assert n_actions > 1

        self.n_agents = n_agents
        self.n_actions = n_actions

        self.create_payoffs(n_agents, n_actions, edgeset)

    def create_payoffs(self, n_agents, n_actions, edgeset):
        n_edges = len(edgeset)
        self.payoff_A = np.zeros((n_actions, n_actions, n_edges))
        self.payoff_B = np.zeros((n_actions, n_actions, n_edges))
        v = np.random.uniform(0, 1, size=n_agents)
        P = np.random.uniform(-5, 5, size=(n_edges, n_actions, n_actions))
        for e, edge in enumerate(edgeset):
            c_1 = np.random.uniform(-1, 1, size=(n_actions, 1))
            c_2 = np.random.uniform(-1, 1, size=(1, n_actions))
            A_kl = v[edge[0]] * P[e] + c_1
            A_lk = v[edge[1]] * (1 - P[e]) + c_2
            self.payoff_A[:, :, e] = A_kl
            self.payoff_B[:, :, e] = A_lk


def _get_payoffs_general(x, edges, payoff_A, payoff_B, n_agents, n_actions):
    #P = jnp.zeros((n_agents, n_actions))
    
    P = np.zeros((n_agents, n_actions))

    agent_0_idx = edges[:, 0]
    agent_1_idx = edges[:, 1]

    x_nbr_0 = x[agent_1_idx]
    x_nbr_1 = x[agent_0_idx]
    
    # jax jitting might improve speed
    contrib_0 = np.einsum("ije, je -> ie", payoff_A, x_nbr_0.T)
    contrib_1 = np.einsum("ije, je -> ie", payoff_B, x_nbr_1.T)

    #P = P.at[agent_0_idx].add(contrib_0.T)
    #P = P.at[agent_1_idx].add(contrib_1.T)

    P[agent_0_idx] += contrib_0.T
    P[agent_1_idx] += contrib_1.T


    return P


def _get_payoffs_no_edges(x, edges, payoff_A, payoff_B, n_agents, n_actions):
    #P = jnp.zeros((n_agents, n_actions))
    P = np.zeros((n_agents, n_actions))

    agent_0_idx = edges[:, 0]
    agent_1_idx = edges[:, 1]

    x_nbr_0 = x[agent_1_idx]
    x_nbr_1 = x[agent_0_idx]

    contrib_0 = np.einsum("ij, je -> ie", payoff_A, x_nbr_0.T)
    contrib_1 = np.einsum("ij, je -> ie", payoff_B, x_nbr_1.T)

    #P = P.at[agent_0_idx].add(contrib_0.T)
    #P = P.at[agent_1_idx].add(contrib_1.T)
    
    P[agent_0_idx] += contrib_0.T
    P[agent_1_idx] += contrib_1.T


    return P


#_get_jit_payoffs = jit(_get_payoffs_general, static_argnames=("n_agents", "n_actions"))
#_get_jit_payoffs_no_edges = jit(
#    _get_payoffs_no_edges, static_argnames=("n_agents", "n_actions")
#)

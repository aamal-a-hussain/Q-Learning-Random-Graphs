import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import networkx as nx


def norm_check(network: nx.Graph):
    G = nx.to_numpy_array(network)

    G_kl = np.zeros_like(G)
    G_lk = np.zeros_like(G)

    for e in network.edges:
        if np.random.choice([0, 1]) == 1:
            G_kl[e[0], e[1]] = 1
            G_lk[e[1], e[0]] = 1
        else:
            G_kl[e[1], e[0]] = 1
            G_lk[e[0], e[1]] = 1


    assert np.all(G_kl + G_lk == G)

    return np.linalg.norm(G, 2) - 0.5 * (np.linalg.norm(G_lk, 2) + np.linalg.norm(G_kl, 2))


if __name__ == "__main__":
    norms = []
    for _ in tqdm(range(10000)):
        norms.append(norm_check(nx.erdos_renyi_graph(10, 0.5)))

    plt.hist(norms, bins=50)
    plt.show()

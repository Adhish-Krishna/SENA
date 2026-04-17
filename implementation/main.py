import argparse
import gzip
import os
import random

import networkx as nx
import numpy as np


np.random.seed(42)
random.seed(42)


def _ordered_nodes(G: nx.DiGraph):
    return list(sorted(G.nodes()))


def _build_matrices(G: nx.DiGraph, L: np.ndarray | None = None):
    nodes = _ordered_nodes(G)
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    N = len(nodes)

    W = np.zeros((N, N), dtype=float)
    if L is None:
        L = np.zeros((N, N), dtype=int)
        for v_i, v_j, data in G.edges(data=True):
            i = node_to_idx[v_i]
            j = node_to_idx[v_j]
            W[i][j] = float(data.get('weight', 1.0))
            L[i][j] = int(data.get('sign', 0))
    else:
        L = np.asarray(L, dtype=int)
        if L.shape != (N, N):
            raise ValueError('L must have shape (N, N)')
        for v_i, v_j, data in G.edges(data=True):
            i = node_to_idx[v_i]
            j = node_to_idx[v_j]
            W[i][j] = float(data.get('weight', 1.0))

    H_in = [set(G.predecessors(node)) for node in nodes]
    H_out = [set(G.successors(node)) for node in nodes]
    H_in_plus = [set() for _ in range(N)]
    H_in_minus = [set() for _ in range(N)]
    alpha = np.zeros(N, dtype=float)
    beta = np.zeros(N, dtype=float)

    for i, node in enumerate(nodes):
        in_neighbors = H_in[i]
        n_total = len(in_neighbors)
        if n_total == 0:
            continue

        n_pos = 0
        for v_j in in_neighbors:
            j = node_to_idx[v_j]
            if L[j][i] == 1:
                H_in_plus[i].add(v_j)
                n_pos += 1
            elif L[j][i] == -1:
                H_in_minus[i].add(v_j)

        alpha[i] = n_pos / n_total
        beta[i] = (n_total - n_pos) / n_total

    return nodes, node_to_idx, W, L, H_in, H_out, H_in_plus, H_in_minus, alpha, beta


def _as_vector(values, N: int, dtype=float):
    array = np.asarray(values, dtype=dtype).reshape(-1)
    if array.shape[0] != N:
        raise ValueError(f'Expected vector of length {N}')
    return array


def _cycle_bounds(T, node, i):
    if isinstance(T, dict):
        T_l_i, T_u_i = T[node]
    else:
        T_l_i, T_u_i = T[i]
    return int(T_l_i), int(T_u_i)


# Algorithm 2 - BUpdate: Belief Update Algorithm
def BUpdate(G: nx.DiGraph, X_t: np.ndarray, A_t: np.ndarray, L: np.ndarray, S_t: set):
    """
    Belief Update Algorithm (BUpdate).

    Inputs:
        G   : signed social network
        X_t : belief vector at time t
        A_t : attitude vector at time t
        L   : label matrix
        S_t : seed set at time t

    Output:
        X_{t+1}, A_{t+1}
    """
    nodes, node_to_idx, W, L, _, _, H_in_plus, H_in_minus, alpha, beta = _build_matrices(G, L)
    N = len(nodes)

    X_t = _as_vector(X_t, N, dtype=float)
    A_t = _as_vector(A_t, N, dtype=int)
    X_t1 = X_t.copy()
    A_t1 = A_t.copy()

    S_t = set(S_t)
    S_idx = {node_to_idx[v_i] for v_i in S_t if v_i in node_to_idx}

    for i, v_i in enumerate(nodes):
        if i in S_idx or A_t[i] == 1:
            X_t1[i] = X_t[i]
            A_t1[i] = A_t[i]
            continue

        Pos = 0.0
        Neg = 0.0

        for v_j in H_in_plus[i]:
            j = node_to_idx[v_j]
            if j in S_idx:
                Pos += W[j][i] * (X_t[j] - X_t[i])

        for v_j in H_in_minus[i]:
            j = node_to_idx[v_j]
            if j in S_idx:
                Neg += W[j][i] * (X_t[j] - X_t[i])

        X_t1[i] = X_t[i] + alpha[i] * Pos - beta[i] * Neg
        X_t1[i] = float(np.clip(X_t1[i], 0.0, 1.0))
        A_t1[i] = 1 if random.random() < X_t1[i] else 0

    return X_t1, A_t1


# Algorithm 3 - SPR: Signed-PageRank
def SPR(G: nx.DiGraph, X_0: np.ndarray, L: np.ndarray, k: int, d: float = 0.85, max_iter: int = 200):
    """
    Signed-PageRank (SPR).

    Inputs:
        G    : signed social network
        X_0  : initial belief vector
        L    : label matrix
        k    : number of seed nodes
        d    : damping coefficient

    Output:
        S_0 : list of top-k seed nodes
    """
    nodes, _, W, L, _, H_out, _, _, _, _ = _build_matrices(G, L)
    N = len(nodes)

    X_0 = _as_vector(X_0, N, dtype=float)

    W_hat = np.zeros((N, N), dtype=float)
    for j in range(N):
        col_sum = float(np.sum(W[:, j]))
        if col_sum > 0.0:
            W_hat[:, j] = W[:, j] / col_sum

    Y = d * (W_hat * L)

    SPR_t = X_0.copy()
    Sort_t = np.arange(N)
    Sort_next = np.argsort(-SPR_t)

    iteration = 0
    while not np.array_equal(Sort_next, Sort_t):
        Sort_t = Sort_next
        diff_matrix = SPR_t[:, None] - SPR_t[None, :]
        SPR_t1 = (diff_matrix * Y).sum(axis=1) + (1.0 - d) / N
        SPR_t = SPR_t1
        Sort_next = np.argsort(-SPR_t)
        iteration += 1
        if iteration >= max_iter:
            break

    return [nodes[i] for i in Sort_t[:k]]

#Algorithm 1 - InPro : Information Propagation Algorithm
def InPro(G: nx.DiGraph, k: int, X_t: np.ndarray, A_t: np.ndarray, L: np.ndarray, T, t: int):
    """
    InPro: Information Propagation.

    Returns the final belief vector, attitude vector, and the initial seed set.
    """
    nodes, node_to_idx, _, L, _, _, _, _, _, _ = _build_matrices(G, L)
    N = len(nodes)

    X_t = _as_vector(X_t, N, dtype=float)
    A_t = _as_vector(A_t, N, dtype=int)

    if t == 0:
        S_0 = SPR(G, X_t, L, k)
        for v_i in S_0:
            A_t[node_to_idx[v_i]] = 1
    else:
        S_0 = [nodes[i] for i in range(N) if A_t[i] == 1]

    while True:
        S_t = {nodes[i] for i in range(N) if A_t[i] == 1}

        F_a = set()
        F_b = set()
        for i, v_i in enumerate(nodes):
            if v_i in S_t or A_t[i] != 0:
                continue
            T_l_i, T_u_i = _cycle_bounds(T, v_i, i)
            if T_l_i <= t <= T_u_i:
                F_a.add(v_i)
            elif t < T_l_i:
                F_b.add(v_i)

        if not F_a and F_b:
            t += 1
            continue

        if F_a:
            X_t1, A_t1 = BUpdate(G, X_t, A_t, L, S_t)

            newly_infected = [i for i, v_i in enumerate(nodes) if v_i not in S_t and A_t[i] == 0 and A_t1[i] == 1]
            for i in newly_infected:
                p = 1.0 - float(X_t1[i])
                if random.random() < p:
                    A_t1[i] = 0

            X_t = X_t1
            A_t = A_t1
            t += 1
            continue

        break

    infected_count = int(np.sum(A_t == 1))
    return X_t, A_t, S_0, infected_count


def _build_epinions_dataset(max_nodes: int = 500, k: int = 10, seed: int = 42):
    rng = np.random.RandomState(seed)
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'soc-sign-epinions.txt.gz')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f'Epinions dataset not found at: {data_path}')

    G_full = nx.DiGraph()
    with gzip.open(data_path, 'rt', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            v_i = int(parts[0])
            v_j = int(parts[1])
            sign = int(parts[2])
            if v_i != v_j and sign in (-1, 1):
                G_full.add_edge(v_i, v_j, sign=sign)

    if G_full.number_of_nodes() > max_nodes:
        degree_map = dict(G_full.degree())
        top_nodes = sorted(degree_map, key=degree_map.get, reverse=True)[:max_nodes]
        G = G_full.subgraph(top_nodes).copy()
    else:
        G = G_full.copy()

    mapping = {old: new for new, old in enumerate(sorted(G.nodes()))}
    G = nx.relabel_nodes(G, mapping)

    for v_i, v_j in G.edges():
        G[v_i][v_j]['weight'] = float(rng.beta(2, 5))

    N = G.number_of_nodes()
    X_0 = rng.beta(2, 3, size=N).astype(float)
    A_0 = np.zeros(N, dtype=int)
    T_l = rng.randint(0, 5, size=N)
    T_u = T_l + rng.randint(2, 12, size=N)
    T = [(int(T_l[i]), int(T_u[i])) for i in range(N)]

    _, _, _, L, _, _, _, _, _, _ = _build_matrices(G, None)
    return G, X_0, A_0, L, T, k


def main():
    parser = argparse.ArgumentParser(description='Run Signed-PageRank on Epinions dataset')
    parser.add_argument('--max-nodes', type=int, default=500)
    parser.add_argument('--k', type=int, default=10)
    args = parser.parse_args()

    G, X_0, A_0, L, T, k = _build_epinions_dataset(max_nodes=args.max_nodes, k=args.k)
    X_t, A_t, S_0, infected_count = InPro(G, k, X_0, A_0, L, T, 0)

    print('nodes =', G.number_of_nodes(), 'edges =', G.number_of_edges())
    print('S_0 =', S_0)
    print('X_final =', np.round(X_t, 6).tolist())
    print('A_final =', A_t.tolist())
    print('infected_count =', infected_count)


if __name__ == '__main__':
    main()

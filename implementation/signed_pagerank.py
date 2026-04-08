"""
Signed-PageRank: An Efficient Influence Maximization Framework
for Signed Social Networks
Yin et al., IEEE TKDE 2021

Implementation of:
  - Information Propagation Model (Algorithm 1 - InPro)
  - Belief Update Rules          (Algorithm 2 - BUpdate)
  - Signed-PageRank              (Algorithm 3 - SPR)

Datasets (--dataset flag):
  synthetic  — Scale-free signed network (default)
  epinions   — Epinions trust network    (SNAP)
  slashdot   — Slashdot social network   (SNAP)
  wiki       — Wikipedia RfA votes       (SNAP)
"""

import argparse
import gzip
import os
import urllib.request

import numpy as np
import networkx as nx
import random
import time

np.random.seed(42)
random.seed(42)

# ─────────────────────────────────────────────────────────────────
# 1.  DATASET LOADING
# ─────────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

SNAP_URLS = {
    'epinions': 'https://snap.stanford.edu/data/soc-sign-epinions.txt.gz',
    'slashdot': 'https://snap.stanford.edu/data/soc-sign-Slashdot081106.txt.gz',
    'wiki':     'https://snap.stanford.edu/data/wiki-RfA.txt.gz',
}


def _download_if_needed(dataset_name: str) -> str:
    """Download SNAP dataset if not already cached locally."""
    os.makedirs(DATA_DIR, exist_ok=True)
    filename = SNAP_URLS[dataset_name].split('/')[-1]
    filepath = os.path.join(DATA_DIR, filename)

    if not os.path.exists(filepath):
        url = SNAP_URLS[dataset_name]
        print(f"[Download] {url}  →  {filepath}")
        urllib.request.urlretrieve(url, filepath)
        print(f"[Download] Done.")
    else:
        print(f"[Cache] Using cached {filepath}")

    return filepath


def _assign_node_attributes(G: nx.DiGraph, rng: np.random.RandomState):
    """Assign belief, recommendation cycle to every node."""
    for node in G.nodes():
        G.nodes[node]['belief']    = float(rng.beta(2, 3))
        G.nodes[node]['rec_start'] = int(rng.randint(0, 5))
        G.nodes[node]['rec_end']   = G.nodes[node]['rec_start'] + int(rng.randint(2, 12))


def _assign_edge_weights(G: nx.DiGraph, rng: np.random.RandomState):
    """Assign random influence weights to edges that don't have them."""
    for u, v, data in G.edges(data=True):
        if 'weight' not in data:
            G[u][v]['weight'] = float(rng.beta(2, 5))


def load_snap_signed(dataset_name: str, max_nodes: int = 5000,
                     seed: int = 42) -> nx.DiGraph:
    """
    Load a signed network from SNAP.

    Format (Epinions / Slashdot): tab-separated  FromNodeId  ToNodeId  Sign
    Format (Wiki RfA): structured text blocks with SRC, TGT, VOT fields.

    For tractability the graph is subsampled to `max_nodes` nodes.
    """
    rng = np.random.RandomState(seed)
    filepath = _download_if_needed(dataset_name)

    G_full = nx.DiGraph()

    if dataset_name in ('epinions', 'slashdot'):
        with gzip.open(filepath, 'rt', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    u, v, s = int(parts[0]), int(parts[1]), int(parts[2])
                    if u != v:
                        G_full.add_edge(u, v, sign=s)

    elif dataset_name == 'wiki':
        # Wiki RfA: structured records with SRC, TGT, VOT fields
        with gzip.open(filepath, 'rt', encoding='utf-8', errors='ignore') as f:
            text = f.read()

        # Parse the record-based format
        name_to_id = {}
        next_id = 0
        src, tgt, vot = None, None, None

        for line in text.split('\n'):
            line = line.strip()
            if line.startswith('SRC:'):
                src = line[4:].strip()
            elif line.startswith('TGT:'):
                tgt = line[4:].strip()
            elif line.startswith('VOT:'):
                try:
                    vot = int(line[4:].strip())
                except ValueError:
                    vot = None
            elif line == '' and src and tgt and vot is not None:
                if src != tgt and vot in (-1, 1):
                    if src not in name_to_id:
                        name_to_id[src] = next_id
                        next_id += 1
                    if tgt not in name_to_id:
                        name_to_id[tgt] = next_id
                        next_id += 1
                    G_full.add_edge(name_to_id[src], name_to_id[tgt], sign=vot)
                src, tgt, vot = None, None, None

    # Subsample for tractability
    if G_full.number_of_nodes() > max_nodes:
        print(f"[Subsample] {G_full.number_of_nodes()} nodes → {max_nodes}")
        # Take the top-degree subgraph for a meaningful sample
        degrees = dict(G_full.degree())
        top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:max_nodes]
        G = G_full.subgraph(top_nodes).copy()
        # Relabel to contiguous 0..N-1
        mapping = {old: new for new, old in enumerate(sorted(G.nodes()))}
        G = nx.relabel_nodes(G, mapping)
    else:
        G = G_full.copy()
        mapping = {old: new for new, old in enumerate(sorted(G.nodes()))}
        G = nx.relabel_nodes(G, mapping)

    _assign_edge_weights(G, rng)
    _assign_node_attributes(G, rng)

    return G


def build_synthetic_network(n_nodes: int = 500,
                            neg_fraction: float = 0.147,
                            seed: int = 42) -> nx.DiGraph:
    """
    Build a scale-free signed directed graph (synthetic dataset).

    Mimics the synthetic setup from the paper:
      - Scale-free topology (preferential attachment)
      - ~14.7% negative edges (similar to Epinions)
      - Edge weights ∈ (0, 1]
      - Node beliefs x_{i,0} ~ Beta(2, 3)
    """
    rng = np.random.RandomState(seed)
    random.seed(seed)

    G_raw = nx.scale_free_graph(n_nodes, seed=seed)
    G = nx.DiGraph()
    G.add_nodes_from(range(n_nodes))

    seen = set()
    for u, v in G_raw.edges():
        if u != v and (u, v) not in seen:
            seen.add((u, v))
            w = float(rng.beta(2, 5))
            s = -1 if rng.random() < neg_fraction else 1
            G.add_edge(u, v, weight=w, sign=s)

    _assign_node_attributes(G, rng)

    return G


def load_dataset(name: str, max_nodes: int = 5000, seed: int = 42) -> nx.DiGraph:
    """Load dataset by name. Dispatcher for synthetic / real datasets."""
    if name == 'synthetic':
        G = build_synthetic_network(n_nodes=500, seed=seed)
    elif name in SNAP_URLS:
        G = load_snap_signed(name, max_nodes=max_nodes, seed=seed)
    else:
        raise ValueError(f"Unknown dataset: {name}. "
                         f"Choose from: synthetic, {', '.join(SNAP_URLS.keys())}")

    n_neg = sum(1 for _, _, d in G.edges(data=True) if d['sign'] == -1)
    n_edges = G.number_of_edges()
    print(f"[Dataset: {name}]  Nodes={G.number_of_nodes()}  "
          f"Edges={n_edges}  "
          f"Neg%={n_neg / max(n_edges, 1) * 100:.1f}%  "
          f"Density={nx.density(G):.5f}")

    return G


# ─────────────────────────────────────────────────────────────────
# 2.  BELIEF UPDATE  (Section 3.2 / Algorithm 2 — BUpdate)
# ─────────────────────────────────────────────────────────────────

def compute_embeddedness(G: nx.DiGraph, node: int):
    """
    Eq. (3):
      α_i = |H^{in+}_i| / |H^{in}_i|    (positive embeddedness)
      β_i = |H^{in−}_i| / |H^{in}_i|    (negative embeddedness)
    """
    in_neighbors = list(G.predecessors(node))
    n_total = len(in_neighbors)
    if n_total == 0:
        return 0.0, 0.0
    n_pos = sum(1 for u in in_neighbors if G[u][node]['sign'] == 1)
    alpha = n_pos / n_total
    beta  = (n_total - n_pos) / n_total
    return alpha, beta


def belief_update(G: nx.DiGraph,
                  beliefs: np.ndarray,
                  attitudes: np.ndarray,
                  seed_set: set,
                  t: int) -> np.ndarray:
    """
    Algorithm 2 — BUpdate.
    Proposition 2 (parallel recommendation belief update):

      x_{i,t+1} = x_{i,t}
                   + α_i · Σ_{j ∈ H^{in+}_i ∩ S_t}  w_{j,i} · (x_{j,t} − x_{i,t})
                   − β_i · Σ_{j ∈ H^{in−}_i ∩ S_t}  w_{j,i} · (x_{j,t} − x_{i,t})
    """
    new_beliefs = beliefs.copy()

    for node in G.nodes():
        if attitudes[node] == 1:
            continue

        rec_start = G.nodes[node]['rec_start']
        rec_end   = G.nodes[node]['rec_end']
        if not (rec_start <= t <= rec_end):
            continue

        alpha, beta = compute_embeddedness(G, node)

        pos_sum = 0.0
        neg_sum = 0.0

        for u in G.predecessors(node):
            if u not in seed_set:
                continue
            w    = G[u][node]['weight']
            s    = G[u][node]['sign']
            diff = beliefs[u] - beliefs[node]
            if s == 1:
                pos_sum += w * diff
            else:
                neg_sum += w * diff

        new_beliefs[node] = beliefs[node] + alpha * pos_sum - beta * neg_sum

    return new_beliefs


# ─────────────────────────────────────────────────────────────────
# 3.  INFORMATION PROPAGATION  (Section 3.1 / Algorithm 1 — InPro)
# ─────────────────────────────────────────────────────────────────

def run_propagation(G: nx.DiGraph,
                    initial_seeds: list,
                    max_rounds: int = 30) -> dict:
    """
    Algorithm 1 — InPro (extended SIR/IC for signed networks).

    States:
       0 = susceptible
       1 = infectious (accepted)
      −1 = removed

    Returns: dict with final_infected count, per-round history, etc.
    """
    N = G.number_of_nodes()
    beliefs   = np.array([G.nodes[n]['belief'] for n in range(N)])
    attitudes = np.zeros(N, dtype=int)

    for s in initial_seeds:
        attitudes[s] = 1

    history = {
        'infected_per_round': [],
        'total_infected':     [],
    }

    t = 0
    for t in range(1, max_rounds + 1):
        seed_set = set(np.where(attitudes == 1)[0])

        susceptible_active = [
            v for v in G.nodes()
            if attitudes[v] == 0
            and G.nodes[v]['rec_start'] <= t <= G.nodes[v]['rec_end']
        ]
        susceptible_waiting = [
            v for v in G.nodes()
            if attitudes[v] == 0
            and t < G.nodes[v]['rec_start']
        ]

        if not susceptible_active and not susceptible_waiting:
            break

        if susceptible_active:
            beliefs = belief_update(G, beliefs, attitudes, seed_set, t)

            for v in susceptible_active:
                # The paper models attitude as B(1, x_{i,t}). We keep the
                # updated belief itself unchanged and only clip the sampling
                # probability for numerical safety on synthetic/random data.
                accept_prob = float(np.clip(beliefs[v], 0.0, 1.0))
                new_att = int(np.random.binomial(1, accept_prob))
                if new_att == 1:
                    recovery_prob = float(np.clip(1.0 - beliefs[v], 0.0, 1.0))
                    if np.random.random() < recovery_prob:
                        attitudes[v] = 0
                    else:
                        attitudes[v] = 1

            for v in G.nodes():
                if attitudes[v] == 0 and t > G.nodes[v]['rec_end']:
                    attitudes[v] = -1

        total_inf = int(np.sum(attitudes == 1))
        history['infected_per_round'].append(total_inf - len(initial_seeds))
        history['total_infected'].append(total_inf)

    history['final_infected'] = int(np.sum(attitudes == 1))
    history['rounds']         = t
    return history


# ─────────────────────────────────────────────────────────────────
# 4.  SIGNED-PAGERANK  (Section 3.3 / Algorithm 3 — SPR)
# ─────────────────────────────────────────────────────────────────

def build_SPR_matrix(G: nx.DiGraph, damping: float = 0.85) -> np.ndarray:
    """
    Build the Signed-PageRank transition matrix.

    Eq. (10):  Y = d · (W̃ ⊙ L)

    where:
      W̃ = row-normalised weight matrix
      L  = sign label matrix
      ⊙  = Hadamard (element-wise) product
      d  = damping factor
    """
    N = G.number_of_nodes()
    W = np.zeros((N, N))
    L = np.zeros((N, N))

    for u, v, data in G.edges(data=True):
        W[u, v] = data['weight']
        L[u, v] = data['sign']

    # Algorithm 3 normalises each node's outgoing weights so that
    # sum_j W_tilde[i, j] = 1 for each source node i. In this matrix
    # layout that is row-wise normalisation of W[u, v].
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    W_tilde = W / row_sums

    Y = damping * (W_tilde * L)
    return Y


def signed_pagerank(G: nx.DiGraph,
                    k: int,
                    damping: float = 0.85,
                    max_iter: int = 200) -> list:
    """
    Algorithm 3 — SPR: Signed-PageRank.

    Rank update (Eq. 9):
      SPR_{i,t+1} = Σ_{j ∈ H^out_i} (SPR_{i,t} − SPR_{j,t}) · y_{i,j}
                    + (1−d) / N

    Convergence: ranking order stops changing.

    Returns: top-k seed nodes ordered by SPR score.
    """
    N = G.number_of_nodes()
    Y = build_SPR_matrix(G, damping)

    # Initialise SPR scores with node beliefs x_{i,0}
    spr = np.array([G.nodes[n]['belief'] for n in range(N)], dtype=float)
    current_order = np.argsort(-spr)

    for iteration in range(max_iter):
        # Vectorised: diff[i,j] = spr[i] - spr[j]
        diff    = spr[:, None] - spr[None, :]
        spr_new = (diff * Y).sum(axis=1) + (1.0 - damping) / N

        new_order = np.argsort(-spr_new)

        if np.array_equal(new_order, current_order):
            print(f"[SPR] Converged in {iteration + 1} iterations")
            break

        spr           = spr_new
        current_order = new_order
    else:
        print(f"[SPR] Reached max iterations ({max_iter})")

    seed_nodes = list(current_order[:k])
    return seed_nodes


# ─────────────────────────────────────────────────────────────────
# 5.  MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Signed-PageRank (Yin et al., IEEE TKDE 2021)")
    parser.add_argument(
        '--dataset', type=str, default='synthetic',
        choices=['synthetic', 'epinions', 'slashdot', 'wiki'],
        help='Dataset to use (default: synthetic)')
    parser.add_argument(
        '--k', type=int, default=10,
        help='Number of seed nodes to select (default: 10)')
    parser.add_argument(
        '--damping', type=float, default=0.85,
        help='Damping factor d (default: 0.85)')
    parser.add_argument(
        '--max-nodes', type=int, default=5000,
        help='Max nodes for real datasets (default: 5000)')
    parser.add_argument(
        '--mc-runs', type=int, default=50,
        help='Infomation propagation runs (default: 50)')
    parser.add_argument(
        '--max-rounds', type=int, default=25,
        help='Max propagation rounds (default: 25)')
    args = parser.parse_args()

    print("=" * 60)
    print("  Signed-PageRank  |  Yin et al., IEEE TKDE 2021")
    print("=" * 60)

    # ── Load dataset ─────────────────────────────────────────────
    print(f"\n[1/3] Loading dataset: {args.dataset}")
    G = load_dataset(args.dataset, max_nodes=args.max_nodes)

    # ── Run Signed-PageRank ──────────────────────────────────────
    print(f"\n[2/3] Running Signed-PageRank (k={args.k}, d={args.damping})")
    t0 = time.time()
    seeds = signed_pagerank(G, k=args.k, damping=args.damping)
    elapsed = time.time() - t0

    print(f"  Seed nodes : {seeds}")
    print(f"  Time       : {elapsed * 1000:.1f} ms")

    # ── Running Information Propagation ─────────────────────
    print(f"\n[3/3] Evaluating influence spread "
          f"({args.mc_runs} MC runs, max {args.max_rounds} rounds)...")
    infected_counts = []
    round_counts    = []

    for run in range(args.mc_runs):
        hist = run_propagation(G, seeds, max_rounds=args.max_rounds)
        infected_counts.append(hist['final_infected'])
        round_counts.append(hist['rounds'])

    avg_infected = np.mean(infected_counts)
    std_infected = np.std(infected_counts)
    avg_rounds   = np.mean(round_counts)

    # ── Results ──────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  RESULTS")
    print("─" * 60)
    print(f"  Dataset          : {args.dataset}")
    print(f"  Nodes            : {G.number_of_nodes()}")
    print(f"  Edges            : {G.number_of_edges()}")
    print(f"  k (seed nodes)   : {args.k}")
    print(f"  Damping factor   : {args.damping}")
    print(f"  Seed selection   : {elapsed * 1000:.1f} ms")
    print(f"  Avg infected     : {avg_infected:.1f} ± {std_infected:.1f}")
    print(f"  Avg rounds       : {avg_rounds:.1f}")
    print(f"  Influence ratio  : {avg_infected / G.number_of_nodes() * 100:.2f}%")
    print("─" * 60)


if __name__ == '__main__':
    main()

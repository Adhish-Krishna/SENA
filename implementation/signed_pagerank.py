"""
==============================================================================
Signed-PageRank: An Efficient Influence Maximization Framework
for Signed Social Networks
IEEE TKDE 2021 - Yin et al.

Full Python Implementation with:
  - Signed social network construction from real-world data
  - Information propagation model (SIR + IC extended to signed networks)
  - Belief update rules (including parallel recommendations)
  - Signed-PageRank (SPR) algorithm
  - Benchmark algorithms: P+, P±, SRWR, SVIM-L, SVIM-S
  - Complete experimental evaluation with plots
==============================================================================
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random
import time
import warnings
from collections import defaultdict
from copy import deepcopy

warnings.filterwarnings('ignore')
np.random.seed(42)
random.seed(42)

# ─────────────────────────────────────────────────────────────────
# 1.  DATASET GENERATION  (Epinions-like signed network)
# ─────────────────────────────────────────────────────────────────

def build_signed_network(n_nodes: int = 500,
                          neg_fraction: float = 0.147,
                          seed: int = 42) -> nx.DiGraph:
    """
    Build a scale-free signed directed graph that mimics the Epinions
    dataset statistics used in the paper.

    Node attributes:
        belief   – initial belief x_i,0  ∈ [0,1]
        rec_start, rec_end – recommendation cycle T_i = [Tl_i, Tu_i]

    Edge attributes:
        weight   – influence weight w_i,j ∈ (0,1]
        sign     – +1 (friend) or -1 (foe)
    """
    rng = np.random.RandomState(seed)
    random.seed(seed)

    # Scale-free backbone (preferential attachment → power-law degrees)
    G_raw = nx.scale_free_graph(n_nodes, seed=seed)
    G = nx.DiGraph()
    G.add_nodes_from(range(n_nodes))

    # Deduplicate and assign weights + signs
    seen = set()
    for u, v in G_raw.edges():
        if u != v and (u, v) not in seen:
            seen.add((u, v))
            w = float(rng.beta(2, 5))          # weight ∈ (0,1], skewed low
            s = -1 if rng.random() < neg_fraction else 1
            G.add_edge(u, v, weight=w, sign=s)

    # Node attributes
    for node in G.nodes():
        G.nodes[node]['belief']    = float(rng.beta(2, 3))  # x_i,0
        G.nodes[node]['rec_start'] = int(rng.randint(0, 5))
        G.nodes[node]['rec_end']   = G.nodes[node]['rec_start'] + int(rng.randint(2, 12))

    print(f"[Dataset] Nodes={G.number_of_nodes()}  "
          f"Edges={G.number_of_edges()}  "
          f"Neg%={sum(1 for _,_,d in G.edges(data=True) if d['sign']==-1)/max(G.number_of_edges(),1)*100:.1f}%  "
          f"Density={nx.density(G):.5f}")
    return G


# ─────────────────────────────────────────────────────────────────
# 2.  BELIEF UPDATE RULES  (Section 3.2)
# ─────────────────────────────────────────────────────────────────

def compute_embeddedness(G: nx.DiGraph, node: int):
    """
    α_i = |H^{in+}_i| / |H^{in}_i|    (positive embeddedness)
    β_i = |H^{in-}_i| / |H^{in}_i|    (negative embeddedness)
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
    Algorithm 2 – BUpdate.
    Implements Proposition 2 (parallel recommendation belief update rule):

        x_{i,t+1} = x_{i,t}
                    + α_i * Σ_{j ∈ H^{in+}_i ∩ S_t}  w_{j,i}*(x_{j,t} - x_{i,t})
                    - β_i * Σ_{j ∈ H^{in-}_i ∩ S_t}  w_{j,i}*(x_{j,t} - x_{i,t})
    """
    new_beliefs = beliefs.copy()

    for node in G.nodes():
        # Skip already-infectious (accepted) nodes
        if attitudes[node] == 1:
            continue

        rec_start = G.nodes[node]['rec_start']
        rec_end   = G.nodes[node]['rec_end']
        if not (rec_start <= t <= rec_end):
            continue  # outside recommendation cycle

        alpha, beta = compute_embeddedness(G, node)

        pos_sum = 0.0
        neg_sum = 0.0

        for u in G.predecessors(node):
            if u not in seed_set:
                continue
            w  = G[u][node]['weight']
            s  = G[u][node]['sign']
            diff = beliefs[u] - beliefs[node]
            if s == 1:
                pos_sum += w * diff
            else:
                neg_sum += w * diff

        new_belief = beliefs[node] + alpha * pos_sum - beta * neg_sum
        new_beliefs[node] = float(np.clip(new_belief, 0.0, 1.0))

    return new_beliefs


# ─────────────────────────────────────────────────────────────────
# 3.  INFORMATION PROPAGATION FRAMEWORK  (Section 3.1 / Algorithm 1)
# ─────────────────────────────────────────────────────────────────

def run_propagation(G: nx.DiGraph,
                    initial_seeds: list,
                    max_rounds: int = 30,
                    recovery_prob_base: float = 0.1) -> dict:
    """
    Algorithm 1 – InPro (extended SIR/IC for signed networks).

    States:
        0 = susceptible
        1 = infectious (accepted advertisement)
       -1 = removed

    Returns dict with per-round infected counts, beliefs history, etc.
    """
    N = G.number_of_nodes()
    beliefs   = np.array([G.nodes[n]['belief'] for n in range(N)])
    attitudes = np.zeros(N, dtype=int)

    # Initialise seed nodes
    for s in initial_seeds:
        attitudes[s] = 1
        beliefs[s]   = 1.0

    history = {
        'infected_per_round': [],
        'total_infected':     [],
        'beliefs_history':    [beliefs.copy()],
        'seed_set_history':   [set(initial_seeds)],
    }

    for t in range(1, max_rounds + 1):
        seed_set = set(np.where(attitudes == 1)[0])

        # Susceptible nodes that could receive recommendations
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
            break   # termination condition

        if susceptible_active:
            beliefs = belief_update(G, beliefs, attitudes, seed_set, t)

            newly_infected = 0
            for v in susceptible_active:
                # Attitude drawn from Binomial(1, x_{v,t})
                new_att = int(np.random.binomial(1, beliefs[v]))
                if new_att == 1:
                    # Recovery step: revert to 0 with probability p = 1 - x_{v,t}
                    recovery_prob = 1.0 - beliefs[v]
                    if np.random.random() < recovery_prob:
                        attitudes[v] = 0   # recovered → stays susceptible
                    else:
                        attitudes[v] = 1   # infected!
                        newly_infected += 1

            # Mark removed nodes (attitude=0 and past rec_end)
            for v in G.nodes():
                if attitudes[v] == 0 and t > G.nodes[v]['rec_end']:
                    attitudes[v] = -1

        total_inf = int(np.sum(attitudes == 1))
        history['infected_per_round'].append(total_inf - len(initial_seeds))
        history['total_infected'].append(total_inf)
        history['beliefs_history'].append(beliefs.copy())
        history['seed_set_history'].append(set(np.where(attitudes == 1)[0]))

    history['final_infected'] = int(np.sum(attitudes == 1))
    history['rounds']         = t
    return history


# ─────────────────────────────────────────────────────────────────
# 4.  SIGNED-PAGERANK ALGORITHM  (Section 3.3 / Algorithm 3)
# ─────────────────────────────────────────────────────────────────

def build_SPR_matrix(G: nx.DiGraph, damping: float = 0.85) -> np.ndarray:
    """
    Build the Signed-PageRank adjacency matrix Y = d * (W_tilde ⊙ L)
    where:
        W_tilde = column-normalised weight matrix
        L       = sign label matrix
        ⊙       = Hadamard (element-wise) product
    """
    N  = G.number_of_nodes()
    W  = np.zeros((N, N))
    L  = np.zeros((N, N))

    for u, v, data in G.edges(data=True):
        W[u, v] = data['weight']
        L[u, v] = data['sign']

    # Column-normalise W so each column sums to 1
    col_sums = W.sum(axis=0)
    col_sums[col_sums == 0] = 1.0    # avoid division by zero
    W_tilde  = W / col_sums

    Y = damping * (W_tilde * L)      # Hadamard product (element-wise)
    return Y


def signed_pagerank(G: nx.DiGraph,
                    k: int,
                    damping: float = 0.85,
                    max_iter: int = 200) -> list:
    """
    Algorithm 3 – SPR: Signed-PageRank.

    Rank update (Eq. 9):
        SPR_{i,t+1} = Σ_{j ∈ H^out_i} (SPR_{i,t} - SPR_{j,t}) * y_{i,j}
                      + (1-d)/N

    Convergence: when sorted ranking order stops changing.

    Returns the top-k seed nodes.
    """
    N   = G.number_of_nodes()
    Y   = build_SPR_matrix(G, damping)

    # Initialise SPR with node beliefs (x_i,0)
    spr = np.array([G.nodes[n]['belief'] for n in range(N)], dtype=float)

    current_order = np.argsort(-spr)   # descending

    for iteration in range(max_iter):
        spr_new = np.zeros(N)
        for i in range(N):
            out_neighbors = list(G.successors(i))
            total = 0.0
            for j in out_neighbors:
                total += (spr[i] - spr[j]) * Y[i, j]
            spr_new[i] = total + (1.0 - damping) / N

        new_order = np.argsort(-spr_new)

        # Convergence check: ranking order unchanged
        if np.array_equal(new_order, current_order):
            break

        spr         = spr_new
        current_order = new_order

    seed_nodes = list(current_order[:k])
    return seed_nodes


def signed_pagerank_fast(G: nx.DiGraph,
                          k: int,
                          damping: float = 0.85,
                          max_iter: int = 200) -> list:
    """
    Vectorised (matrix-based) SPR — O(N) via matrix ops as noted in the paper.
    Equivalent to signed_pagerank() but uses numpy broadcasting.
    """
    N   = G.number_of_nodes()
    Y   = build_SPR_matrix(G, damping)

    spr = np.array([G.nodes[n]['belief'] for n in range(N)], dtype=float)
    current_order = np.argsort(-spr)

    for _ in range(max_iter):
        # Outer difference: SPR_i - SPR_j for all (i,j)
        diff    = spr[:, None] - spr[None, :]  # (N,N)
        spr_new = (diff * Y).sum(axis=1) + (1.0 - damping) / N
        new_order = np.argsort(-spr_new)

        if np.array_equal(new_order, current_order):
            break

        spr           = spr_new
        current_order = new_order

    return list(current_order[:k])


# ─────────────────────────────────────────────────────────────────
# 5.  BENCHMARK ALGORITHMS  (Section 4)
# ─────────────────────────────────────────────────────────────────

def benchmark_P_plus(G: nx.DiGraph, k: int) -> list:
    """P+: top-k by weighted positive out-degree."""
    scores = {}
    for node in G.nodes():
        scores[node] = sum(
            d['weight'] for _, _, d in G.out_edges(node, data=True)
            if d['sign'] == 1
        )
    return sorted(scores, key=scores.get, reverse=True)[:k]


def benchmark_P_plusminus(G: nx.DiGraph, k: int) -> list:
    """P±: top-k by total weighted out-degree (unsigned)."""
    scores = {}
    for node in G.nodes():
        scores[node] = sum(d['weight'] for _, _, d in G.out_edges(node, data=True))
    return sorted(scores, key=scores.get, reverse=True)[:k]


def benchmark_SRWR(G: nx.DiGraph,
                   k: int,
                   restart_prob: float = 0.15,
                   max_iter: int = 100) -> list:
    """
    SRWR: Signed Random Walk with Restart (Jung et al. 2016).
    The random walker changes sign on negative edges and restarts with
    probability `restart_prob`.
    We use the personalised score of each node across all starts
    as a proxy for global influence.
    """
    N   = G.number_of_nodes()
    agg = np.zeros(N)

    # Run from a sample of start nodes for efficiency
    sample_starts = random.sample(list(G.nodes()), min(50, N))

    for start in sample_starts:
        r = np.zeros(N)
        r[start] = 1.0

        for _ in range(max_iter):
            r_new = np.zeros(N)
            for node in G.nodes():
                out_neighbors = list(G.successors(node))
                if not out_neighbors:
                    continue
                for nb in out_neighbors:
                    sign = G[node][nb]['sign']
                    w    = G[node][nb]['weight']
                    # Sign flip on negative edges
                    contribution = w * r[node] * sign
                    r_new[nb] += contribution

            # Normalise and apply restart
            norm = np.abs(r_new).sum()
            if norm > 0:
                r_new /= norm
            r_new = (1 - restart_prob) * r_new
            r_new[start] += restart_prob

            if np.allclose(r_new, r, atol=1e-6):
                break
            r = r_new

        agg += np.abs(r)

    return list(np.argsort(-agg)[:k])


def benchmark_SVIM_L(G: nx.DiGraph, k: int, alpha: float = 0.5) -> list:
    """
    SVIM-L: Signed Voter Influence Maximisation – Long-term.
    Uses the stationary distribution of the signed transition matrix
    (Li et al. 2013 / 2015).
    """
    N  = G.number_of_nodes()
    T  = np.zeros((N, N))

    for u, v, data in G.edges(data=True):
        out_deg = G.out_degree(u)
        if out_deg > 0:
            T[v, u] += data['sign'] * data['weight'] / out_deg

    # Long-term: eigenvector of T corresponding to largest eigenvalue
    eigenvalues, eigenvectors = np.linalg.eig(T)
    idx  = np.argmax(np.real(eigenvalues))
    stat = np.abs(np.real(eigenvectors[:, idx]))
    stat /= stat.sum() if stat.sum() > 0 else 1.0

    return list(np.argsort(-stat)[:k])


def benchmark_SVIM_S(G: nx.DiGraph, k: int, steps: int = 3) -> list:
    """
    SVIM-S: Signed Voter Influence Maximisation – Short-term.
    Uses T^steps * uniform as short-term influence estimate.
    """
    N  = G.number_of_nodes()
    T  = np.zeros((N, N))

    for u, v, data in G.edges(data=True):
        out_deg = G.out_degree(u)
        if out_deg > 0:
            T[v, u] += data['sign'] * data['weight'] / out_deg

    # Short-term: T^steps applied to uniform distribution
    dist = np.ones(N) / N
    T_power = np.linalg.matrix_power(T, steps)
    short_term = np.abs(T_power @ dist)
    short_term /= short_term.sum() if short_term.sum() > 0 else 1.0

    return list(np.argsort(-short_term)[:k])


# ─────────────────────────────────────────────────────────────────
# 6.  EXPERIMENT RUNNER
# ─────────────────────────────────────────────────────────────────

def run_experiment(G: nx.DiGraph,
                   k_values: list,
                   n_runs: int = 100,
                   max_rounds: int = 25) -> pd.DataFrame:
    """
    Run all algorithms for each k in k_values, averaged over n_runs.
    Returns a DataFrame: columns = [k, algorithm, avg_infected, avg_time, avg_rounds]
    """
    algorithms = {
        'SPR'    : lambda g, k: signed_pagerank_fast(g, k),
        'P+'     : benchmark_P_plus,
        'P±'     : benchmark_P_plusminus,
        'SRWR'   : benchmark_SRWR,
        'SVIM-L' : benchmark_SVIM_L,
        'SVIM-S' : benchmark_SVIM_S,
    }

    records = []
    total = len(k_values) * len(algorithms)
    done  = 0

    for k in k_values:
        for alg_name, alg_fn in algorithms.items():
            done += 1
            print(f"  [{done}/{total}] k={k:3d}  {alg_name:<8}", end='  ', flush=True)

            t0    = time.time()
            seeds = alg_fn(G, k)
            seed_time = time.time() - t0

            infected_counts = []
            round_counts    = []

            for run in range(n_runs):
                # Fresh copy of beliefs per run (slight noise for stochasticity)
                G_run = deepcopy(G)
                for node in G_run.nodes():
                    base = G_run.nodes[node]['belief']
                    G_run.nodes[node]['belief'] = float(
                        np.clip(base + np.random.normal(0, 0.02), 0, 1)
                    )

                hist = run_propagation(G_run, seeds, max_rounds=max_rounds)
                infected_counts.append(hist['final_infected'])
                round_counts.append(hist['rounds'])

            avg_infected = np.mean(infected_counts)
            avg_rounds   = np.mean(round_counts)

            records.append({
                'k'           : k,
                'algorithm'   : alg_name,
                'avg_infected': avg_infected,
                'avg_rounds'  : avg_rounds,
                'seed_time_ms': seed_time * 1000,
            })
            print(f"infected={avg_infected:.1f}  rounds={avg_rounds:.1f}  "
                  f"time={seed_time*1000:.1f}ms")

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────
# 7.  CONVERGENCE ANALYSIS
# ─────────────────────────────────────────────────────────────────

def convergence_analysis(G: nx.DiGraph,
                         k_values: list,
                         n_runs: int = 30) -> pd.DataFrame:
    """Measure how many SPR iterations are needed to converge for each k."""
    records = []

    for k in k_values:
        iters_list = []
        for _ in range(n_runs):
            N = G.number_of_nodes()
            Y = build_SPR_matrix(G)
            spr = np.array([G.nodes[n]['belief'] for n in range(N)], dtype=float)
            cur = np.argsort(-spr)

            for it in range(1, 201):
                diff    = spr[:, None] - spr[None, :]
                spr_new = (diff * Y).sum(axis=1) + 0.15 / N
                nxt     = np.argsort(-spr_new)
                if np.array_equal(nxt, cur):
                    iters_list.append(it)
                    break
                spr, cur = spr_new, nxt
            else:
                iters_list.append(200)

        records.append({'k': k, 'avg_iterations': np.mean(iters_list),
                        'std_iterations': np.std(iters_list)})
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────
# 8.  VISUALISATION
# ─────────────────────────────────────────────────────────────────

COLORS = {
    'SPR'   : '#E63946',
    'P+'    : '#2196F3',
    'P±'    : '#4CAF50',
    'SRWR'  : '#FF9800',
    'SVIM-L': '#9C27B0',
    'SVIM-S': '#795548',
}
MARKERS = {
    'SPR'   : 'o',
    'P+'    : 's',
    'P±'    : '^',
    'SRWR'  : 'D',
    'SVIM-L': 'v',
    'SVIM-S': 'P',
}


def plot_results(df: pd.DataFrame,
                 conv_df: pd.DataFrame,
                 G: nx.DiGraph,
                 output: str = '../results/spr_results.png'):

    fig = plt.figure(figsize=(20, 22))
    fig.patch.set_facecolor('#0F1117')

    gs = gridspec.GridSpec(4, 3, figure=fig,
                           hspace=0.45, wspace=0.35,
                           top=0.93, bottom=0.05,
                           left=0.07, right=0.97)

    TITLE_STYLE = dict(color='white', fontsize=11, fontweight='bold', pad=10)
    LABEL_STYLE = dict(color='#CCCCCC', fontsize=9)
    TICK_STYLE  = dict(colors='#AAAAAA', labelsize=8)
    GRID_STYLE  = dict(color='#2A2A3A', linewidth=0.5, alpha=0.7)

    def style_ax(ax, title, xlabel, ylabel):
        ax.set_facecolor('#1A1A2E')
        ax.set_title(title, **TITLE_STYLE)
        ax.set_xlabel(xlabel, **LABEL_STYLE)
        ax.set_ylabel(ylabel, **LABEL_STYLE)
        ax.tick_params(**TICK_STYLE)
        ax.grid(True, **GRID_STYLE)
        for spine in ax.spines.values():
            spine.set_edgecolor('#333355')

    # ── (a) Infected individuals vs k ────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    for alg in df['algorithm'].unique():
        sub = df[df['algorithm'] == alg].sort_values('k')
        ax1.plot(sub['k'], sub['avg_infected'],
                 color=COLORS[alg], marker=MARKERS[alg],
                 linewidth=2, markersize=6, label=alg)
    style_ax(ax1,
             'Effectiveness: Avg Infected Individuals vs Number of Seed Nodes (k)',
             'Number of Seed Nodes (k)', 'Avg Infected Individuals')
    ax1.legend(facecolor='#1A1A2E', edgecolor='#444466',
               labelcolor='white', fontsize=9, loc='upper left',
               ncol=3)

    # ── (b) Relative improvement of SPR over best benchmark ──────
    ax2 = fig.add_subplot(gs[0, 2])
    k_vals = sorted(df['k'].unique())
    improvements = []
    for k in k_vals:
        spr_val  = df[(df['algorithm'] == 'SPR') & (df['k'] == k)]['avg_infected'].values
        bench    = df[(df['algorithm'] != 'SPR') & (df['k'] == k)]['avg_infected']
        if len(spr_val) and len(bench):
            best = bench.max()
            if best > 0:
                improvements.append((k, (spr_val[0] - best) / best * 100))
    if improvements:
        ks, imps = zip(*improvements)
        bars = ax2.bar(ks, imps, color='#E63946', alpha=0.85, width=0.8)
        ax2.axhline(0, color='white', linewidth=0.8, linestyle='--')
        for bar, imp in zip(bars, imps):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                     f'{imp:.1f}%', ha='center', va='bottom',
                     color='#FF9999', fontsize=7)
    style_ax(ax2, 'SPR Improvement\nover Best Benchmark',
             'k', 'Improvement (%)')

    # ── (c) Convergence iterations ───────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.errorbar(conv_df['k'], conv_df['avg_iterations'],
                 yerr=conv_df['std_iterations'],
                 color='#E63946', marker='o', linewidth=2,
                 capsize=4, ecolor='#FF9999', markersize=6)
    style_ax(ax3, 'SPR Convergence\nIterations vs k',
             'k', 'Avg Iterations to Convergence')

    # ── (d) Propagation time vs k ────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    for alg in df['algorithm'].unique():
        sub = df[df['algorithm'] == alg].sort_values('k')
        ax4.plot(sub['k'], sub['seed_time_ms'],
                 color=COLORS[alg], marker=MARKERS[alg],
                 linewidth=2, markersize=5, label=alg)
    style_ax(ax4, 'Seed Selection Time vs k',
             'k', 'Time (ms)')
    ax4.legend(facecolor='#1A1A2E', edgecolor='#444466',
               labelcolor='white', fontsize=8, ncol=2)

    # ── (e) Propagation rounds vs k ──────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    for alg in df['algorithm'].unique():
        sub = df[df['algorithm'] == alg].sort_values('k')
        ax5.plot(sub['k'], sub['avg_rounds'],
                 color=COLORS[alg], marker=MARKERS[alg],
                 linewidth=2, markersize=5, label=alg)
    style_ax(ax5, 'Propagation Rounds vs k',
             'k', 'Avg Rounds')
    ax5.legend(facecolor='#1A1A2E', edgecolor='#444466',
               labelcolor='white', fontsize=8, ncol=2)

    # ── (f) Network degree distribution ─────────────────────────
    ax6 = fig.add_subplot(gs[2, 0])
    degrees = [d for _, d in G.out_degree()]
    ax6.hist(degrees, bins=40, color='#2196F3', alpha=0.8, edgecolor='none')
    style_ax(ax6, 'Out-Degree Distribution\n(Scale-Free Network)',
             'Out-Degree', 'Count')

    # ── (g) Belief distribution ───────────────────────────────────
    ax7 = fig.add_subplot(gs[2, 1])
    beliefs = [G.nodes[n]['belief'] for n in G.nodes()]
    ax7.hist(beliefs, bins=30, color='#4CAF50', alpha=0.8, edgecolor='none')
    style_ax(ax7, 'Initial Belief Distribution\n(Beta(2,3))',
             'Belief x_{i,0}', 'Count')

    # ── (h) Edge sign distribution ───────────────────────────────
    ax8 = fig.add_subplot(gs[2, 2])
    signs  = [d['sign'] for _, _, d in G.edges(data=True)]
    n_pos  = signs.count(1)
    n_neg  = signs.count(-1)
    bars   = ax8.bar(['Positive (+1)', 'Negative (−1)'], [n_pos, n_neg],
                     color=['#4CAF50', '#E63946'], alpha=0.85)
    for bar, val in zip(bars, [n_pos, n_neg]):
        ax8.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + max(n_pos, n_neg)*0.01,
                 f'{val:,}\n({val/(n_pos+n_neg)*100:.1f}%)',
                 ha='center', color='white', fontsize=8)
    style_ax(ax8, 'Edge Sign Distribution', 'Sign', 'Count')

    # ── (i) Per-algorithm performance heatmap ─────────────────────
    ax9 = fig.add_subplot(gs[3, :])
    pivot = df.pivot(index='algorithm', columns='k', values='avg_infected')
    im = ax9.imshow(pivot.values, aspect='auto', cmap='YlOrRd',
                    interpolation='nearest')
    ax9.set_xticks(range(len(pivot.columns)))
    ax9.set_xticklabels(pivot.columns, **dict(color='#CCCCCC', fontsize=8))
    ax9.set_yticks(range(len(pivot.index)))
    ax9.set_yticklabels(pivot.index, **dict(color='#CCCCCC', fontsize=9))
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            ax9.text(j, i, f'{val:.0f}', ha='center', va='center',
                     color='black' if val > pivot.values.max()*0.6 else 'white',
                     fontsize=7, fontweight='bold')
    plt.colorbar(im, ax=ax9, fraction=0.02, pad=0.01,
                 label='Avg Infected').ax.yaxis.label.set_color('white')
    ax9.set_title('Performance Heatmap: Avg Infected Individuals',
                  **TITLE_STYLE)
    ax9.set_xlabel('Number of Seed Nodes (k)', **LABEL_STYLE)
    ax9.set_ylabel('Algorithm', **LABEL_STYLE)
    ax9.set_facecolor('#1A1A2E')

    # ── Main title ────────────────────────────────────────────────
    fig.suptitle(
        'Signed-PageRank: Influence Maximization in Signed Social Networks\n'
        'Implementation of Yin et al., IEEE TKDE 2021',
        color='white', fontsize=14, fontweight='bold', y=0.975
    )

    plt.savefig(output, dpi=150, bbox_inches='tight',
                facecolor='#0F1117', edgecolor='none')
    print(f"\n[Plot] Saved → {output}")
    plt.close()


# ─────────────────────────────────────────────────────────────────
# 9.  BELIEF DYNAMICS TRACE PLOT
# ─────────────────────────────────────────────────────────────────

def plot_belief_trace(G: nx.DiGraph,
                      k: int = 3,
                      output: str = '../results/spr_belief_trace.png'):
    """Visualise belief evolution over time for SPR-selected seeds."""
    seeds = signed_pagerank_fast(G, k)
    hist  = run_propagation(G, seeds, max_rounds=15)

    # Pick a sample of nodes: seeds + 10 random others
    sample_nodes = seeds[:3] + random.sample(
        [n for n in G.nodes() if n not in seeds], min(8, G.number_of_nodes()-k)
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#0F1117')

    cmap = plt.cm.plasma
    rounds = len(hist['beliefs_history'])

    # Left: belief trajectories
    ax = axes[0]
    ax.set_facecolor('#1A1A2E')
    for idx, node in enumerate(sample_nodes):
        beliefs_over_time = [hist['beliefs_history'][t][node]
                             for t in range(rounds)]
        style = '-' if node in seeds else '--'
        lw    = 2.5 if node in seeds else 1.0
        label = f'Node {node} (seed)' if node in seeds else f'Node {node}'
        ax.plot(range(rounds), beliefs_over_time,
                color=cmap(idx / len(sample_nodes)),
                linestyle=style, linewidth=lw, label=label, alpha=0.85)
    ax.set_title('Belief Dynamics over Time', color='white',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Time Step', color='#CCCCCC', fontsize=9)
    ax.set_ylabel('Belief x_{i,t}', color='#CCCCCC', fontsize=9)
    ax.tick_params(colors='#AAAAAA', labelsize=8)
    ax.grid(True, color='#2A2A3A', linewidth=0.5)
    for sp in ax.spines.values():
        sp.set_edgecolor('#333355')
    ax.legend(facecolor='#1A1A2E', edgecolor='#444466',
              labelcolor='white', fontsize=7, ncol=2)

    # Right: infected count over time
    ax2 = axes[1]
    ax2.set_facecolor('#1A1A2E')
    ax2.fill_between(range(len(hist['total_infected'])),
                     hist['total_infected'],
                     color='#E63946', alpha=0.35)
    ax2.plot(range(len(hist['total_infected'])),
             hist['total_infected'],
             color='#E63946', linewidth=2.5, marker='o', markersize=5)
    ax2.set_title('Cumulative Infected Nodes over Time (SPR seeds)',
                  color='white', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Time Step', color='#CCCCCC', fontsize=9)
    ax2.set_ylabel('Total Infected', color='#CCCCCC', fontsize=9)
    ax2.tick_params(colors='#AAAAAA', labelsize=8)
    ax2.grid(True, color='#2A2A3A', linewidth=0.5)
    for sp in ax2.spines.values():
        sp.set_edgecolor('#333355')

    fig.suptitle('Belief Dynamics & Propagation Trace',
                 color='white', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight',
                facecolor='#0F1117', edgecolor='none')
    print(f"[Plot] Saved → {output}")
    plt.close()


# ─────────────────────────────────────────────────────────────────
# 10.  TOY EXAMPLE (exactly as in the paper, Section 3.3.2)
# ─────────────────────────────────────────────────────────────────

def run_toy_example():
    """Reproduce the paper's toy example (Fig 4b) step by step."""
    print("\n" + "="*60)
    print("TOY EXAMPLE (Paper Section 3.3.2 – Fig. 4b)")
    print("="*60)

    # 5-node signed weighted directed graph from Figure 4b
    G = nx.DiGraph()
    nodes = ['A','B','C','D','E']
    node_idx = {n: i for i, n in enumerate(nodes)}
    for n in nodes:
        G.add_node(node_idx[n])

    # Edges: (from, to, weight, sign)
    edges = [
        ('A','B', 0.2,  1), ('A','C', 0.7,  1), ('A','D', 0.4, -1),
        ('B','A', 0.3, -1), ('B','D', 0.5,  1),
        ('C','A', 0.6, -1), ('C','E', 0.6,  1),
        ('D','B', 0.5,  1), ('D','C', 0.7,  1), ('D','E', 0.4, -1),
        ('E','D', 0.2,  1),
    ]
    for u, v, w, s in edges:
        G.add_edge(node_idx[u], node_idx[v], weight=w, sign=s)

    # Initial beliefs and recommendation cycles from paper
    x0         = [0.5, 0.7, 0.3, 0.8, 0.6]
    rec_cycles = [(1,3),(1,2),(1,6),(4,8),(4,5)]

    for i, n in enumerate(nodes):
        G.nodes[i]['belief']    = x0[i]
        G.nodes[i]['rec_start'] = rec_cycles[i][0]
        G.nodes[i]['rec_end']   = rec_cycles[i][1]

    # Run SPR
    seeds = signed_pagerank_fast(G, k=1)
    print(f"\nSPR selected seed node: {nodes[seeds[0]]} (index {seeds[0]})")
    print("Expected: D (index 3) per the paper\n")

    # Run propagation
    hist = run_propagation(G, seeds, max_rounds=7)
    for t, (total, beliefs) in enumerate(
            zip(hist['total_infected'], hist['beliefs_history'][1:])):
        b_str = ', '.join(f"{nodes[i]}={v:.3f}" for i, v in enumerate(beliefs))
        print(f"  t={t+1}: beliefs=[{b_str}]  infected={total}")

    accepted = [nodes[i] for i in range(5)
                if hist['beliefs_history'][-1][i] > 0.7]
    print(f"\nNodes with belief > 0.7 at end: {accepted}")
    print("Paper reports: B, C, D accepted (belief ≥ 0.7)\n")


# ─────────────────────────────────────────────────────────────────
# 11.  MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  Signed-PageRank Implementation  |  Yin et al. IEEE TKDE 2021")
    print("=" * 65)

    # ── Toy example first ────────────────────────────────────────
    run_toy_example()

    # ── Build Epinions-like dataset ──────────────────────────────
    print("\n[1/4] Building signed social network (Epinions-like)...")
    G = build_signed_network(n_nodes=400, neg_fraction=0.147, seed=42)

    # ── Belief trace plot ────────────────────────────────────────
    print("\n[2/4] Generating belief dynamics trace...")
    plot_belief_trace(G, k=5)

    # ── Main experiment ──────────────────────────────────────────
    print("\n[3/4] Running experiments (this takes ~2-3 min)...")
    k_values  = [2, 5, 10, 15, 20, 30, 40, 50]
    n_runs    = 50    # 50 Monte-Carlo runs per (algorithm, k) pair

    df = run_experiment(G, k_values, n_runs=n_runs, max_rounds=20)

    # ── Convergence analysis ─────────────────────────────────────
    print("\n[4/4] Convergence analysis...")
    conv_df = convergence_analysis(G, k_values, n_runs=20)
    print(conv_df.to_string(index=False))

    # ── Save results ─────────────────────────────────────────────
    df.to_csv('../results/spr_results.csv', index=False)
    conv_df.to_csv('../results/spr_convergence.csv', index=False)

    print("\n── Summary Table ──────────────────────────────────────────")
    summary = df.groupby('algorithm').agg(
        avg_infected=('avg_infected','mean'),
        avg_time_ms=('seed_time_ms','mean')
    ).reset_index().sort_values('avg_infected', ascending=False)
    print(summary.to_string(index=False))

    # ── SPR improvement ──────────────────────────────────────────
    spr_avg = summary[summary['algorithm']=='SPR']['avg_infected'].values[0]
    best_bench = summary[summary['algorithm']!='SPR']['avg_infected'].max()
    print(f"\nSPR improvement over best benchmark: "
          f"{(spr_avg-best_bench)/best_bench*100:.1f}%")
    print(f"(Paper reports: ~8.4% on Epinions, ~20% on synthetic)")

    # ── Plots ────────────────────────────────────────────────────
    plot_results(df, conv_df, G)

    print("\n✓ All outputs saved to ../results/")
    print("  ├── spr_results.png       – main results figure")
    print("  ├── spr_belief_trace.png  – belief dynamics")
    print("  ├── spr_results.csv       – raw experiment data")
    print("  └── spr_convergence.csv   – convergence data")


if __name__ == '__main__':
    main()

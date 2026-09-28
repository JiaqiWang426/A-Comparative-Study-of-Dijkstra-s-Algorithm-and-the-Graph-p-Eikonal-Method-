import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import graphlearning as gl
import numpy as np
import sklearn.datasets as datasets


@dataclass
class DatasetBundle:
    name: str
    X: np.ndarray
    source_idx: np.ndarray
    target_idx: np.ndarray
    metadata: Dict


# =========================
# Synthetic datasets used in thesis
# =========================


def make_two_moons(n: int = 1500, noise: float = 0.08, seed: int = 0) -> DatasetBundle:
    X, labels = datasets.make_moons(n_samples=n, noise=noise, random_state=seed)

    moon0 = np.where(labels == 0)[0]
    moon1 = np.where(labels == 1)[0]

    source_idx = moon0[np.argsort(X[moon0, 0])[: max(3, len(moon0) // 40)]]
    target_idx = moon1[np.argsort(-X[moon1, 0])[: max(3, len(moon1) // 40)]]

    return DatasetBundle(
        name="two_moons",
        X=X,
        source_idx=source_idx,
        target_idx=target_idx,
        metadata={"seed": seed, "n": n, "noise": noise, "labels": labels},
    )



def make_gaussian_mixture(n: int = 1500, seed: int = 0) -> DatasetBundle:
    rng = np.random.default_rng(seed)

    n1 = n // 2
    n2 = n - n1

    X1 = 0.50 * rng.standard_normal((n1, 2)) + np.array([-1.5, 0.0])
    X2 = 0.80 * rng.standard_normal((n2, 2)) + np.array([2.0, 0.5])
    X = np.vstack((X1, X2))

    source_idx = np.arange(n1)[np.argsort(X1[:, 0])[: max(5, n1 // 40)]]
    target_idx = (n1 + np.arange(n2))[np.argsort(-X2[:, 0])[: max(5, n2 // 40)]]

    return DatasetBundle(
        name="gaussian_mixture",
        X=X,
        source_idx=source_idx,
        target_idx=target_idx,
        metadata={"seed": seed, "n": n},
    )



def make_obstacle_grid(
    m: int = 45,
    obstacle_xmin: float = 0.40,
    obstacle_xmax: float = 0.60,
    obstacle_ymin: float = 0.03,
    obstacle_ymax: float = 0.53,
) -> DatasetBundle:
    """
    Grid with a clearly downward-shifted rectangular obstacle.
    """
    xs = np.linspace(0.0, 1.0, m)
    ys = np.linspace(0.0, 1.0, m)
    XX, YY = np.meshgrid(xs, ys)
    X = np.column_stack([XX.ravel(), YY.ravel()])

    obstacle = (
        (X[:, 0] > obstacle_xmin)
        & (X[:, 0] < obstacle_xmax)
        & (X[:, 1] > obstacle_ymin)
        & (X[:, 1] < obstacle_ymax)
    )
    X = X[~obstacle]

    source_idx = np.where((X[:, 0] < 0.08) & (np.abs(X[:, 1] - 0.5) < 0.12))[0]
    target_idx = np.where((X[:, 0] > 0.92) & (np.abs(X[:, 1] - 0.5) < 0.12))[0]

    if len(source_idx) == 0:
        source_idx = np.array([np.argmin(X[:, 0])])
    if len(target_idx) == 0:
        target_idx = np.array([np.argmax(X[:, 0])])

    return DatasetBundle(
        name="obstacle_grid",
        X=X,
        source_idx=source_idx,
        target_idx=target_idx,
        metadata={
            "m": m,
            "obstacle_xmin": obstacle_xmin,
            "obstacle_xmax": obstacle_xmax,
            "obstacle_ymin": obstacle_ymin,
            "obstacle_ymax": obstacle_ymax,
        },
    )



def make_dataset(name: str, seed: int = 0) -> DatasetBundle:
    if name == "two_moons":
        return make_two_moons(seed=seed)
    if name == "gaussian_mixture":
        return make_gaussian_mixture(seed=seed)
    if name == "obstacle_grid":
        return make_obstacle_grid()
    raise ValueError(
        f"Unknown dataset: {name}. Supported datasets: two_moons, gaussian_mixture, obstacle_grid"
    )


# =========================
# Graph construction
# =========================


def build_graph_knn(
    X: np.ndarray,
    k: int = 10,
    knn_search_k: int = 30,
    kernel: str = "uniform",
):
    knn_ind, knn_dist = gl.weightmatrix.knnsearch(X, knn_search_k)
    W = gl.weightmatrix.knn(X, k, knn_data=(knn_ind, knn_dist), kernel=kernel)
    G = gl.graph(W)
    return W, G, knn_ind, knn_dist



def ensure_connected_graph(
    X: np.ndarray,
    k_list: List[int] = [8, 10, 12, 15, 20],
    knn_search_k: int = 40,
    kernel: str = "uniform",
):
    for k in k_list:
        W, G, knn_ind, knn_dist = build_graph_knn(
            X, k=k, knn_search_k=knn_search_k, kernel=kernel
        )
        if G.isconnected():
            return W, G, knn_ind, knn_dist, k
    raise RuntimeError("Failed to build a connected graph with tried k values.")


# =========================
# Cost / density functions
# =========================


def kde_from_knn_dist(knn_dist: np.ndarray) -> np.ndarray:
    d = np.max(knn_dist, axis=1)
    d = np.maximum(d, 1e-12)
    return (d / d.max()) ** (-1)



def density_weight_from_alpha(kde: np.ndarray, alpha: float = 0.0) -> np.ndarray:
    return kde ** (-alpha)



def obstacle_cost_field(X: np.ndarray) -> np.ndarray:
    y = X[:, 1]
    corridor = np.exp(-((y - 0.5) ** 2) / 0.01)
    f = 1.5 - 0.7 * corridor
    return np.maximum(f, 0.3)


# =========================
# Noise models
# =========================


def multiply_edge_weights_noise(W, sigma: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    Wn = W.copy().tocsr()
    data = Wn.data.copy()
    noise = 1.0 + sigma * rng.standard_normal(size=data.shape[0])
    noise = np.maximum(noise, 1e-3)
    Wn.data = data * noise
    return Wn


# =========================
# Solvers
# =========================


def run_peikonal(G, source_idx: np.ndarray, p: int, f: Optional[np.ndarray] = None):
    start = time.perf_counter()
    if f is None:
        u = G.peikonal(source_idx.tolist(), p=p)
    else:
        u = G.peikonal(source_idx.tolist(), p=p, f=f)
    runtime = time.perf_counter() - start
    return u, runtime



def run_dijkstra(G, source_idx: np.ndarray, f: Optional[np.ndarray] = None):
    start = time.perf_counter()
    if f is None:
        u = G.dijkstra(source_idx.tolist(), f=1)
    else:
        u = G.dijkstra(source_idx.tolist(), f=f)
    runtime = time.perf_counter() - start
    return u, runtime


# =========================
# Path extraction and path metrics
# =========================


def greedy_path_to_source(
    W,
    u: np.ndarray,
    start_idx: int,
    source_set: np.ndarray,
    max_steps: int = 2000,
    tol: float = 1e-12,
) -> List[int]:
    source_lookup = set(source_set.tolist())
    current = int(start_idx)
    path = [current]
    Wcsr = W.tocsr()

    for _ in range(max_steps):
        if current in source_lookup:
            break

        neigh = Wcsr[current, :].nonzero()[1]
        if len(neigh) == 0:
            break

        vals = u[neigh]
        min_val = np.min(vals)
        cand = neigh[vals <= min_val + tol]
        if len(cand) == 0:
            cand = neigh[np.argmin(vals): np.argmin(vals) + 1]
        next_idx = int(np.min(cand))

        if len(path) >= 2 and next_idx == path[-2]:
            break
        if next_idx == current:
            break
        if u[next_idx] > u[current] + tol:
            break

        path.append(next_idx)
        current = next_idx

    return path



def choose_representative_target(X: np.ndarray, target_idx: np.ndarray, source_idx: np.ndarray) -> int:
    src_center = X[source_idx].mean(axis=0)
    d = np.linalg.norm(X[target_idx] - src_center, axis=1)
    return int(target_idx[np.argmax(d)])



def path_length(X: np.ndarray, path: Sequence[int]) -> float:
    if len(path) <= 1:
        return 0.0
    P = X[np.asarray(path, dtype=int)]
    return float(np.sum(np.linalg.norm(P[1:] - P[:-1], axis=1)))



def path_cost(X: np.ndarray, path: Sequence[int], f: Optional[np.ndarray] = None) -> float:
    if len(path) <= 1:
        return 0.0
    idx = np.asarray(path, dtype=int)
    P = X[idx]
    seg_len = np.linalg.norm(P[1:] - P[:-1], axis=1)
    if f is None:
        return float(np.sum(seg_len))
    node_cost = np.asarray(f, dtype=float)[idx]
    seg_cost = 0.5 * (node_cost[1:] + node_cost[:-1]) * seg_len
    return float(np.sum(seg_cost))



def relative_cost_gap(cost_ref: float, cost: float) -> float:
    if abs(cost_ref) < 1e-12:
        return 0.0 if abs(cost) < 1e-12 else float("inf")
    return float((cost - cost_ref) / cost_ref)



def cost_fidelity_from_gap(cost_gap: float) -> float:
    return float(1.0 / (1.0 + abs(cost_gap)))



def field_fidelity_from_error(field_error: float) -> float:
    return float(1.0 / (1.0 + max(0.0, field_error)))



def robustness_from_stability(stability: float) -> float:
    return float(1.0 / (1.0 + max(0.0, stability)))



def discrete_frechet_distance_coords(A: np.ndarray, B: np.ndarray) -> float:
    if A.size == 0 or B.size == 0:
        return np.nan

    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    p, q = len(A), len(B)
    ca = np.full((p, q), np.nan, dtype=float)

    def c(i: int, j: int) -> float:
        if not np.isnan(ca[i, j]):
            return float(ca[i, j])

        dij = float(np.linalg.norm(A[i] - B[j]))
        if i == 0 and j == 0:
            ca[i, j] = dij
        elif i > 0 and j == 0:
            ca[i, j] = max(c(i - 1, 0), dij)
        elif i == 0 and j > 0:
            ca[i, j] = max(c(0, j - 1), dij)
        else:
            ca[i, j] = max(min(c(i - 1, j), c(i - 1, j - 1), c(i, j - 1)), dij)
        return float(ca[i, j])

    return c(p - 1, q - 1)



def discrete_frechet_distance(X: np.ndarray, path_a: Sequence[int], path_b: Sequence[int]) -> float:
    if len(path_a) == 0 or len(path_b) == 0:
        return np.nan
    A = X[np.asarray(path_a, dtype=int)]
    B = X[np.asarray(path_b, dtype=int)]
    return discrete_frechet_distance_coords(A, B)



def normalized_discrete_frechet_distance(
    X: np.ndarray,
    reference_path: Sequence[int],
    compared_path: Sequence[int],
) -> float:
    ref_len = path_length(X, reference_path)
    if ref_len < 1e-12:
        return np.nan
    d_f = discrete_frechet_distance(X, reference_path, compared_path)
    return float(d_f / ref_len)



def representative_path_index_from_distances(distances: Sequence[float]) -> int:
    arr = np.asarray(distances, dtype=float)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return 0
    med = np.median(arr[finite])
    idx = int(np.nanargmin(np.abs(arr - med)))
    return idx


# =========================
# Field metrics
# =========================


def relative_l2_error(u_ref: np.ndarray, u: np.ndarray) -> float:
    denom = np.linalg.norm(u_ref)
    if denom < 1e-12:
        return float(np.linalg.norm(u - u_ref))
    return float(np.linalg.norm(u - u_ref) / denom)


# =========================
# Figure helpers
# =========================


def plot_paths_on_dataset(
    X: np.ndarray,
    dataset_name: str,
    source_idx: np.ndarray,
    target_idx: np.ndarray,
    path_dict: Dict[str, Sequence[int]],
    title: str,
    outfile: str,
    figsize: Tuple[float, float] = (7, 5),
):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(X[:, 0], X[:, 1], s=8, c="lightgray", alpha=0.8)
    ax.scatter(X[source_idx, 0], X[source_idx, 1], c="red", s=24, marker="o", label="source")
    ax.scatter(X[target_idx, 0], X[target_idx, 1], c="black", s=30, marker="s", label="target")

    for label, path in path_dict.items():
        if path is None or len(path) < 2:
            continue
        P = X[np.asarray(path, dtype=int)]
        lw = 2.8 if "Dijkstra" in label else 1.8
        zorder = 4 if "Dijkstra" in label else 3
        ax.plot(P[:, 0], P[:, 1], linewidth=lw, label=label, zorder=zorder)

    ax.set_title(title)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(outfile, dpi=220)
    plt.close(fig)



def plot_path_fan_on_dataset(
    X: np.ndarray,
    source_idx: np.ndarray,
    target_idx: np.ndarray,
    clean_path: Sequence[int],
    noisy_paths: Sequence[Sequence[int]],
    representative_path: Optional[Sequence[int]],
    title: str,
    outfile: str,
    figsize: Tuple[float, float] = (7, 5),
):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(X[:, 0], X[:, 1], s=8, c="lightgray", alpha=0.8)
    ax.scatter(X[source_idx, 0], X[source_idx, 1], c="red", s=24, marker="o", label="source")
    ax.scatter(X[target_idx, 0], X[target_idx, 1], c="black", s=30, marker="s", label="target")

    for i, path in enumerate(noisy_paths):
        if path is None or len(path) < 2:
            continue
        Pn = X[np.asarray(path, dtype=int)]
        ax.plot(
            Pn[:, 0],
            Pn[:, 1],
            color="tab:blue",
            alpha=0.18,
            linewidth=1.0,
            label="noisy repeats" if i == 0 else None,
            zorder=2,
        )

    if clean_path is not None and len(clean_path) >= 2:
        Pc = X[np.asarray(clean_path, dtype=int)]
        ax.plot(Pc[:, 0], Pc[:, 1], color="black", linewidth=2.8, label="clean reference", zorder=4)

    if representative_path is not None and len(representative_path) >= 2:
        Pr = X[np.asarray(representative_path, dtype=int)]
        ax.plot(Pr[:, 0], Pr[:, 1], color="tab:red", linewidth=2.2, label="median representative", zorder=5)

    ax.set_title(title)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(outfile, dpi=220)
    plt.close(fig)

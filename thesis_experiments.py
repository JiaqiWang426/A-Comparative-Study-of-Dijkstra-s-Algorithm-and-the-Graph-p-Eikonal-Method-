import argparse
import os
from typing import Dict, List, Optional

import graphlearning as gl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from thesis_utils import (
    choose_representative_target,
    cost_fidelity_from_gap,
    density_weight_from_alpha,
    ensure_connected_graph,
    field_fidelity_from_error,
    greedy_path_to_source,
    kde_from_knn_dist,
    make_dataset,
    multiply_edge_weights_noise,
    normalized_discrete_frechet_distance,
    obstacle_cost_field,
    path_cost,
    path_length,
    plot_path_fan_on_dataset,
    plot_paths_on_dataset,
    relative_cost_gap,
    relative_l2_error,
    representative_path_index_from_distances,
    robustness_from_stability,
    run_dijkstra,
    run_peikonal,
)

DEFAULT_DATASETS = ["gaussian_mixture", "obstacle_grid", "two_moons"]
DEFAULT_P_LIST = [1, 2, 4, 8, 16, 32, 64, 128]
DEFAULT_SIGMA_LIST = [0.0, 0.05, 0.10, 0.20]


def get_cost_field(dataset_name: str, X: np.ndarray, knn_dist: np.ndarray, alpha: float) -> np.ndarray:
    if dataset_name == "obstacle_grid":
        return obstacle_cost_field(X)
    kde = kde_from_knn_dist(knn_dist)
    return density_weight_from_alpha(kde, alpha=alpha)



def clean_experiment_one_dataset(
    dataset_name: str,
    p_list: List[int],
    alpha: float = 0.0,
    make_figures: bool = False,
    fig_dir: str = "figures",
    seed: int = 0,
):
    bundle = make_dataset(dataset_name, seed=seed)
    X = bundle.X
    source_idx = bundle.source_idx
    target_idx = bundle.target_idx

    W, G, _, knn_dist, used_k = ensure_connected_graph(X)
    f = get_cost_field(dataset_name, X, knn_dist, alpha)
    target0 = choose_representative_target(X, target_idx, source_idx)
    clean_ref_target = np.array([target0])

    rows = []
    path_dict = {}

    u_dij, _ = run_dijkstra(G, source_idx, f=f)
    path_dij = greedy_path_to_source(W, u_dij, start_idx=target0, source_set=source_idx)
    len_dij = path_length(X, path_dij)
    cost_dij = path_cost(X, path_dij, f=f)
    path_dict["Dijkstra"] = path_dij

    rows.append(
        {
            "dataset": dataset_name,
            "method": "dijkstra",
            "p": np.nan,
            "graph_k": used_k,
            "path_length": len_dij,
            "path_cost_clean": cost_dij,
            "cost_gap_vs_dijkstra": 0.0,
            "cost_fidelity": 1.0,
            "field_vs_dijkstra": 0.0,
            "field_fidelity": 1.0,
            "target_idx": target0,
        }
    )

    for p in p_list:
        u_p, _ = run_peikonal(G, source_idx, p=p, f=f)
        path_p = greedy_path_to_source(W, u_p, start_idx=target0, source_set=source_idx)
        len_p = path_length(X, path_p)
        cost_p = path_cost(X, path_p, f=f)
        cost_gap = relative_cost_gap(cost_dij, cost_p)
        field_error = relative_l2_error(u_dij, u_p)
        path_dict[f"p={p}"] = path_p

        rows.append(
            {
                "dataset": dataset_name,
                "method": "peikonal",
                "p": p,
                "graph_k": used_k,
                "path_length": len_p,
                "path_cost_clean": cost_p,
                "cost_gap_vs_dijkstra": cost_gap,
                "cost_fidelity": cost_fidelity_from_gap(cost_gap),
                "field_vs_dijkstra": field_error,
                "field_fidelity": field_fidelity_from_error(field_error),
                "target_idx": target0,
            }
        )

    if make_figures:
        os.makedirs(fig_dir, exist_ok=True)
        outfile = os.path.join(fig_dir, f"{dataset_name}_clean_paths_all.png")
        plot_paths_on_dataset(
            X=X,
            dataset_name=dataset_name,
            source_idx=source_idx,
            target_idx=clean_ref_target,
            path_dict=path_dict,
            title=f"{dataset_name}: clean setting paths",
            outfile=outfile,
        )

    return pd.DataFrame(rows)



def noisy_experiment_one_dataset(
    dataset_name: str,
    p_list: List[int],
    alpha: float = 0.0,
    repetitions: int = 10,
    sigma_list: Optional[List[float]] = None,
    make_figures: bool = False,
    fig_dir: str = "figures",
    seed: int = 0,
):
    if sigma_list is None:
        sigma_list = DEFAULT_SIGMA_LIST

    bundle = make_dataset(dataset_name, seed=seed)
    X = bundle.X
    source_idx = bundle.source_idx
    target_idx = bundle.target_idx

    W0, G0, _, knn_dist, used_k = ensure_connected_graph(X)
    f = get_cost_field(dataset_name, X, knn_dist, alpha)
    target0 = choose_representative_target(X, target_idx, source_idx)
    clean_ref_target = np.array([target0])

    noise_type = "weight_noise"
    noise_levels = sigma_list

    u_dij_ref, _ = run_dijkstra(G0, source_idx, f=f)
    path_dij_ref = greedy_path_to_source(W0, u_dij_ref, start_idx=target0, source_set=source_idx)

    u_ref_dict: Dict[int, np.ndarray] = {}
    path_ref_dict: Dict[int, List[int]] = {}
    for p in p_list:
        u_ref, _ = run_peikonal(G0, source_idx, p=p, f=f)
        path_ref = greedy_path_to_source(W0, u_ref, start_idx=target0, source_set=source_idx)
        u_ref_dict[p] = u_ref
        path_ref_dict[p] = path_ref

    rows = []

    for noise_level in noise_levels:
        dijkstra_noisy_paths = []
        dijkstra_distances = []
        p_noisy_paths: Dict[int, List[List[int]]] = {p: [] for p in p_list}
        p_distances: Dict[int, List[float]] = {p: [] for p in p_list}

        for rep in range(repetitions):
            Wn = multiply_edge_weights_noise(W0, sigma=float(noise_level), seed=2000 + 100 * seed + rep)
            Gn = gl.graph(Wn)
            if not Gn.isconnected():
                continue

            u_dij, _ = run_dijkstra(Gn, source_idx, f=f)
            path_dij = greedy_path_to_source(Wn, u_dij, start_idx=target0, source_set=source_idx)
            dijkstra_stability = normalized_discrete_frechet_distance(X, path_dij_ref, path_dij)
            dijkstra_robustness = robustness_from_stability(dijkstra_stability)
            dijkstra_noisy_paths.append(path_dij)
            dijkstra_distances.append(dijkstra_stability)

            rows.append(
                {
                    "dataset": dataset_name,
                    "noise_type": noise_type,
                    "noise_level": noise_level,
                    "rep": rep,
                    "method": "dijkstra",
                    "p": np.nan,
                    "graph_k": used_k,
                    "field_error_vs_clean": relative_l2_error(u_dij_ref, u_dij),
                    "stability": dijkstra_stability,
                    "robustness": dijkstra_robustness,
                    "target_idx": target0,
                }
            )

            for p in p_list:
                u_p, _ = run_peikonal(Gn, source_idx, p=p, f=f)
                path_p = greedy_path_to_source(Wn, u_p, start_idx=target0, source_set=source_idx)
                stability_p = normalized_discrete_frechet_distance(X, path_ref_dict[p], path_p)
                robustness_p = robustness_from_stability(stability_p)
                p_noisy_paths[p].append(path_p)
                p_distances[p].append(stability_p)

                rows.append(
                    {
                        "dataset": dataset_name,
                        "noise_type": noise_type,
                        "noise_level": noise_level,
                        "rep": rep,
                        "method": "peikonal",
                        "p": p,
                        "graph_k": used_k,
                        "field_error_vs_clean": relative_l2_error(u_ref_dict[p], u_p),
                        "stability": stability_p,
                        "robustness": robustness_p,
                        "target_idx": target0,
                    }
                )

        if make_figures:
            os.makedirs(fig_dir, exist_ok=True)
            noise_label = str(noise_level).replace(".", "p")

            if len(dijkstra_noisy_paths) > 0:
                ridx = representative_path_index_from_distances(dijkstra_distances)
                plot_path_fan_on_dataset(
                    X=X,
                    source_idx=source_idx,
                    target_idx=clean_ref_target,
                    clean_path=path_dij_ref,
                    noisy_paths=dijkstra_noisy_paths,
                    representative_path=dijkstra_noisy_paths[ridx],
                    title=f"{dataset_name}: Dijkstra fan, noise={noise_level}",
                    outfile=os.path.join(fig_dir, f"{dataset_name}_{noise_type}_{noise_label}_dijkstra_fan.png"),
                )

            for p in p_list:
                if len(p_noisy_paths[p]) == 0:
                    continue
                ridx = representative_path_index_from_distances(p_distances[p])
                plot_path_fan_on_dataset(
                    X=X,
                    source_idx=source_idx,
                    target_idx=clean_ref_target,
                    clean_path=path_ref_dict[p],
                    noisy_paths=p_noisy_paths[p],
                    representative_path=p_noisy_paths[p][ridx],
                    title=f"{dataset_name}: p={p} fan, noise={noise_level}",
                    outfile=os.path.join(fig_dir, f"{dataset_name}_{noise_type}_{noise_label}_p{p}_fan.png"),
                )

    return pd.DataFrame(rows)



def grouped_summary(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "noise_type", "noise_level", "method", "p"]
    agg = (
        df.groupby(keys, dropna=False)[["field_error_vs_clean", "stability", "robustness"]]
        .agg(["mean", "std", "median"])
        .reset_index()
    )
    agg.columns = ["_".join([str(c) for c in col if c != ""]).strip("_") for col in agg.columns.to_flat_index()]
    return agg



def make_summary_plots(clean_df: pd.DataFrame, noisy_grouped_df: pd.DataFrame, outdir: str):
    os.makedirs(outdir, exist_ok=True)

    for dataset in clean_df["dataset"].unique():
        sub = clean_df[clean_df["dataset"] == dataset].copy()
        psub = sub[sub["method"] == "peikonal"].sort_values("p")

        plt.figure(figsize=(5.8, 4.2))
        plt.plot(psub["p"], psub["cost_fidelity"], marker="o", label="cost fidelity vs clean Dijkstra")
        plt.plot(psub["p"], psub["field_fidelity"], marker="o", label="field fidelity vs clean Dijkstra")
        plt.xlabel("p")
        plt.ylabel("Fidelity score (higher is better)")
        plt.title(f"Clean fidelity scores - {dataset}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{dataset}_clean_fidelity_metrics.png"), dpi=220)
        plt.close()

    for dataset in noisy_grouped_df["dataset"].unique():
        sub = noisy_grouped_df[noisy_grouped_df["dataset"] == dataset].copy()
        noise_type = sub["noise_type"].iloc[0]

        plt.figure(figsize=(6.4, 4.4))
        dsub = sub[sub["method"] == "dijkstra"].sort_values("noise_level")
        if len(dsub) > 0:
            plt.errorbar(
                dsub["noise_level"],
                dsub["robustness_median"],
                yerr=dsub["robustness_std"],
                marker="s",
                label="Dijkstra",
            )

        for p in sorted(sub["p"].dropna().unique()):
            psub = sub[(sub["method"] == "peikonal") & (sub["p"] == p)].sort_values("noise_level")
            plt.errorbar(
                psub["noise_level"],
                psub["robustness_median"],
                yerr=psub["robustness_std"],
                marker="o",
                label=f"p={int(p)}",
            )

        plt.xlabel("Noise level")
        plt.ylabel("Robustness score median (higher is better)")
        plt.title(f"{dataset} - {noise_type} robustness")
        plt.grid(True)
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{dataset}_{noise_type}_robustness.png"), dpi=220)
        plt.close()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--p_list", nargs="+", type=int, default=DEFAULT_P_LIST)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="thesis_results")
    parser.add_argument("--make_figures", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    fig_dir = os.path.join(args.output_dir, "figures")
    summary_dir = os.path.join(args.output_dir, "summary_plots")
    selection_dir = os.path.join(args.output_dir, "selection_plots")

    if args.make_figures:
        os.makedirs(fig_dir, exist_ok=True)
        os.makedirs(summary_dir, exist_ok=True)
        os.makedirs(selection_dir, exist_ok=True)

    clean_frames = []
    noisy_frames = []

    for i, dataset_name in enumerate(args.datasets):
        print(f"Running clean experiments on {dataset_name} ...")
        cdf = clean_experiment_one_dataset(
            dataset_name=dataset_name,
            p_list=args.p_list,
            alpha=args.alpha,
            make_figures=args.make_figures,
            fig_dir=fig_dir,
            seed=i,
        )
        clean_frames.append(cdf)

        print(f"Running noisy experiments on {dataset_name} ...")
        ndf = noisy_experiment_one_dataset(
            dataset_name=dataset_name,
            p_list=args.p_list,
            alpha=args.alpha,
            repetitions=args.repetitions,
            make_figures=args.make_figures,
            fig_dir=fig_dir,
            seed=i,
        )
        noisy_frames.append(ndf)

    clean_df = pd.concat(clean_frames, ignore_index=True)
    noisy_df = pd.concat(noisy_frames, ignore_index=True)
    noisy_grouped_df = grouped_summary(noisy_df)

    from selection_utils import (
        build_fidelity_scores,
        build_robustness_scores,
        build_selection_scores,
        detect_elbows,
        make_selection_plots,
        select_best_p_by_score,
    )

    fidelity_df = build_fidelity_scores(clean_df)
    robustness_df = build_robustness_scores(noisy_grouped_df)
    selection_df = build_selection_scores(fidelity_df, robustness_df, betas=(0.25, 0.5, 0.75))

    best_p_025 = select_best_p_by_score(selection_df, beta=0.25)
    best_p_05 = select_best_p_by_score(selection_df, beta=0.5)
    best_p_075 = select_best_p_by_score(selection_df, beta=0.75)
    elbow_df = detect_elbows(selection_df)

    clean_df.to_csv(os.path.join(args.output_dir, "clean_summary.csv"), index=False)
    noisy_df.to_csv(os.path.join(args.output_dir, "noisy_summary_raw.csv"), index=False)
    noisy_grouped_df.to_csv(os.path.join(args.output_dir, "noisy_summary_grouped.csv"), index=False)
    fidelity_df.to_csv(os.path.join(args.output_dir, "fidelity_scores.csv"), index=False)
    robustness_df.to_csv(os.path.join(args.output_dir, "robustness_scores.csv"), index=False)
    selection_df.to_csv(os.path.join(args.output_dir, "selection_scores.csv"), index=False)
    best_p_025.to_csv(os.path.join(args.output_dir, "best_p_beta_025.csv"), index=False)
    best_p_05.to_csv(os.path.join(args.output_dir, "best_p_beta_05.csv"), index=False)
    best_p_075.to_csv(os.path.join(args.output_dir, "best_p_beta_075.csv"), index=False)
    elbow_df.to_csv(os.path.join(args.output_dir, "elbow_fidelity_robustness.csv"), index=False)

    if args.make_figures:
        make_summary_plots(clean_df, noisy_grouped_df, outdir=summary_dir)
        make_selection_plots(selection_df, outdir=selection_dir)

    print("Done.")
    print(f"Saved clean summary to: {os.path.join(args.output_dir, 'clean_summary.csv')}")
    print(f"Saved noisy raw summary to: {os.path.join(args.output_dir, 'noisy_summary_raw.csv')}")
    print(f"Saved noisy grouped summary to: {os.path.join(args.output_dir, 'noisy_summary_grouped.csv')}")


if __name__ == "__main__":
    main()

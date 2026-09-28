import os
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Higher is better throughout:
# - Fidelity score: cost fidelity = 1 / (1 + |cost gap|)
# - Robustness score: 1 / (1 + stability median)
# - Tradeoff score: weighted sum, larger is better


def build_fidelity_scores(clean_df: pd.DataFrame) -> pd.DataFrame:
    peik = clean_df[clean_df["method"] == "peikonal"].copy()
    out_rows = []

    for dataset, sub in peik.groupby("dataset"):
        sub = sub.sort_values("p").copy()
        out_rows.append(
            sub[
                [
                    "dataset",
                    "p",
                    "cost_gap_vs_dijkstra",
                    "cost_fidelity",
                    "field_vs_dijkstra",
                    "field_fidelity",
                ]
            ].rename(columns={"cost_fidelity": "F_score"})
        )

    return pd.concat(out_rows, ignore_index=True)



def build_robustness_scores(noisy_grouped_df: pd.DataFrame) -> pd.DataFrame:
    peik = noisy_grouped_df[noisy_grouped_df["method"] == "peikonal"].copy()
    out_rows = []

    for dataset, sub in peik.groupby("dataset"):
        agg = (
            sub.groupby("p", dropna=False)[["stability_median", "robustness_median"]]
            .mean()
            .reset_index()
            .sort_values("p")
        )
        agg["dataset"] = dataset
        out_rows.append(
            agg[
                [
                    "dataset",
                    "p",
                    "stability_median",
                    "robustness_median",
                ]
            ].rename(columns={"robustness_median": "R_score"})
        )

    return pd.concat(out_rows, ignore_index=True)



def build_selection_scores(
    fidelity_df: pd.DataFrame,
    robustness_df: pd.DataFrame,
    betas: Iterable[float] = (0.25, 0.5, 0.75),
) -> pd.DataFrame:
    df = pd.merge(
        fidelity_df[["dataset", "p", "F_score"]],
        robustness_df[["dataset", "p", "R_score"]],
        on=["dataset", "p"],
        how="inner",
    ).sort_values(["dataset", "p"]).copy()

    for beta in betas:
        df[f"J_beta_{beta}"] = (1 - beta) * df["F_score"] + beta * df["R_score"]

    return df



def select_best_p_by_score(selection_df: pd.DataFrame, beta: float) -> pd.DataFrame:
    col = f"J_beta_{beta}"
    rows = []
    for dataset, sub in selection_df.groupby("dataset"):
        idx = sub[col].idxmax()
        best = sub.loc[idx, ["dataset", "p", col]].copy()
        rows.append(best)
    return pd.DataFrame(rows).rename(columns={col: "best_score"})



def elbow_by_chord_distance(x: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 3:
        return 0, x[0]

    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-12)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-12)

    p1 = np.array([x_norm[0], y_norm[0]])
    p2 = np.array([x_norm[-1], y_norm[-1]])
    v = p2 - p1
    v_norm = np.linalg.norm(v)

    if v_norm < 1e-12:
        return 0, x[0]

    distances = []
    for i in range(len(x_norm)):
        p = np.array([x_norm[i], y_norm[i]])
        dist = abs(np.cross(v, p - p1)) / v_norm
        distances.append(dist)

    distances = np.asarray(distances)
    idx = int(np.argmax(distances))
    return idx, x[idx]



def detect_elbows(selection_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, sub in selection_df.groupby("dataset"):
        sub = sub.sort_values("p").reset_index(drop=True)
        idx, elbow_p = elbow_by_chord_distance(sub["R_score"].values, sub["F_score"].values)
        rows.append({"dataset": dataset, "elbow_p": elbow_p, "elbow_idx": idx})
    return pd.DataFrame(rows)



def make_selection_plots(selection_df: pd.DataFrame, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    elbow_df = detect_elbows(selection_df)

    for dataset, sub in selection_df.groupby("dataset"):
        sub = sub.sort_values("p").copy().reset_index(drop=True)
        elbow_row = elbow_df[elbow_df["dataset"] == dataset]
        elbow_idx = None if elbow_row.empty else elbow_row.iloc[0]["elbow_idx"]

        plt.figure(figsize=(6.4, 5.0))
        plt.plot(sub["R_score"], sub["F_score"], marker="o", linewidth=2)

        for _, row in sub.iterrows():
            plt.annotate(
                f"p={int(row['p'])}",
                (row["R_score"], row["F_score"]),
                fontsize=8,
                xytext=(4, 4),
                textcoords="offset points",
            )

        if elbow_idx is not None and pd.notna(elbow_idx):
            elbow_idx = int(elbow_idx)
            if 0 <= elbow_idx < len(sub):
                erow = sub.iloc[elbow_idx]
                plt.scatter(
                    [erow["R_score"]],
                    [erow["F_score"]],
                    s=120,
                    marker="*",
                    label=f"elbow p={int(erow['p'])}",
                    zorder=5,
                )

        plt.xlabel("Robustness score (higher is better)")
        plt.ylabel("Fidelity score (higher is better)")
        plt.title(f"Fidelity-Robustness curve - {dataset}")
        plt.grid(True)
        if plt.gca().get_legend_handles_labels()[0]:
            plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{dataset}_fidelity_robustness_curve.png"), dpi=220)
        plt.close()

"""
================================================================================
  Advanced Visualizations Module
  PCA · t-SNE · Anomaly scatter · Heatmaps · Histograms · Feature importance
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ── Global aesthetics ────────────────────────────────────────────────────────
PALETTE = ["#00B4D8", "#F77F00", "#2DC653", "#E63946", "#7B2D8B",
           "#FFB703", "#8338EC", "#06D6A0", "#EF476F", "#118AB2"]
ZONE_COLORS = {
    "Industrial Zone":        "#E63946",
    "Traffic-Heavy Zone":     "#F77F00",
    "Construction Zone":      "#FFB703",
    "Transport Hub":          "#8338EC",
    "Residential Quiet Zone": "#2DC653",
    "Commercial Zone":        "#00B4D8",
    "Green/Park Zone":        "#06D6A0",
    "Mixed Zone":             "#ADB5BD",
    "Noise Anomaly":          "#FF0000",
}

plt.rcParams.update({
    "figure.facecolor": "#0D1117",
    "axes.facecolor":   "#161B22",
    "axes.edgecolor":   "#30363D",
    "axes.labelcolor":  "#C9D1D9",
    "text.color":       "#C9D1D9",
    "xtick.color":      "#8B949E",
    "ytick.color":      "#8B949E",
    "grid.color":       "#21262D",
    "font.family":      "DejaVu Sans",
})

SAVE_DIR = Path("reports/figures")
SAVE_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig, name: str):
    path = SAVE_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  📊 Saved → {path}")
    return str(path)


# ─────────────────────────────────────────────────────────────────────────────
#  1. PCA Cluster Visualization
# ─────────────────────────────────────────────────────────────────────────────

def plot_pca_clusters(pca_2d: np.ndarray, labels: np.ndarray,
                      zone_map: dict = None, title="PCA Cluster View") -> str:
    fig, ax = plt.subplots(figsize=(11, 8))
    unique = sorted(set(labels))
    for i, lbl in enumerate(unique):
        mask  = labels == lbl
        color = PALETTE[i % len(PALETTE)]
        zlabel = zone_map.get(lbl, f"Cluster {lbl}") if zone_map else f"Cluster {lbl}"
        ax.scatter(pca_2d[mask, 0], pca_2d[mask, 1],
                   c=color, label=zlabel, s=18, alpha=0.72, edgecolors="none")

    ax.set_xlabel("PC 1", fontsize=12)
    ax.set_ylabel("PC 2", fontsize=12)
    ax.set_title(title, fontsize=15, fontweight="bold", color="#58A6FF", pad=14)
    ax.legend(loc="best", fontsize=9, framealpha=0.3)
    ax.grid(True, alpha=0.15)
    return _save(fig, "pca_clusters.png")


# ─────────────────────────────────────────────────────────────────────────────
#  2. t-SNE Visualization
# ─────────────────────────────────────────────────────────────────────────────

def plot_tsne(tsne_2d: np.ndarray, labels: np.ndarray,
              zone_map: dict = None, title="t-SNE Cluster View") -> str:
    fig, ax = plt.subplots(figsize=(11, 8))
    unique = sorted(set(labels))
    for i, lbl in enumerate(unique):
        mask   = labels == lbl
        color  = PALETTE[i % len(PALETTE)]
        zlabel = zone_map.get(lbl, f"Cluster {lbl}") if zone_map else f"Cluster {lbl}"
        ax.scatter(tsne_2d[mask, 0], tsne_2d[mask, 1],
                   c=color, label=zlabel, s=18, alpha=0.72, edgecolors="none")

    ax.set_xlabel("t-SNE 1", fontsize=12)
    ax.set_ylabel("t-SNE 2", fontsize=12)
    ax.set_title(title, fontsize=15, fontweight="bold", color="#58A6FF", pad=14)
    ax.legend(loc="best", fontsize=9, framealpha=0.3)
    ax.grid(True, alpha=0.15)
    return _save(fig, "tsne_clusters.png")


# ─────────────────────────────────────────────────────────────────────────────
#  3. Anomaly Scatter Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_anomaly_scatter(df: pd.DataFrame,
                         pca_2d: np.ndarray,
                         anomaly_col: str = "if_anomaly") -> str:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, col in zip(axes, [anomaly_col, "if_score"]):
        if col == anomaly_col and col in df.columns:
            is_anom = df[col] == -1
            ax.scatter(pca_2d[~is_anom, 0], pca_2d[~is_anom, 1],
                       c="#00B4D8", s=14, alpha=0.5, label="Normal", edgecolors="none")
            ax.scatter(pca_2d[is_anom, 0],  pca_2d[is_anom, 1],
                       c="#E63946", s=55, alpha=0.9, label="Anomaly",
                       edgecolors="#FFB703", linewidths=0.6)
            ax.set_title("Anomaly Detection (Isolation Forest)",
                         fontsize=13, fontweight="bold", color="#58A6FF")
            ax.legend(fontsize=10, framealpha=0.3)

        elif col == "if_score" and col in df.columns:
            scores = df[col].values
            sc = ax.scatter(pca_2d[:, 0], pca_2d[:, 1],
                            c=scores, cmap="RdYlGn_r", s=14, alpha=0.7, edgecolors="none")
            plt.colorbar(sc, ax=ax, label="Anomaly Score")
            ax.set_title("Anomaly Score Distribution",
                         fontsize=13, fontweight="bold", color="#58A6FF")
        else:
            ax.set_visible(False)
            continue

        ax.set_xlabel("PC 1", fontsize=11)
        ax.set_ylabel("PC 2", fontsize=11)
        ax.grid(True, alpha=0.12)

    fig.suptitle("Anomaly Analysis — PCA Space", fontsize=16,
                 fontweight="bold", color="#C9D1D9", y=1.01)
    return _save(fig, "anomaly_scatter.png")


# ─────────────────────────────────────────────────────────────────────────────
#  4. Noise Density Histogram
# ─────────────────────────────────────────────────────────────────────────────

def plot_noise_histogram(df: pd.DataFrame, cluster_col: str = "kmeans_cluster",
                         zone_map: dict = None) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Overall distribution
    ax = axes[0]
    ax.hist(df["avg_decibels"], bins=50, color="#00B4D8", edgecolor="#0D1117",
            alpha=0.85, density=True)
    ax.axvline(55, color="#2DC653", lw=1.5, linestyle="--", label="WHO guideline 55 dB")
    ax.axvline(70, color="#FFB703", lw=1.5, linestyle="--", label="Alert threshold 70 dB")
    ax.axvline(85, color="#E63946", lw=1.5, linestyle="--", label="Danger threshold 85 dB")
    ax.set_xlabel("Average Decibels (dB)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("City-Wide Noise Distribution", fontsize=13,
                 fontweight="bold", color="#58A6FF")
    ax.legend(fontsize=9, framealpha=0.3)
    ax.grid(True, alpha=0.12)

    # Per-cluster KDE
    ax = axes[1]
    unique = sorted(df[cluster_col].unique())
    for i, lbl in enumerate(unique):
        sub   = df[df[cluster_col] == lbl]["avg_decibels"]
        color = PALETTE[i % len(PALETTE)]
        zlabel = zone_map.get(lbl, f"Cluster {lbl}") if zone_map else f"Cluster {lbl}"
        sub.plot.kde(ax=ax, label=zlabel, color=color, linewidth=2.0)

    ax.axvline(70, color="#E63946", lw=1.5, linestyle="--", alpha=0.7, label="Alert 70 dB")
    ax.set_xlabel("Average Decibels (dB)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Noise Distribution per Cluster", fontsize=13,
                 fontweight="bold", color="#58A6FF")
    ax.legend(fontsize=9, framealpha=0.3)
    ax.grid(True, alpha=0.12)

    fig.suptitle("Noise Level Density Analysis", fontsize=16,
                 fontweight="bold", color="#C9D1D9")
    fig.tight_layout()
    return _save(fig, "noise_histogram.png")


# ─────────────────────────────────────────────────────────────────────────────
#  5. Correlation Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame) -> str:
    num_cols = [
        "avg_decibels", "traffic_density", "vehicle_count",
        "honking_frequency", "industrial_activity", "construction_activity",
        "population_density", "temperature_c", "humidity_pct",
        "wind_speed_kmh", "heavy_vehicle_pct", "event_activity",
    ]
    num_cols = [c for c in num_cols if c in df.columns]
    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(13, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=0.5, annot=True, fmt=".2f",
                annot_kws={"size": 8}, ax=ax, cbar_kws={"shrink": 0.85})
    ax.set_title("Feature Correlation Heatmap", fontsize=15,
                 fontweight="bold", color="#58A6FF", pad=14)
    return _save(fig, "correlation_heatmap.png")


# ─────────────────────────────────────────────────────────────────────────────
#  6. Feature Importance (PCA Loadings)
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 15) -> str:
    top = importance_df.head(top_n).copy()
    fig, ax = plt.subplots(figsize=(12, 7))
    y_pos = range(len(top))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(top))]

    bars = ax.barh(y_pos, top["total_importance"], color=colors,
                   edgecolor="#0D1117", alpha=0.88)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top.index, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Aggregate PCA Loading (PC1+PC2)", fontsize=11)
    ax.set_title(f"Top {top_n} Feature Importances (PCA)", fontsize=14,
                 fontweight="bold", color="#58A6FF", pad=14)
    ax.grid(True, axis="x", alpha=0.15)

    for bar, val in zip(bars, top["total_importance"]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=9, color="#C9D1D9")

    return _save(fig, "feature_importance.png")


# ─────────────────────────────────────────────────────────────────────────────
#  7. Cluster Summary Bar Chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_cluster_summary_bars(summary_df: pd.DataFrame) -> str:
    db_col = "avg_decibels_mean" if "avg_decibels_mean" in summary_df.columns else "avg_decibels"
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    labels  = summary_df["zone_label"].tolist()
    metrics = [
        (db_col,                   "Avg Decibels (dB)",         "#00B4D8"),
        ("traffic_density_mean",   "Avg Traffic Density",       "#F77F00"),
        ("industrial_activity_mean","Avg Industrial Activity",  "#E63946"),
    ]
    for ax, (col, ylabel, color) in zip(axes, metrics):
        col = col if col in summary_df.columns else col.replace("_mean","")
        if col not in summary_df.columns:
            ax.set_visible(False)
            continue
        values = summary_df[col].tolist()
        bar_colors = [PALETTE[i % len(PALETTE)] for i in range(len(labels))]
        bars = ax.bar(range(len(labels)), values, color=bar_colors, edgecolor="#0D1117", alpha=0.88)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(ylabel, fontsize=11, fontweight="bold", color="#58A6FF")
        ax.grid(True, axis="y", alpha=0.15)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=8, color="#C9D1D9")

    fig.suptitle("Cluster-wise Key Metrics", fontsize=16,
                 fontweight="bold", color="#C9D1D9", y=1.02)
    fig.tight_layout()
    return _save(fig, "cluster_summary_bars.png")


# ─────────────────────────────────────────────────────────────────────────────
#  8. Peak Hour Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_peak_hour_heatmap(df: pd.DataFrame) -> str:
    pivot = (
        df.groupby(["hour", "day_of_week"])["avg_decibels"]
        .mean()
        .unstack(fill_value=0)
    )
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    pivot.columns = [day_names[c] for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(13, 8))
    sns.heatmap(pivot, cmap="RdYlGn_r", vmin=40, vmax=100,
                linewidths=0.3, ax=ax, cbar_kws={"label": "Mean dB"})
    ax.set_xlabel("Day of Week", fontsize=11)
    ax.set_ylabel("Hour of Day", fontsize=11)
    ax.set_title("Noise Level Heatmap — Hour × Day", fontsize=14,
                 fontweight="bold", color="#58A6FF", pad=12)
    return _save(fig, "peak_hour_heatmap.png")


# ─────────────────────────────────────────────────────────────────────────────
#  9. Model Comparison Radar Chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_model_comparison(metrics: dict) -> str:
    """
    metrics = {
      "KMeans":    {"Silhouette":0.45, "DBI":1.2, "CH":380},
      "DBSCAN":    {...},
      ...
    }
    """
    import matplotlib.patches as mpatches

    models     = list(metrics.keys())
    all_keys   = list(list(metrics.values())[0].keys())
    n_metrics  = len(all_keys)
    angles     = np.linspace(0, 2*np.pi, n_metrics, endpoint=False).tolist()
    angles    += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    ax.set_facecolor("#161B22")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), all_keys, fontsize=11, color="#C9D1D9")
    ax.tick_params(axis="y", colors="#8B949E")

    for i, model in enumerate(models):
        raw_vals  = [metrics[model][k] for k in all_keys]
        # Normalise 0-1 per metric for radar
        maxv = [max(metrics[m][k] for m in models) for k in all_keys]
        vals = [v/mx if mx > 0 else 0 for v, mx in zip(raw_vals, maxv)]
        vals += vals[:1]
        color = PALETTE[i % len(PALETTE)]
        ax.plot(angles, vals, "o-", linewidth=2, color=color, label=model)
        ax.fill(angles, vals, color=color, alpha=0.10)

    ax.set_title("Model Comparison — Radar Chart", fontsize=14,
                 fontweight="bold", color="#58A6FF", pad=25)
    ax.legend(loc="lower right", bbox_to_anchor=(1.3, -0.1), fontsize=10,
              framealpha=0.25)
    return _save(fig, "model_comparison_radar.png")


# ─────────────────────────────────────────────────────────────────────────────
#  10. Silhouette Score Elbow
# ─────────────────────────────────────────────────────────────────────────────

def plot_elbow_silhouette(elbow_data: dict) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(elbow_data["k"], elbow_data["inertia"], "o-",
            color="#00B4D8", linewidth=2.2, markersize=7)
    ax.set_xlabel("Number of Clusters (k)", fontsize=11)
    ax.set_ylabel("Inertia (WCSS)", fontsize=11)
    ax.set_title("Elbow Curve", fontsize=13, fontweight="bold", color="#58A6FF")
    ax.grid(True, alpha=0.15)

    ax = axes[1]
    ax.plot(elbow_data["k"], elbow_data["silhouette"], "s-",
            color="#2DC653", linewidth=2.2, markersize=7)
    best_k = elbow_data["k"][int(np.argmax(elbow_data["silhouette"]))]
    ax.axvline(best_k, color="#E63946", linestyle="--", lw=1.5, label=f"Best k={best_k}")
    ax.set_xlabel("Number of Clusters (k)", fontsize=11)
    ax.set_ylabel("Silhouette Score", fontsize=11)
    ax.set_title("Silhouette Score vs k", fontsize=13, fontweight="bold", color="#58A6FF")
    ax.legend(fontsize=10, framealpha=0.3)
    ax.grid(True, alpha=0.15)

    fig.suptitle("Optimal Cluster Selection", fontsize=15,
                 fontweight="bold", color="#C9D1D9")
    fig.tight_layout()
    return _save(fig, "elbow_silhouette.png")

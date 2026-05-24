"""
================================================================================
  Main Pipeline Runner
  Run full end-to-end pipeline: generate → preprocess → cluster → anomaly
  → dimensionality reduction → cluster analysis → visualizations → maps
================================================================================
"""

import sys, os
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from data_generator           import generate_dataset
from preprocessing            import NoisePollutionPreprocessor, add_noise_severity_label
from clustering               import KMeansClustering, DBSCANClustering, GMMClustering
from anomaly_detection        import IsolationForestDetector
from dimensionality_reduction import DimensionalityReducer
from cluster_analysis         import ClusterAnalyzer
from visualizations           import (
    plot_pca_clusters, plot_tsne, plot_anomaly_scatter,
    plot_noise_histogram, plot_correlation_heatmap,
    plot_feature_importance, plot_cluster_summary_bars,
    plot_peak_hour_heatmap, plot_elbow_silhouette,
)
from noise_maps               import create_master_map, create_noise_heatmap, \
                                     create_anomaly_map, create_cluster_map
from model_comparison         import evaluate_clustering, evaluate_anomaly_detector, \
                                     build_comparison_table


def run_pipeline(n_samples: int = 5000, n_clusters: int = 5):
    print("\n" + "="*70)
    print("  SMART CITY NOISE POLLUTION — END-TO-END PIPELINE")
    print("="*70)

    # ── Step 1: Data Generation ──────────────────────────────────────────────
    print("\n[1/8] Generating synthetic smart-city noise dataset …")
    df = generate_dataset(n_samples=n_samples, save_path="data/smart_city_noise.csv")
    df = add_noise_severity_label(df)
    print(f"  ✅ Dataset: {df.shape}")

    # ── Step 2: Preprocessing ────────────────────────────────────────────────
    print("\n[2/8] Preprocessing & feature engineering …")
    prep = NoisePollutionPreprocessor()
    X    = prep.fit_transform(df)
    print(f"  ✅ Feature matrix: {X.shape}")

    # ── Step 3: Clustering ───────────────────────────────────────────────────
    print(f"\n[3/8] K-Means clustering (k={n_clusters}) …")
    km      = KMeansClustering(n_clusters=n_clusters)
    km.fit(X)
    df["kmeans_cluster"] = km.labels_
    km.save("models/kmeans.pkl")

    print("  DBSCAN clustering …")
    dbs = DBSCANClustering(eps=0.6, min_samples=8)
    dbs.fit(X)
    df["dbscan_cluster"] = dbs.labels_
    dbs.save("models/dbscan.pkl")

    print("  GMM clustering …")
    gmm = GMMClustering(n_components=n_clusters)
    gmm.fit(X)
    df["gmm_cluster"] = gmm.labels_
    gmm.save()

    print("  ✅ Clustering complete")

    # ── Step 4: Elbow analysis ───────────────────────────────────────────────
    print("\n[4/8] Computing elbow curve …")
    elbow = km.elbow_curve(X)
    plot_elbow_silhouette(elbow)

    # ── Step 5: Anomaly Detection ────────────────────────────────────────────
    print("\n[5/8] Anomaly detection (Isolation Forest) …")
    ifd = IsolationForestDetector(contamination=0.05)
    ifd.fit(X)
    df["if_anomaly"] = ifd.predict(X)
    df["if_score"]   = ifd.anomaly_scores(X)
    ifd.save("models/isolation_forest.pkl")
    print(f"  ✅ Anomalies detected: {(df['if_anomaly']==-1).sum()}")

    # ── Step 6: Dimensionality Reduction ─────────────────────────────────────
    print("\n[6/8] Dimensionality reduction (PCA + t-SNE) …")
    dr = DimensionalityReducer()
    dr.fit_pca(X)
    pca_2d = dr.transform_pca(X, dims=2)
    dr.save("models")

    print("  Running t-SNE …")
    tsne_2d = dr.tsne_2d(X)
    fi_df   = dr.feature_importance_pca(prep.feature_cols)
    print("  ✅ Dimensionality reduction complete")

    # ── Step 7: Cluster Analysis ─────────────────────────────────────────────
    print("\n[7/8] Cluster analysis & zone identification …")
    ca = ClusterAnalyzer(df, cluster_col="kmeans_cluster")
    ca.compute_summary()
    zone_map = ca.build_zone_map()
    df["zone_label"] = df["kmeans_cluster"].map(zone_map)
    ca.print_report()

    # ── Step 8: Visualizations & Maps ────────────────────────────────────────
    print("\n[8/8] Generating visualizations & maps …")

    plot_pca_clusters(pca_2d, df["kmeans_cluster"].values, zone_map)
    plot_tsne(tsne_2d, df["kmeans_cluster"].values[:len(tsne_2d)], zone_map)
    plot_anomaly_scatter(df, pca_2d, anomaly_col="if_anomaly")
    plot_noise_histogram(df, zone_map=zone_map)
    plot_correlation_heatmap(df)
    plot_feature_importance(fi_df)
    plot_cluster_summary_bars(ca.summary_df)
    plot_peak_hour_heatmap(df)

    print("  Generating interactive maps …")
    create_noise_heatmap(df.sample(min(2000,len(df)), random_state=42))
    create_cluster_map(df.sample(min(2000,len(df)), random_state=42), zone_map=zone_map)
    create_anomaly_map(df)
    create_master_map(df.sample(min(2000,len(df)), random_state=42), zone_map=zone_map)

    # ── Model Comparison ─────────────────────────────────────────────────────
    print("\n--- Model Comparison Metrics ---")
    true_bin   = df.get("is_anomaly_label", pd.Series(np.zeros(len(df)))).values
    km_metrics = evaluate_clustering(X, df["kmeans_cluster"].values, "K-Means")
    db_metrics = evaluate_clustering(X, df["dbscan_cluster"].values, "DBSCAN")
    gm_metrics = evaluate_clustering(X, df["gmm_cluster"].values,    "GMM")
    if_metrics = evaluate_anomaly_detector(
        df["if_score"].values,
        (df["if_anomaly"]==-1).astype(int).values,
        true_bin,
        "Isolation Forest"
    )
    table = build_comparison_table(
        [km_metrics, db_metrics, gm_metrics],
        [if_metrics]
    )
    print(table.to_string(index=False))

    # Save enriched dataset
    df.to_csv("data/smart_city_noise_enriched.csv", index=False)
    print("\n  ✅ Enriched dataset saved → data/smart_city_noise_enriched.csv")
    print("\n✅ PIPELINE COMPLETE")
    print("="*70)
    return df, X, pca_2d, tsne_2d, zone_map, ca


if __name__ == "__main__":
    run_pipeline(n_samples=5000, n_clusters=5)

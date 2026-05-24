"""
================================================================================
  🏙️  Smart City Noise Pollution Dashboard
  Streamlit Interactive Application
  Full pipeline: load → cluster → anomaly → visualize → simulate → report
================================================================================
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import time
import warnings
warnings.filterwarnings("ignore")

# # ── Internal modules ──────────────────────────────────────────────────────────
# from data_generator  import generate_dataset
# from preprocessing   import NoisePollutionPreprocessor, add_noise_severity_label
# from clustering      import KMeansClustering, DBSCANClustering, GMMClustering
# from anomaly_detection import IsolationForestDetector
# from dimensionality_reduction import DimensionalityReducer
# from cluster_analysis import ClusterAnalyzer
# from noise_maps      import create_noise_heatmap, create_cluster_map, \
#                             create_anomaly_map, create_master_map
# from realtime_simulator import RealTimeSimulator, get_alert_level, classify_noise_severity
# from model_comparison import evaluate_clustering, evaluate_anomaly_detector, \
#                              build_comparison_table
# ── Internal modules ──────────────────────────────────────────────────────────
from generate_data import generate_dataset

from data_preprocessing import (
    NoisePollutionPreprocessor,
    add_noise_severity_label
)

from clustering import (
    KMeansClustering,
    DBSCANClustering,
    GMMClustering,
    evaluate_clustering
)

from anomaly_detection import IsolationForestDetector

from dimensionality_reduction import DimensionalityReducer

from cluster_analysis import ClusterAnalyzer

from noise_maps import (
    create_noise_heatmap,
    create_cluster_map,
    create_anomaly_map,
    create_master_map
)

from realtime_simulator import (
    RealTimeSimulator,
    get_alert_level,
    classify_noise_severity
)

from model_comparison import (
    evaluate_anomaly_detector,
    build_comparison_table
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🏙️ Smart City Noise Dashboard",
    page_icon="🔊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Dark theme overrides */
  [data-testid="stAppViewContainer"] { background: #0D1117; }
  [data-testid="stSidebar"]          { background: #161B22; }
  .main-title {
    font-size: 2.4rem; font-weight: 800;
    background: linear-gradient(90deg, #00B4D8, #58A6FF, #BC6FF1);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    text-align: center; margin-bottom: 0.2rem;
  }
  .metric-card {
    background: #161B22; border: 1px solid #30363D; border-radius: 10px;
    padding: 16px 20px; text-align: center;
  }
  .metric-value { font-size: 2rem; font-weight: 700; color: #58A6FF; }
  .metric-label { font-size: 0.85rem; color: #8B949E; margin-top: 4px; }
  .alert-critical { background:#3D0B0B; border-left:4px solid #E63946;
                    padding:10px 14px; border-radius:6px; margin:6px 0; }
  .alert-high     { background:#3D1C0B; border-left:4px solid #F77F00;
                    padding:10px 14px; border-radius:6px; margin:6px 0; }
  .alert-moderate { background:#3D300B; border-left:4px solid #FFB703;
                    padding:10px 14px; border-radius:6px; margin:6px 0; }
  .alert-normal   { background:#0B3D1C; border-left:4px solid #2DC653;
                    padding:10px 14px; border-radius:6px; margin:6px 0; }
  .rec-item { background:#161B22; border:1px solid #30363D; border-radius:8px;
              padding:10px 14px; margin:6px 0; font-size:0.88rem; color:#C9D1D9; }
  .section-header {
    color:#58A6FF; font-size:1.3rem; font-weight:700;
    border-bottom:1px solid #30363D; padding-bottom:6px; margin:18px 0 12px;
  }
  div[data-testid="stMetric"] > div { background:#161B22; border-radius:8px;
                                       padding:12px; border:1px solid #30363D; }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  Data & Model Caching
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="⚙️  Generating smart-city dataset …")
def load_or_generate_data(n_samples: int = 3000) -> pd.DataFrame:
    df = generate_dataset(n_samples=n_samples, save_path="data/smart_city_noise.csv")
    df = add_noise_severity_label(df)
    return df


@st.cache_data(show_spinner="🔧 Running ML pipeline …")
def run_pipeline(n_samples: int = 3000, n_clusters: int = 5):
    df = load_or_generate_data(n_samples)
    prep = NoisePollutionPreprocessor()
    X    = prep.fit_transform(df)

    # KMeans
    km = KMeansClustering(n_clusters=n_clusters)
    km.fit(X)
    df["kmeans_cluster"] = km.labels_

    # DBSCAN
    dbs = DBSCANClustering(eps=0.6, min_samples=8)
    dbs.fit(X)
    df["dbscan_cluster"] = dbs.labels_

    # Isolation Forest
    ifd = IsolationForestDetector(contamination=0.05)
    ifd.fit(X)
    df["if_anomaly"] = ifd.predict(X)
    df["if_score"]   = ifd.anomaly_scores(X)

    # Dimensionality reduction
    dr = DimensionalityReducer()
    dr.fit_pca(X)
    pca_2d = dr.transform_pca(X, dims=2)

    # Cluster analysis
    ca = ClusterAnalyzer(df, cluster_col="kmeans_cluster")
    ca.compute_summary()
    zone_map = ca.build_zone_map()
    df["zone_label"] = df["kmeans_cluster"].map(zone_map)

    # Elbow data
    elbow = km.elbow_curve(X)

    return df, X, pca_2d, zone_map, ca, elbow, prep


# ═════════════════════════════════════════════════════════════════════════════
#  Sidebar
# ═════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/city.png", width=70)
    st.markdown("## 🏙️ Smart City Noise")
    st.markdown("**Noise Pollution Zone Identification**")
    st.divider()

    page = st.radio("📌 Navigation", [
        "🏠 Overview",
        "🗺️ Noise Hotspot Map",
        "📊 Cluster Analysis",
        "🚨 Anomaly Detection",
        "📈 Analytics & Trends",
        "⚡ Real-Time Simulation",
        "🤖 Model Comparison",
        "📋 Reports & Insights",
    ])

    st.divider()
    st.markdown("### ⚙️ Settings")
    n_samples  = st.slider("Dataset Size",   1000, 5000, 3000, 500)
    n_clusters = st.slider("KMeans Clusters", 3, 10, 5)

    if st.button("🔄 Re-run Pipeline", type="primary", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.caption("Built with ❤️ | Unsupervised ML Project")
    st.caption("Stack: Python · Sklearn · Streamlit · Folium · Plotly")


# ── Load data ─────────────────────────────────────────────────────────────────
df, X, pca_2d, zone_map, ca, elbow, prep = run_pipeline(n_samples, n_clusters)
summary_df = ca.summary_df
db_col = "avg_decibels_mean" if "avg_decibels_mean" in summary_df.columns else "avg_decibels"
td_col = "traffic_density_mean" if "traffic_density_mean" in summary_df.columns else "traffic_density"


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE: Overview
# ═════════════════════════════════════════════════════════════════════════════

if page == "🏠 Overview":
    st.markdown('<div class="main-title">🔊 Smart City Noise Pollution Monitor</div>',
                unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#8B949E;margin-bottom:24px'>"
                "Unsupervised Machine Learning · New Delhi Smart City Simulation</p>",
                unsafe_allow_html=True)

    # ── KPI row ───────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    anomaly_count = int((df["if_anomaly"] == -1).sum())
    high_noise    = int((df["avg_decibels"] > 70).sum())

    c1.metric("📍 Data Points",    f"{len(df):,}")
    c2.metric("🔊 Avg Noise (dB)", f"{df['avg_decibels'].mean():.1f}")
    c3.metric("🏘️ Clusters",       n_clusters)
    c4.metric("🚨 Anomalies",      anomaly_count)
    c5.metric("⚠️ High-Noise Pts", high_noise)

    st.markdown("---")

    # ── Quick charts row ──────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Noise Distribution</div>',
                    unsafe_allow_html=True)
        fig = px.histogram(df, x="avg_decibels", nbins=50, color_discrete_sequence=["#00B4D8"],
                           labels={"avg_decibels": "Decibels (dB)"})
        fig.add_vline(x=55, line_dash="dash", line_color="#2DC653", annotation_text="WHO 55dB")
        fig.add_vline(x=70, line_dash="dash", line_color="#FFB703", annotation_text="Alert 70dB")
        fig.add_vline(x=85, line_dash="dash", line_color="#E63946", annotation_text="Danger 85dB")
        fig.update_layout(template="plotly_dark", plot_bgcolor="#161B22",
                          paper_bgcolor="#161B22", height=330, margin=dict(t=30))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Zone Distribution</div>',
                    unsafe_allow_html=True)
        zone_counts = df["zone_label"].value_counts().reset_index()
        zone_counts.columns = ["Zone", "Count"]
        fig = px.pie(zone_counts, values="Count", names="Zone", hole=0.45,
                     color_discrete_sequence=px.colors.qualitative.Set3)
        fig.update_layout(template="plotly_dark", paper_bgcolor="#161B22",
                          height=330, margin=dict(t=30))
        st.plotly_chart(fig, use_container_width=True)

    # ── Hour & zone noise trend ───────────────────────────────────────────────
    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="section-header">Hourly Noise Trend</div>',
                    unsafe_allow_html=True)
        hourly = df.groupby("hour")["avg_decibels"].mean().reset_index()
        fig = px.line(hourly, x="hour", y="avg_decibels", markers=True,
                      color_discrete_sequence=["#00B4D8"],
                      labels={"hour":"Hour of Day","avg_decibels":"Avg dB"})
        fig.update_layout(template="plotly_dark", plot_bgcolor="#161B22",
                          paper_bgcolor="#161B22", height=280, margin=dict(t=30))
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.markdown('<div class="section-header">Avg Noise per Zone Type</div>',
                    unsafe_allow_html=True)
        zone_db = df.groupby("zone_type")["avg_decibels"].mean().reset_index().sort_values("avg_decibels")
        fig = px.bar(zone_db, x="avg_decibels", y="zone_type", orientation="h",
                     color="avg_decibels", color_continuous_scale="RdYlGn_r",
                     labels={"avg_decibels":"Avg dB","zone_type":"Zone"})
        fig.update_layout(template="plotly_dark", plot_bgcolor="#161B22",
                          paper_bgcolor="#161B22", height=280, margin=dict(t=30))
        st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE: Noise Hotspot Map
# ═════════════════════════════════════════════════════════════════════════════

elif page == "🗺️ Noise Hotspot Map":
    st.markdown('<div class="main-title">🗺️ Smart City Noise Hotspot Map</div>',
                unsafe_allow_html=True)

    map_type = st.radio("Map Type", [
        "🔥 Noise Heatmap", "📍 Cluster Zones", "⚠️ Anomaly Hotspots", "🌐 Master Map"
    ], horizontal=True)

    sample_df = df.sample(min(2000, len(df)), random_state=42)

    if map_type == "🔥 Noise Heatmap":
        m = create_noise_heatmap(sample_df)
    elif map_type == "📍 Cluster Zones":
        m = create_cluster_map(sample_df, zone_map=zone_map)
    elif map_type == "⚠️ Anomaly Hotspots":
        m = create_anomaly_map(sample_df)
    else:
        m = create_master_map(sample_df, zone_map=zone_map)

    st_folium(m, width=None, height=580)

    # Plotly 3D scatter for noise
    st.markdown('<div class="section-header">3D Noise Scatter (lat · lon · dB)</div>',
                unsafe_allow_html=True)
    fig3d = px.scatter_3d(
        sample_df, x="longitude", y="latitude", z="avg_decibels",
        color="avg_decibels", color_continuous_scale="RdYlGn_r",
        size="avg_decibels", size_max=8,
        hover_data=["zone_type", "traffic_density", "hour"],
        opacity=0.75,
        labels={"avg_decibels": "dB"},
    )
    fig3d.update_layout(template="plotly_dark", paper_bgcolor="#0D1117",
                        height=520, margin=dict(t=20))
    st.plotly_chart(fig3d, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE: Cluster Analysis
# ═════════════════════════════════════════════════════════════════════════════

elif page == "📊 Cluster Analysis":
    st.markdown('<div class="main-title">📊 Cluster Analysis</div>', unsafe_allow_html=True)

    # Summary table
    st.markdown('<div class="section-header">Cluster Summary Table</div>',
                unsafe_allow_html=True)
    display_cols = ["cluster_id", "zone_label", "count",
                    db_col, td_col,
                    "industrial_activity_mean", "honking_frequency_mean"]
    display_cols = [c for c in display_cols if c in summary_df.columns]
    st.dataframe(
        summary_df[display_cols].style.background_gradient(
            cmap="RdYlGn_r", subset=[db_col] if db_col in display_cols else []
        ),
        use_container_width=True, height=240
    )

    # PCA scatter
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">PCA Cluster Visualization</div>',
                    unsafe_allow_html=True)
        pca_df = pd.DataFrame(pca_2d, columns=["PC1", "PC2"])
        pca_df["cluster"] = df["kmeans_cluster"].astype(str)
        pca_df["zone"]    = df.get("zone_label", df["kmeans_cluster"]).astype(str)
        pca_df["dB"]      = df["avg_decibels"]
        fig = px.scatter(pca_df, x="PC1", y="PC2", color="zone",
                         hover_data=["dB"], opacity=0.7, size_max=6,
                         color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(template="plotly_dark", plot_bgcolor="#161B22",
                          paper_bgcolor="#161B22", height=400, margin=dict(t=30))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Average Noise per Cluster</div>',
                    unsafe_allow_html=True)
        fig = px.bar(summary_df, x="zone_label", y=db_col,
                     color=db_col, color_continuous_scale="RdYlGn_r",
                     labels={db_col: "Avg dB", "zone_label": "Zone"},
                     text_auto=".1f")
        fig.update_layout(template="plotly_dark", plot_bgcolor="#161B22",
                          paper_bgcolor="#161B22", height=400,
                          xaxis_tickangle=-30, margin=dict(t=30))
        st.plotly_chart(fig, use_container_width=True)

    # Risk scores
    st.markdown('<div class="section-header">Zone Risk Scores</div>',
                unsafe_allow_html=True)
    risk_df = ca.risk_score()
    col3, col4 = st.columns([2, 1])
    with col3:
        fig = px.bar(risk_df, x="zone_label", y="risk_score",
                     color="risk_score", color_continuous_scale="RdYlGn_r",
                     text_auto=".1f",
                     labels={"risk_score": "Risk Score (0-100)", "zone_label": "Zone"})
        fig.update_layout(template="plotly_dark", plot_bgcolor="#161B22",
                          paper_bgcolor="#161B22", height=320,
                          xaxis_tickangle=-25, margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)
    with col4:
        st.dataframe(risk_df[["zone_label","risk_score","risk_level"]],
                     use_container_width=True, height=320)

    # Peak hour per cluster
    st.markdown('<div class="section-header">Peak Noise Hour per Cluster</div>',
                unsafe_allow_html=True)
    peak_df = ca.peak_hour_analysis()
    fig = px.line(peak_df, x="hour", y="mean_db", color="cluster_id",
                  markers=True, labels={"mean_db": "Avg dB", "hour": "Hour"})
    fig.update_layout(template="plotly_dark", plot_bgcolor="#161B22",
                      paper_bgcolor="#161B22", height=330, margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)

    # Recommendations
    st.markdown('<div class="section-header">🧠 Smart Recommendations</div>',
                unsafe_allow_html=True)
    for rec in ca.generate_recommendations():
        st.markdown(f'<div class="rec-item">{rec}</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE: Anomaly Detection
# ═════════════════════════════════════════════════════════════════════════════

elif page == "🚨 Anomaly Detection":
    st.markdown('<div class="main-title">🚨 Anomaly Detection Monitor</div>',
                unsafe_allow_html=True)

    n_anom   = int((df["if_anomaly"] == -1).sum())
    anom_pct = n_anom / len(df) * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Anomalies",   n_anom)
    c2.metric("Anomaly Rate",      f"{anom_pct:.1f}%")
    c3.metric("Max dB (Anomaly)",  f"{df[df['if_anomaly']==-1]['avg_decibels'].max():.1f}")
    c4.metric("Avg Anomaly Score", f"{df['if_score'].mean():.3f}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Anomaly Score Distribution</div>',
                    unsafe_allow_html=True)
        fig = px.histogram(df, x="if_score", color="if_anomaly",
                           color_discrete_map={-1:"#E63946", 1:"#00B4D8"},
                           nbins=60, opacity=0.8,
                           labels={"if_score": "Anomaly Score", "if_anomaly": "Status"})
        fig.update_layout(template="plotly_dark", plot_bgcolor="#161B22",
                          paper_bgcolor="#161B22", height=340, margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">PCA — Normal vs Anomaly</div>',
                    unsafe_allow_html=True)
        pca_anom = pd.DataFrame(pca_2d, columns=["PC1","PC2"])
        pca_anom["is_anomaly"] = df["if_anomaly"].map({1:"Normal",-1:"Anomaly"})
        pca_anom["score"]      = df["if_score"]
        pca_anom["dB"]         = df["avg_decibels"]
        fig = px.scatter(pca_anom, x="PC1", y="PC2", color="is_anomaly",
                         size="score", size_max=14,
                         color_discrete_map={"Normal":"#00B4D8","Anomaly":"#E63946"},
                         hover_data=["dB","score"], opacity=0.75)
        fig.update_layout(template="plotly_dark", plot_bgcolor="#161B22",
                          paper_bgcolor="#161B22", height=340, margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

    # Anomaly table
    st.markdown('<div class="section-header">Top Anomalous Readings</div>',
                unsafe_allow_html=True)
    anom_df = (df[df["if_anomaly"] == -1]
               .sort_values("if_score", ascending=False)
               [["zone_type","avg_decibels","traffic_density","industrial_activity",
                 "honking_frequency","hour","if_score","latitude","longitude"]]
               .head(25)
               .reset_index(drop=True))
    st.dataframe(anom_df.style.background_gradient(cmap="RdYlGn_r", subset=["avg_decibels","if_score"]),
                 use_container_width=True, height=340)

    # Anomaly by zone
    st.markdown('<div class="section-header">Anomaly Count by Zone Type</div>',
                unsafe_allow_html=True)
    zone_anom = (df[df["if_anomaly"]==-1]["zone_type"]
                 .value_counts().reset_index())
    zone_anom.columns = ["Zone", "Count"]
    fig = px.bar(zone_anom, x="Zone", y="Count", color="Count",
                 color_continuous_scale="Reds", text_auto=True)
    fig.update_layout(template="plotly_dark", plot_bgcolor="#161B22",
                      paper_bgcolor="#161B22", height=300, margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE: Analytics & Trends
# ═════════════════════════════════════════════════════════════════════════════

elif page == "📈 Analytics & Trends":
    st.markdown('<div class="main-title">📈 Analytics & Noise Trends</div>',
                unsafe_allow_html=True)

    # Correlation heatmap
    st.markdown('<div class="section-header">Feature Correlation Heatmap</div>',
                unsafe_allow_html=True)
    num_cols = ["avg_decibels","traffic_density","vehicle_count","honking_frequency",
                "industrial_activity","construction_activity","population_density",
                "temperature_c","humidity_pct","wind_speed_kmh"]
    num_cols = [c for c in num_cols if c in df.columns]
    corr = df[num_cols].corr()
    fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    text_auto=".2f", aspect="auto")
    fig.update_layout(template="plotly_dark", paper_bgcolor="#161B22",
                      height=480, margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        # Hour × Day heatmap
        st.markdown('<div class="section-header">Noise Heatmap: Hour × Day</div>',
                    unsafe_allow_html=True)
        pivot = (df.groupby(["hour","day_of_week"])["avg_decibels"]
                   .mean().unstack(fill_value=0))
        pivot.columns = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        fig = px.imshow(pivot, color_continuous_scale="RdYlGn_r",
                        labels={"x":"Day","y":"Hour","color":"dB"}, aspect="auto")
        fig.update_layout(template="plotly_dark", paper_bgcolor="#161B22",
                          height=380, margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Weather vs noise
        st.markdown('<div class="section-header">Noise vs Weather Condition</div>',
                    unsafe_allow_html=True)
        fig = px.box(df, x="weather_condition", y="avg_decibels",
                     color="weather_condition",
                     color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(template="plotly_dark", plot_bgcolor="#161B22",
                          paper_bgcolor="#161B22", height=380,
                          showlegend=False, margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

    # Feature importance
    st.markdown('<div class="section-header">Feature Importance (PCA Loadings)</div>',
                unsafe_allow_html=True)
    dr = DimensionalityReducer()
    dr.fit_pca(X)
    fi_df = dr.feature_importance_pca(prep.feature_cols)
    fi_top = fi_df.head(14).reset_index()
    fi_top.columns = ["Feature", "PC1", "PC2", "Total"]
    fig = px.bar(fi_top.sort_values("Total"), x="Total", y="Feature",
                 orientation="h", color="Total", color_continuous_scale="Blues",
                 text_auto=".3f")
    fig.update_layout(template="plotly_dark", plot_bgcolor="#161B22",
                      paper_bgcolor="#161B22", height=420, margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)

    # Elbow + silhouette
    st.markdown('<div class="section-header">Optimal Cluster Selection</div>',
                unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        fig = px.line(x=elbow["k"], y=elbow["inertia"], markers=True,
                      labels={"x":"k","y":"Inertia"},
                      color_discrete_sequence=["#00B4D8"])
        fig.update_layout(template="plotly_dark", plot_bgcolor="#161B22",
                          paper_bgcolor="#161B22", height=280,
                          title="Elbow Curve", margin=dict(t=40))
        st.plotly_chart(fig, use_container_width=True)
    with col4:
        best_k = elbow["k"][int(np.argmax(elbow["silhouette"]))]
        fig = px.line(x=elbow["k"], y=elbow["silhouette"], markers=True,
                      labels={"x":"k","y":"Silhouette Score"},
                      color_discrete_sequence=["#2DC653"])
        fig.add_vline(x=best_k, line_dash="dash", line_color="#E63946",
                      annotation_text=f"Best k={best_k}")
        fig.update_layout(template="plotly_dark", plot_bgcolor="#161B22",
                          paper_bgcolor="#161B22", height=280,
                          title="Silhouette Score", margin=dict(t=40))
        st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE: Real-Time Simulation
# ═════════════════════════════════════════════════════════════════════════════

elif page == "⚡ Real-Time Simulation":
    st.markdown('<div class="main-title">⚡ Real-Time Noise Simulation</div>',
                unsafe_allow_html=True)

    sim = RealTimeSimulator(anomaly_rate=0.08)

    st.markdown("### 🎛️ Manual Area Simulation")
    col1, col2, col3 = st.columns(3)
    with col1:
        sim_zone  = st.selectbox("Zone", list(sim.ZONES.keys())
                                 if hasattr(sim, "ZONES") else
                                 ["Industrial","Commercial_CBD","Residential",
                                  "Transport_Hub","Green_Zone","Suburban"])
    with col2:
        sim_hour  = st.slider("Hour of Day", 0, 23, 12)
    with col3:
        force_anom = st.checkbox("Force Anomaly", value=False)

    if st.button("🔊 Generate Reading", type="primary", use_container_width=True):
        record = sim.generate_single(hour=sim_hour, zone=sim_zone,
                                     force_anomaly=force_anom)
        alert  = get_alert_level(record["avg_decibels"],
                                  1.0 if record["simulated_anomaly"] else 0.2)
        sev    = classify_noise_severity(record["avg_decibels"])

        st.markdown(f"""
        <div style='background:#161B22;border:2px solid {alert["color"]};
                    border-radius:12px;padding:20px;margin:12px 0'>
          <h3 style='color:{alert["color"]};margin:0'>{alert["icon"]} {alert["level"]} — {sev} Noise</h3>
          <hr style='border-color:#30363D'>
          <div style='display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-top:12px'>
            <div><div style='color:#8B949E;font-size:.8rem'>Decibels</div>
                 <div style='color:#58A6FF;font-size:1.6rem;font-weight:700'>{record["avg_decibels"]:.1f} dB</div></div>
            <div><div style='color:#8B949E;font-size:.8rem'>Traffic</div>
                 <div style='color:#C9D1D9;font-size:1.2rem'>{record["traffic_density"]:.0f}</div></div>
            <div><div style='color:#8B949E;font-size:.8rem'>Honking Freq</div>
                 <div style='color:#C9D1D9;font-size:1.2rem'>{record["honking_frequency"]:.1f}</div></div>
            <div><div style='color:#8B949E;font-size:.8rem'>Weather</div>
                 <div style='color:#C9D1D9;font-size:1.2rem'>{record["weather_condition"]}</div></div>
          </div>
          <div style='margin-top:12px;color:#8B949E;font-size:.85rem'>
            📍 ({record["latitude"]:.4f}, {record["longitude"]:.4f}) &nbsp;|&nbsp;
            🏙️ {record["zone_type"]} &nbsp;|&nbsp;
            ⏰ {record["hour"]:02d}:00 &nbsp;|&nbsp;
            🎬 Action: <b style='color:{alert["color"]}'>{alert["action"]}</b>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Streaming simulation ──────────────────────────────────────────────────
    st.markdown("### 🔴 Live Streaming Simulation")
    st.info("Click **Start Stream** to simulate live incoming sensor data (runs for 30 readings).")

    if st.button("▶ Start Stream", use_container_width=True):
        chart_placeholder = st.empty()
        alert_placeholder = st.empty()
        stream_data: list = []

        progress = st.progress(0)
        for i in range(30):
            record = sim.generate_single()
            stream_data.append(record)
            s_df  = pd.DataFrame(stream_data)
            alert = get_alert_level(record["avg_decibels"],
                                     1.0 if record["simulated_anomaly"] else 0.2)

            # Live chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(len(s_df))), y=s_df["avg_decibels"],
                mode="lines+markers",
                line=dict(color="#00B4D8", width=2.5),
                marker=dict(
                    color=["#E63946" if a else "#00B4D8"
                           for a in s_df["simulated_anomaly"]],
                    size=9,
                ),
                name="Noise (dB)"
            ))
            fig.add_hline(y=70, line_dash="dash", line_color="#FFB703",
                          annotation_text="Alert 70dB")
            fig.add_hline(y=85, line_dash="dash", line_color="#E63946",
                          annotation_text="Danger 85dB")
            fig.update_layout(
                template="plotly_dark", plot_bgcolor="#161B22",
                paper_bgcolor="#161B22", height=320,
                title=f"Live Noise Stream — {len(s_df)} readings",
                xaxis_title="Reading #", yaxis_title="dB",
                margin=dict(t=50), showlegend=False,
            )
            chart_placeholder.plotly_chart(fig, use_container_width=True)

            alert_html = (
                f'<div class="alert-{alert["level"].lower()}">'
                f'{alert["icon"]} <b>{record["zone_type"]}</b>: '
                f'{record["avg_decibels"]:.1f} dB — {alert["level"]} — {alert["action"]}'
                f'</div>'
            )
            alert_placeholder.markdown(alert_html, unsafe_allow_html=True)
            progress.progress((i+1)/30)
            time.sleep(0.15)

        st.success("✅ Stream complete. Red dots = anomalous readings.")


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE: Model Comparison
# ═════════════════════════════════════════════════════════════════════════════

elif page == "🤖 Model Comparison":
    st.markdown('<div class="main-title">🤖 Model Comparison</div>', unsafe_allow_html=True)

    # Evaluate clustering
    km_metrics  = evaluate_clustering(X, df["kmeans_cluster"].values,  "K-Means")
    dbs_metrics = evaluate_clustering(X, df["dbscan_cluster"].values,  "DBSCAN")

    # Anomaly metrics (using ground truth label)
    true_bin = df["is_anomaly_label"].values if "is_anomaly_label" in df.columns else np.zeros(len(df))
    if_pred  = (df["if_anomaly"] == -1).astype(int).values
    if_score = df["if_score"].values
    if_metrics = evaluate_anomaly_detector(if_score, if_pred, true_bin, "Isolation Forest")

    table = build_comparison_table(
        [km_metrics, dbs_metrics],
        [if_metrics]
    )

    st.markdown('<div class="section-header">Metric Comparison Table</div>',
                unsafe_allow_html=True)
    st.dataframe(table.style.format(na_rep="—").background_gradient(
        cmap="Greens", subset=["Silhouette ↑","CH Index ↑","ROC-AUC ↑","F1 Score ↑"],
        axis=0
    ), use_container_width=True, height=200)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Silhouette Score</div>',
                    unsafe_allow_html=True)
        sil_vals  = [km_metrics.get("silhouette",0), dbs_metrics.get("silhouette",0)]
        sil_names = ["K-Means","DBSCAN"]
        fig = px.bar(x=sil_names, y=sil_vals, color=sil_vals,
                     color_continuous_scale="Greens", text_auto=".3f",
                     labels={"x":"Model","y":"Silhouette Score"})
        fig.update_layout(template="plotly_dark", plot_bgcolor="#161B22",
                          paper_bgcolor="#161B22", height=300, margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">DBI Score (lower = better)</div>',
                    unsafe_allow_html=True)
        dbi_vals = [km_metrics.get("dbi",0), dbs_metrics.get("dbi",0)]
        fig = px.bar(x=sil_names, y=dbi_vals, color=dbi_vals,
                     color_continuous_scale="Reds_r", text_auto=".3f",
                     labels={"x":"Model","y":"Davies-Bouldin Index"})
        fig.update_layout(template="plotly_dark", plot_bgcolor="#161B22",
                          paper_bgcolor="#161B22", height=300, margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

    # Anomaly metrics
    st.markdown('<div class="section-header">Anomaly Detection Metrics</div>',
                unsafe_allow_html=True)
    anom_met = {k:v for k,v in if_metrics.items() if isinstance(v,(int,float)) and v is not None}
    anom_df  = pd.DataFrame([{"Metric":k,"Value":v} for k,v in anom_met.items()])
    fig = px.bar(anom_df, x="Metric", y="Value", color="Value",
                 color_continuous_scale="Blues", text_auto=".3f")
    fig.update_layout(template="plotly_dark", plot_bgcolor="#161B22",
                      paper_bgcolor="#161B22", height=300, margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE: Reports & Insights
# ═════════════════════════════════════════════════════════════════════════════

elif page == "📋 Reports & Insights":
    st.markdown('<div class="main-title">📋 Smart City Insights & Report</div>',
                unsafe_allow_html=True)

    # Auto-generated insights
    st.markdown('<div class="section-header">🧠 AI-Generated Insights</div>',
                unsafe_allow_html=True)
    avg_db   = df["avg_decibels"].mean()
    n_anom   = int((df["if_anomaly"] == -1).sum())
    worst_zn = df.groupby("zone_type")["avg_decibels"].mean().idxmax()
    best_zn  = df.groupby("zone_type")["avg_decibels"].mean().idxmin()
    peak_hr  = df.groupby("hour")["avg_decibels"].mean().idxmax()

    insights = [
        ("🔊", "City Average Noise",
         f"{avg_db:.1f} dB — {'exceeds WHO limit of 70 dB' if avg_db>70 else 'within acceptable range'}"),
        ("🏭", "Noisiest Zone",
         f"{worst_zn} — highest average noise, priority intervention needed"),
        ("🌿", "Quietest Zone",
         f"{best_zn} — maintain green-zone policies"),
        ("⏰", "Peak Noise Hour",
         f"Hour {peak_hr:02d}:00 — deploy traffic management during this window"),
        ("🚨", "Anomaly Events",
         f"{n_anom} anomalous readings detected ({n_anom/len(df)*100:.1f}% of all data)"),
        ("📈", "High-Noise Events",
         f"{(df['avg_decibels']>80).sum()} readings above 80 dB — immediate action zones"),
    ]
    cols = st.columns(2)
    for i, (icon, title, desc) in enumerate(insights):
        with cols[i % 2]:
            st.markdown(f"""
            <div style='background:#161B22;border:1px solid #30363D;border-radius:10px;
                        padding:14px;margin-bottom:10px'>
              <div style='font-size:1.6rem'>{icon}</div>
              <div style='color:#58A6FF;font-weight:700;font-size:.95rem'>{title}</div>
              <div style='color:#C9D1D9;font-size:.85rem;margin-top:4px'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # Recommendations
    st.markdown('<div class="section-header">📌 Authority Recommendations</div>',
                unsafe_allow_html=True)
    recs = ca.generate_recommendations()
    for rec in recs:
        st.markdown(f'<div class="rec-item">{rec}</div>', unsafe_allow_html=True)

    # Export
    st.markdown('<div class="section-header">📥 Export Report Data</div>',
                unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        csv = df.to_csv(index=False)
        st.download_button("📄 Download Full Dataset", csv,
                           "smart_city_noise_full.csv", "text/csv",
                           use_container_width=True)
    with col2:
        anom_csv = df[df["if_anomaly"]==-1].to_csv(index=False)
        st.download_button("🚨 Download Anomalies", anom_csv,
                           "anomalies.csv", "text/csv",
                           use_container_width=True)
    with col3:
        summ_csv = summary_df.to_csv(index=False)
        st.download_button("📊 Download Cluster Summary", summ_csv,
                           "cluster_summary.csv", "text/csv",
                           use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align:center;color:#8B949E;font-size:.82rem;padding:12px'>
      🏙️ Smart City Noise Pollution Identification · Unsupervised ML Project<br>
      Stack: Python · Sklearn · TensorFlow · Streamlit · Folium · Plotly · Pandas<br>
      Models: K-Means · DBSCAN · Isolation Forest · PCA · t-SNE
    </div>
    """, unsafe_allow_html=True)
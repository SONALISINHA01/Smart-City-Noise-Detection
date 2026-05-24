"""
================================================================================
  Smart City Noise Mapping Module
  Folium heatmaps · Cluster marker maps · Anomaly overlays · Plotly 3D
================================================================================
"""

import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap, MarkerCluster, MiniMap
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

SAVE_DIR = Path("reports/maps")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ── Colour palette for cluster markers ───────────────────────────────────────
CLUSTER_COLORS = [
    "blue", "red", "green", "purple", "orange",
    "darkred", "lightred", "beige", "darkblue", "darkgreen",
]

SEVERITY_COLOR = {
    "Very Low":  "#2DC653",
    "Low":       "#57CC99",
    "Moderate":  "#FFB703",
    "High":      "#F77F00",
    "Extreme":   "#E63946",
}


# ─────────────────────────────────────────────────────────────────────────────
#  1. City Noise Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def create_noise_heatmap(df: pd.DataFrame,
                          save_path: str = None) -> folium.Map:
    """Full-city noise heatmap using avg_decibels as weight."""
    center = [df["latitude"].mean(), df["longitude"].mean()]
    m = folium.Map(location=center, zoom_start=13,
                   tiles="CartoDB dark_matter")

    # Normalise weight to 0-1
    weights = ((df["avg_decibels"] - df["avg_decibels"].min()) /
               (df["avg_decibels"].max() - df["avg_decibels"].min() + 1e-9))

    heat_data = list(zip(df["latitude"], df["longitude"], weights))
    HeatMap(
        heat_data,
        radius=18,
        blur=20,
        max_zoom=16,
        gradient={0.0: "blue", 0.35: "cyan", 0.55: "lime",
                  0.75: "orange", 0.90: "red", 1.0: "darkred"},
    ).add_to(m)

    MiniMap(toggle_display=True).add_to(m)
    folium.LayerControl().add_to(m)

    _add_legend(m, "Noise Heatmap", {
        "Low (< 55 dB)":      "#2DC653",
        "Moderate (55–70 dB)":"#FFB703",
        "High (70–85 dB)":    "#F77F00",
        "Extreme (> 85 dB)":  "#E63946",
    })

    path = save_path or str(SAVE_DIR / "noise_heatmap.html")
    m.save(path)
    print(f"  🗺️  Heatmap saved → {path}")
    return m


# ─────────────────────────────────────────────────────────────────────────────
#  2. Cluster Marker Map
# ─────────────────────────────────────────────────────────────────────────────

def create_cluster_map(df: pd.DataFrame,
                        cluster_col: str = "kmeans_cluster",
                        zone_map: dict = None,
                        save_path: str = None) -> folium.Map:
    center = [df["latitude"].mean(), df["longitude"].mean()]
    m = folium.Map(location=center, zoom_start=13,
                   tiles="CartoDB dark_matter")

    cluster_group = MarkerCluster(name="Noise Zones").add_to(m)
    unique_clusters = sorted(df[cluster_col].unique())

    for _, row in df.iterrows():
        cid   = row[cluster_col]
        color = CLUSTER_COLORS[int(cid) % len(CLUSTER_COLORS)] if cid != -1 else "black"
        zone  = zone_map.get(cid, f"Cluster {cid}") if zone_map else f"Cluster {cid}"

        popup_html = f"""
        <div style='font-family:monospace;font-size:12px;min-width:180px'>
          <b style='color:#00B4D8'>{zone}</b><br>
          🔊 dB: <b>{row['avg_decibels']:.1f}</b><br>
          🚗 Traffic: {row.get('traffic_density',0):.0f}<br>
          🏭 Industrial: {row.get('industrial_activity',0):.0f}<br>
          ⏰ Hour: {row.get('hour',0):02d}:00<br>
          📍 ({row['latitude']:.4f}, {row['longitude']:.4f})
        </div>"""

        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=220),
            tooltip=f"{zone} | {row['avg_decibels']:.1f} dB",
        ).add_to(cluster_group)

    MiniMap(toggle_display=True).add_to(m)
    folium.LayerControl().add_to(m)
    path = save_path or str(SAVE_DIR / "cluster_map.html")
    m.save(path)
    print(f"  🗺️  Cluster map saved → {path}")
    return m


# ─────────────────────────────────────────────────────────────────────────────
#  3. Anomaly Hotspot Map
# ─────────────────────────────────────────────────────────────────────────────

def create_anomaly_map(df: pd.DataFrame,
                        anomaly_col: str = "if_anomaly",
                        score_col:   str = "if_score",
                        save_path: str = None) -> folium.Map:
    center = [df["latitude"].mean(), df["longitude"].mean()]
    m = folium.Map(location=center, zoom_start=13,
                   tiles="CartoDB dark_matter")

    # Normal points as subtle heatmap
    normal = df[df[anomaly_col] != -1] if anomaly_col in df.columns else df
    if not normal.empty:
        HeatMap(
            list(zip(normal["latitude"], normal["longitude"],
                     [0.2] * len(normal))),
            radius=12, blur=15, max_zoom=16,
            gradient={0.0: "blue", 1.0: "cyan"},
        ).add_to(m)

    # Anomaly markers
    anom = df[df[anomaly_col] == -1] if anomaly_col in df.columns else pd.DataFrame()
    for _, row in anom.iterrows():
        score = row.get(score_col, 0.9)
        radius = 8 + score * 12
        popup_html = f"""
        <div style='font-family:monospace;font-size:12px'>
          <b style='color:#E63946'>⚠️ ANOMALY DETECTED</b><br>
          🔊 dB: <b style='color:red'>{row['avg_decibels']:.1f}</b><br>
          🚗 Traffic: {row.get('traffic_density',0):.0f}<br>
          📊 Score: {score:.3f}<br>
          ⏰ {row.get('hour',0):02d}:00
        </div>"""
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=radius,
            color="#E63946",
            fill=True,
            fill_color="#FF6B6B",
            fill_opacity=0.75,
            weight=2,
            popup=folium.Popup(popup_html, max_width=200),
            tooltip=f"⚠️ Anomaly | {row['avg_decibels']:.1f} dB",
        ).add_to(m)

    _add_legend(m, "Anomaly Hotspots", {
        "Normal":         "#00B4D8",
        "Anomaly (high)": "#E63946",
    })
    path = save_path or str(SAVE_DIR / "anomaly_map.html")
    m.save(path)
    print(f"  🗺️  Anomaly map saved → {path}")
    return m


# ─────────────────────────────────────────────────────────────────────────────
#  4. Combined Master Map
# ─────────────────────────────────────────────────────────────────────────────

def create_master_map(df: pd.DataFrame,
                       cluster_col: str = "kmeans_cluster",
                       anomaly_col: str = "if_anomaly",
                       zone_map: dict = None,
                       save_path: str = None) -> folium.Map:
    center = [df["latitude"].mean(), df["longitude"].mean()]
    m = folium.Map(location=center, zoom_start=13,
                   tiles="CartoDB dark_matter")

    # ── Layer 1: Heatmap ──────────────────────────────────────────────────────
    heat_layer = folium.FeatureGroup(name="🔥 Noise Heatmap")
    weights = ((df["avg_decibels"] - df["avg_decibels"].min()) /
               (df["avg_decibels"].max() - df["avg_decibels"].min() + 1e-9))
    HeatMap(
        list(zip(df["latitude"], df["longitude"], weights)),
        radius=16, blur=18,
        gradient={0.0:"blue", 0.4:"cyan", 0.65:"lime",
                  0.8:"orange", 1.0:"darkred"},
    ).add_to(heat_layer)
    heat_layer.add_to(m)

    # ── Layer 2: Cluster markers ──────────────────────────────────────────────
    cluster_layer = folium.FeatureGroup(name="📍 Zone Clusters")
    sample = df.sample(min(800, len(df)), random_state=42)
    for _, row in sample.iterrows():
        cid   = row.get(cluster_col, 0)
        color = CLUSTER_COLORS[int(cid) % len(CLUSTER_COLORS)]
        zone  = zone_map.get(cid, f"Cluster {cid}") if zone_map else f"Cluster {cid}"
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=4,
            color=color, fill=True, fill_color=color, fill_opacity=0.65,
            tooltip=f"{zone} | {row['avg_decibels']:.1f} dB",
        ).add_to(cluster_layer)
    cluster_layer.add_to(m)

    # ── Layer 3: Anomaly markers ──────────────────────────────────────────────
    if anomaly_col in df.columns:
        anom_layer = folium.FeatureGroup(name="⚠️ Anomaly Hotspots")
        anom = df[df[anomaly_col] == -1]
        for _, row in anom.iterrows():
            folium.Marker(
                location=[row["latitude"], row["longitude"]],
                icon=folium.Icon(color="red", icon="exclamation-sign", prefix="glyphicon"),
                tooltip=f"⚠️ {row['avg_decibels']:.1f} dB",
            ).add_to(anom_layer)
        anom_layer.add_to(m)

    MiniMap(toggle_display=True).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    _add_legend(m, "Smart City Noise Map", {
        "Heatmap Intensity": "#FF4500",
        "Zone Cluster":      "#00B4D8",
        "Anomaly Hotspot":   "#E63946",
    })

    path = save_path or str(SAVE_DIR / "master_map.html")
    m.save(path)
    print(f"  🗺️  Master map saved → {path}")
    return m


# ─────────────────────────────────────────────────────────────────────────────
#  Helper – HTML legend
# ─────────────────────────────────────────────────────────────────────────────

def _add_legend(m: folium.Map, title: str, items: dict):
    items_html = "".join(
        f"<li><span style='background:{color};width:14px;height:14px;"
        f"display:inline-block;border-radius:50%;margin-right:6px;'></span>{label}</li>"
        for label, color in items.items()
    )
    legend_html = f"""
    <div style='position:fixed;bottom:30px;left:30px;z-index:1000;
                background:#161B22;border:1px solid #30363D;
                padding:12px 16px;border-radius:8px;font-family:monospace;
                font-size:12px;color:#C9D1D9;min-width:180px'>
      <b style='color:#58A6FF'>{title}</b>
      <ul style='list-style:none;padding:0;margin:8px 0 0 0'>{items_html}</ul>
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))

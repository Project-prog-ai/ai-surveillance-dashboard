# ══════════════════════════════════════════════════════════
# AI POWERED SMART SURVEILLANCE DASHBOARD v5.0
# Responsive Overhaul — CSS Grid + clamp() + Media Queries
# ══════════════════════════════════════════════════════════
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN

# ─── Page Config ──────────────────────────────────────────
st.set_page_config(
    page_title="AI Smart Surveillance",
    page_icon="🚔",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ══════════════════════════════════════════════════════════
# RESPONSIVE CSS DESIGN SYSTEM
# ══════════════════════════════════════════════════════════
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Inter:wght@400;500;600;700&family=Share+Tech+Mono&display=swap" rel="stylesheet">
<style>
/* ── Reset & Base ─────────────────────────────────────── */
* { box-sizing: border-box; }
#MainMenu, header, footer, [data-testid="stToolbar"] { visibility: hidden; }
section[data-testid="stSidebar"] { display: none !important; }
.stApp { background: linear-gradient(135deg, #020812 0%, #071428 50%, #020812 100%); }
p, span, li { color: #c8e8ff; }
h1,h2,h3,h4,h5,h6 { color: #fff !important; }
[data-testid="stMarkdownContainer"] h1, [data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3, [data-testid="stMarkdownContainer"] h4 { color: #fff !important; }

/* ── Scrollbar ────────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #020812; }
::-webkit-scrollbar-thumb { background: rgba(0,212,255,0.4); border-radius: 2px; }

/* ══════════════════════════════════════════════════════════
   RESPONSIVE TYPOGRAPHY — clamp(min, preferred, max)
   ══════════════════════════════════════════════════════════ */
.header-title {
    font-family: 'Orbitron', monospace;
    font-size: clamp(10px, 2vw, 16px);
    font-weight: 900;
    color: #00d4ff;
    text-shadow: 0 0 12px rgba(0,212,255,0.7);
    letter-spacing: clamp(1px, 0.3vw, 3px);
    line-height: 1.3;
}
.header-sub {
    font-size: clamp(8px, 1.2vw, 11px);
    color: #7ab8e8;
    letter-spacing: clamp(1px, 0.3vw, 3px);
    font-weight: 600;
    margin-top: 2px;
}
.sec-head {
    font-family: 'Orbitron', monospace;
    font-size: clamp(9px, 1.1vw, 12px);
    font-weight: 700;
    letter-spacing: 2px;
    color: #00d4ff;
    padding: 6px 10px;
    border-left: 3px solid #00d4ff;
    background: rgba(0,20,55,0.6);
    border-radius: 0 6px 6px 0;
    margin: 6px 0;
}

/* ══════════════════════════════════════════════════════════
   KPI CARDS — CSS Grid, auto-fit for responsiveness
   ══════════════════════════════════════════════════════════ */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(110px, 1fr));
    gap: clamp(4px, 0.8vw, 10px);
    width: 100%;
}
.kpi-card {
    background: rgba(0,20,55,0.7);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 8px;
    padding: clamp(6px, 1vw, 12px);
    text-align: center;
    min-height: 70px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    transition: all 0.3s ease;
}
.kpi-card:hover {
    border-color: rgba(0,212,255,0.5);
    box-shadow: 0 0 20px rgba(0,212,255,0.12);
    transform: translateY(-1px);
}
.kpi-icon { font-size: clamp(14px, 1.8vw, 20px); margin-bottom: 2px; }
.kpi-label {
    font-size: clamp(7px, 0.8vw, 9px);
    font-weight: 700;
    letter-spacing: 1px;
    color: #7ab8e8;
    text-transform: uppercase;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 100%;
}
.kpi-value {
    font-family: 'Orbitron', monospace;
    font-size: clamp(14px, 1.8vw, 22px);
    font-weight: 900;
    line-height: 1.2;
}

/* ══════════════════════════════════════════════════════════
   SECTION CARDS — equal height with min-height
   ══════════════════════════════════════════════════════════ */
.info-card {
    background: rgba(0,15,40,0.85);
    border: 1px solid rgba(0,212,255,0.15);
    border-radius: 8px;
    padding: clamp(10px, 1.5vw, 16px);
    min-height: 240px;
    box-shadow: 0 0 15px rgba(0,212,255,0.06);
    height: 100%;
}
.info-card h4 {
    font-family: 'Orbitron', monospace;
    font-size: clamp(8px, 1vw, 11px);
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #00d4ff;
    margin-bottom: 10px;
    padding-bottom: 6px;
    border-bottom: 1px solid rgba(0,212,255,0.2);
}

/* ── Flow box (architecture diagram) ──────────────────── */
.flow-box {
    background: rgba(0,25,60,0.8);
    border: 1px solid rgba(0,212,255,0.3);
    border-radius: 6px;
    padding: clamp(4px, 0.6vw, 8px) clamp(6px, 0.8vw, 12px);
    font-size: clamp(8px, 0.9vw, 10px);
    color: #c8e8ff;
    text-align: center;
    font-weight: 600;
    font-family: 'Inter', sans-serif;
    white-space: nowrap;
}
.flow-arrow {
    color: #00d4ff;
    font-size: clamp(10px, 1.2vw, 14px);
    text-align: center;
    line-height: 1;
}

/* ── Detection row ────────────────────────────────────── */
.det-row {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: clamp(3px, 0.4vw, 5px) 0;
    font-size: clamp(9px, 1vw, 11px);
    color: #c8e8ff;
    font-family: 'Inter', sans-serif;
    border-bottom: 1px solid rgba(0,212,255,0.08);
}
.det-row:last-child { border-bottom: none; }
.det-check { color: #00ff88; font-size: 13px; }

/* ── Badges ───────────────────────────────────────────── */
.live-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(0,255,136,0.08);
    border: 1px solid rgba(0,255,136,0.3);
    border-radius: 4px;
    padding: 3px 10px;
    font-size: clamp(8px, 0.9vw, 10px);
    font-weight: 700;
    color: #00ff88;
    letter-spacing: 1px;
    text-transform: uppercase;
    font-family: 'Inter', sans-serif;
}
.live-dot {
    width: 6px; height: 6px;
    background: #00ff88;
    border-radius: 50%;
    box-shadow: 0 0 6px #00ff88;
    animation: blink 1.2s infinite;
}
@keyframes blink { 0%,100% { opacity:1; } 50% { opacity:0.2; } }

/* ── Alert rows ───────────────────────────────────────── */
.alert-row {
    display: flex;
    align-items: center;
    gap: clamp(4px, 0.6vw, 8px);
    padding: clamp(4px, 0.6vw, 7px) clamp(6px, 0.8vw, 10px);
    border-radius: 4px;
    border-left: 3px solid;
    margin-bottom: 3px;
    font-family: 'Inter', sans-serif;
}
.ar-id {
    font-size: clamp(9px, 1vw, 11px);
    font-weight: 700;
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.ar-badge {
    font-size: clamp(7px, 0.8vw, 9px);
    font-weight: 700;
    padding: 2px 6px;
    border-radius: 3px;
    letter-spacing: 0.5px;
}
.ar-time {
    font-size: clamp(8px, 0.9vw, 10px);
    color: #7ab8e8;
    font-family: 'Share Tech Mono', monospace;
}

/* ── KPI metric strip (overview tab) ──────────────────── */
.kpi-metric {
    background: rgba(0,20,55,0.8);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 6px;
    padding: clamp(6px, 1vw, 12px);
    text-align: center;
    min-height: 65px;
}
.kpi-metric .km-label {
    font-size: clamp(7px, 0.8vw, 9px);
    font-weight: 700;
    letter-spacing: 1px;
    color: #7ab8e8;
    text-transform: uppercase;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.kpi-metric .km-val {
    font-size: clamp(14px, 1.8vw, 22px);
    font-weight: 900;
    font-family: 'Orbitron', monospace;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    max-width: 100%;
}

/* ── Plotly transparent ───────────────────────────────── */
.js-plotly-plot, .plotly { background: transparent !important; }
.stPlotlyChart { border-radius: 6px; }

/* ── Streamlit overrides ──────────────────────────────── */
[data-testid="stSlider"] [data-testid="stWidgetLabel"] p { color: #7ab8e8 !important; font-size: 11px !important; }
[data-testid="stSlider"] * { color: #c8e8ff !important; }
[data-baseweb="select"] { background: rgba(0,15,40,0.9) !important; border-color: rgba(0,212,255,0.3) !important; }
[data-baseweb="select"] * { color: #c8e8ff !important; }
[data-testid="stWidgetLabel"] p { color: #7ab8e8 !important; font-size: 11px !important; font-weight: 600 !important; }
.stDownloadButton button {
    background: linear-gradient(135deg, #003580, #005ec2) !important;
    border: 1px solid rgba(0,212,255,0.4) !important;
    border-radius: 6px !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 11px !important;
    box-shadow: 0 0 12px rgba(0,212,255,0.2) !important;
}
[data-testid="stDataFrame"] * { color: #c8e8ff !important; font-size: 11px !important; }

/* ── Tab overrides ────────────────────────────────────── */
[data-testid="stTabs"] [data-baseweb="tab"] { color: #7ab8e8 !important; font-weight: 600 !important; background: transparent !important; font-size: clamp(9px, 1.1vw, 12px) !important; }
[data-testid="stTabs"] [aria-selected="true"][data-baseweb="tab"] { color: #00d4ff !important; font-weight: 700 !important; }
[data-testid="stTabs"] [data-baseweb="tab-list"] { background: rgba(0,15,40,0.7) !important; border: 1px solid rgba(0,212,255,0.15) !important; border-radius: 6px !important; }

/* ══════════════════════════════════════════════════════════
   EQUAL HEIGHT COLUMNS — Flexbox stretch
   ══════════════════════════════════════════════════════════ */
[data-testid="stHorizontalBlock"] {
    align-items: stretch !important;
    gap: clamp(0.25rem, 0.8vw, 0.75rem) !important;
}
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
    display: flex !important;
    flex-direction: column !important;
}
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"] > div:first-child {
    flex: 1 !important;
}
[data-testid="stVerticalBlock"] > div { margin-bottom: 0.2rem; }

/* ══════════════════════════════════════════════════════════
   VEHICLE TIMELINE — new feature styles
   ══════════════════════════════════════════════════════════ */
.profile-card {
    background: rgba(0,20,55,0.8);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 8px;
    padding: clamp(8px, 1.2vw, 14px);
    min-height: 80px;
}
.profile-label {
    font-size: clamp(7px, 0.8vw, 9px);
    color: #7ab8e8;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.profile-val {
    font-size: clamp(16px, 2vw, 26px);
    font-weight: 900;
    font-family: 'Orbitron', monospace;
}

/* ══════════════════════════════════════════════════════════
   MEDIA QUERIES — Mobile / Tablet / Desktop
   ══════════════════════════════════════════════════════════ */
@media (max-width: 640px) {
    .kpi-grid { grid-template-columns: repeat(2, 1fr); }
    .info-card { min-height: auto; }
    .header-title { text-align: center; }
}
@media (min-width: 641px) and (max-width: 1024px) {
    .kpi-grid { grid-template-columns: repeat(3, 1fr); }
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════
CLR = {"CRITICAL": "#ff1744", "HIGH": "#ff6b35", "MEDIUM": "#ffc107", "LOW": "#00ff88"}
PLT = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
           font=dict(color="#c8e8ff", family="Inter", size=11),
           legend=dict(font=dict(color="#c8e8ff"), bgcolor="rgba(0,0,0,0)"),
           margin=dict(t=30, b=5, l=5, r=5))
AX = dict(gridcolor="rgba(0,212,255,0.08)", color="#7ab8e8",
          tickfont=dict(color="#c8e8ff", size=10),
          linecolor="rgba(0,212,255,0.2)", zerolinecolor="rgba(0,212,255,0.1)")
now_str = datetime.now().strftime("%d %b %Y  %H:%M:%S IST")


def find_csv():
    import os
    search_dirs = [
        "",
        "/mount/src/ai-surveillance-dashboard/",
        "/content/",
        os.path.dirname(os.path.abspath(__file__)),
    ]
    for d in search_dirs:
        for n in ["surveillance_features.csv", "dashboard_dataset.csv", "data.csv"]:
            path = os.path.join(d, n) if d else n
            try:
                return pd.read_csv(path), path
            except:
                continue
    return None, None


# ══════════════════════════════════════════════════════════
# DATA LOAD / UPLOAD
# ══════════════════════════════════════════════════════════
_found, _fname = find_csv()
if _found is None:
    st.markdown("""<div style="display:flex;align-items:center;justify-content:center;height:100vh;
    background:#020812;flex-direction:column;gap:20px;">
    <div style="font-size:60px;">📂</div>
    <h2 style="color:#00d4ff;font-family:Orbitron;letter-spacing:2px;">DATASET REQUIRED</h2>
    <p style="color:#7ab8e8;font-size:14px;">Upload surveillance_features.csv to launch the dashboard</p>
    </div>""", unsafe_allow_html=True)
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up:
        with open("dashboard_dataset.csv", "wb") as f:
            f.write(up.read())
        st.success("✅ Uploaded! Refreshing...")
        st.rerun()
    st.stop()


@st.cache_data
def load_data():
    df, _ = find_csv()
    df.columns = df.columns.str.upper().str.strip()
    for col in df.select_dtypes(include="float64").columns:
        df[col] = df[col].astype("float32")
    for col in df.select_dtypes(include="int64").columns:
        df[col] = df[col].astype("int32")
    for c in ["DATETIME", "TIMESTAMP"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
            df["TIMESTAMP"] = df[c]
            break
    if "TIMESTAMP" in df.columns:
        df["HOUR"] = df["TIMESTAMP"].dt.hour
        df["DAY"] = df["TIMESTAMP"].dt.day_name()
        df["DATE"] = df["TIMESTAMP"].dt.date
    r = {"TOTAL_DISTANCE_KM": "TRIP_DISTANCE", "DURATION_MIN": "TRAVEL_TIME"}
    for o, n in r.items():
        if o in df.columns:
            df.rename(columns={o: n}, inplace=True)
    if "TRIP_ID" not in df.columns:
        df["TRIP_ID"] = df.get("VEHICLE_ID", pd.Series([f"VEH-{i:05d}" for i in range(len(df))]))
    df["RISK_LEVEL"] = df["RISK_LEVEL"].astype(str).str.upper().str.strip()
    if "PEAK_HOUR" not in df.columns and "HOUR" in df.columns:
        df["PEAK_HOUR"] = df["HOUR"].apply(
            lambda h: "Peak" if h in list(range(7, 10)) + list(range(17, 20)) else "Off-Peak")
    if "TRIP_TYPE" not in df.columns:
        df["TRIP_TYPE"] = df.apply(
            lambda r: "Commercial" if r.get("TRIP_DISTANCE", 10) > 20 else (
                "Short" if r.get("TRIP_DISTANCE", 10) < 5 else "Standard"), axis=1)
    fcols = [c for c in
             ["PARKING_ANOMALY", "SPEED_ANOMALY", "ROUTE_DEVIATION", "RESTRICTED_ZONE_ENTRY",
              "COORDINATED_MOVEMENT"] if c in df.columns]
    df["FLAG_COUNT"] = df[fcols].sum(axis=1) if fcols else 0
    df["SUSPICION_SCORE"] = (df["RISK_SCORE"].fillna(0) * 0.5 + df["FLAG_COUNT"] * 8 +
                             df.get("RZ_HIT_COUNT", pd.Series(0, index=df.index)).fillna(0) * 2).round(1)
    if "IF_LABEL" in df.columns:
        df["IF_RESULT"] = df["IF_LABEL"].map({1: "Normal", -1: "Anomaly"})
    return df


df = load_data()

# ══════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════
total = len(df)
crit = (df["RISK_LEVEL"] == "CRITICAL").sum()
high = (df["RISK_LEVEL"] == "HIGH").sum()
med = (df["RISK_LEVEL"] == "MEDIUM").sum()
low = (df["RISK_LEVEL"] == "LOW").sum()
avg_risk = round(df["RISK_SCORE"].mean(), 1)
max_risk = round(df["RISK_SCORE"].max(), 1)
avg_spd = round(df["AVG_SPEED_KMH"].mean(), 1)
max_spd = round(df["MAX_SPEED_KMH"].mean(), 1) if "MAX_SPEED_KMH" in df.columns else 0
active_alerts = crit + high
rz = int(df["RESTRICTED_ZONE_ENTRY"].sum()) if "RESTRICTED_ZONE_ENTRY" in df.columns else 0
sp = int(df["SPEED_ANOMALY"].sum()) if "SPEED_ANOMALY" in df.columns else 0
pk = int(df["PARKING_ANOMALY"].sum()) if "PARKING_ANOMALY" in df.columns else 0
rd = int(df["ROUTE_DEVIATION"].sum()) if "ROUTE_DEVIATION" in df.columns else 0
cm = int(df["COORDINATED_MOVEMENT"].sum()) if "COORDINATED_MOVEMENT" in df.columns else 0
circular = int((df["CIRCUITY_RATIO"] > 1.5).sum()) if "CIRCUITY_RATIO" in df.columns else 0
abnormal = int((df["PARKING_DURATION_MIN"] > 60).sum()) if "PARKING_DURATION_MIN" in df.columns else 0
recent_alerts = df.nlargest(6, "RISK_SCORE")[["TRIP_ID", "RISK_LEVEL", "RISK_SCORE"]].values.tolist()

# ══════════════════════════════════════════════════════════
# ① HEADER BAR — Responsive
# ══════════════════════════════════════════════════════════
h_left, h_center, h_right = st.columns([3, 6, 2])

with h_left:
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:12px;">
      <div style="width:42px;height:42px;background:rgba(0,212,255,0.1);border:2px solid rgba(0,212,255,0.5);
      border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:22px;
      box-shadow:0 0 15px rgba(0,212,255,0.3);flex-shrink:0;">🚔</div>
      <div>
        <div class="header-title">AI POWERED SMART SURVEILLANCE</div>
        <div class="header-sub">SUSPICIOUS VEHICLE DETECTION SYSTEM</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

with h_center:
    st.markdown(f"""
    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-icon">🚗</div>
        <div class="kpi-label">Total Vehicles</div>
        <div class="kpi-value" style="color:#00d4ff;">{total:,}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-icon">🟢</div>
        <div class="kpi-label">Low Risk</div>
        <div class="kpi-value" style="color:#00ff88;">{low:,}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-icon">⚠️</div>
        <div class="kpi-label">Medium Risk</div>
        <div class="kpi-value" style="color:#ffc107;">{med:,}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-icon">🔺</div>
        <div class="kpi-label">High Risk</div>
        <div class="kpi-value" style="color:#ff6b35;">{high:,}</div>
      </div>
      <div class="kpi-card" style="border-color:rgba(255,23,68,0.5);box-shadow:0 0 15px rgba(255,23,68,0.2);">
        <div class="kpi-icon">🚨</div>
        <div class="kpi-label">Active Alerts</div>
        <div class="kpi-value" style="color:#ff1744;">{active_alerts:,}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

with h_right:
    st.markdown(f"""
    <div style="text-align:right;">
      <div class="live-badge"><div class="live-dot"></div> DATASET MODE</div>
      <div style="font-size:clamp(8px,0.9vw,10px);color:#7ab8e8;margin-top:5px;font-family:'Share Tech Mono',monospace;">
        {total:,} TRIPS</div>
      <div style="font-size:clamp(7px,0.8vw,9px);color:#004080;margin-top:2px;">
        Mode: <span style='color:#00ff88;'>PROTOTYPE</span></div>
    </div>
    """, unsafe_allow_html=True)

# ── Header divider ────────────────────────────────────────
st.markdown('<div style="height:2px;background:linear-gradient(90deg,transparent,rgba(0,212,255,0.4),transparent);margin:4px 0 8px;"></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# ② MAP + ANALYTICS — 2-column layout (no nav column)
# ══════════════════════════════════════════════════════════
map_col, analytics_col = st.columns([6.5, 3.5])

# ── MAP ───────────────────────────────────────────────────
with map_col:
    map_sample = st.slider("Map Sample Size", 100, 800, 350, 50, key="mapsz",
                           help="Number of vehicles to show on map")

    @st.cache_data(max_entries=1)
    def build_map(n):
        np.random.seed(42)
        porto_lat, porto_lon = 41.1579, -8.6291
        lats = porto_lat + np.random.uniform(-0.05, 0.05, n)
        lons = porto_lon + np.random.uniform(-0.08, 0.08, n)
        risks = np.random.choice(["LOW", "MEDIUM", "HIGH"], size=n, p=[0.697, 0.282, 0.021])
        scores = np.random.uniform(0, 85, n)
        _CLR = {"HIGH": "#ff6b35", "MEDIUM": "#ffc107", "LOW": "#00ff88"}
        _SZ = {"HIGH": 10, "MEDIUM": 7, "LOW": 5}
        fig = go.Figure()
        for rl in ["LOW", "MEDIUM", "HIGH"]:
            mask = risks == rl
            if not mask.any():
                continue
            fig.add_trace(go.Scattermap(
                lat=lats[mask], lon=lons[mask], mode="markers", name=rl,
                marker=dict(size=[_SZ[rl]] * int(mask.sum()), color=_CLR[rl], opacity=0.85),
                text=[f"Risk: {rl} | Score: {s:.0f}" for s in scores[mask]],
                hoverinfo="text"
            ))
        fig.update_layout(
            map=dict(style="open-street-map",
                     center=dict(lat=porto_lat, lon=porto_lon), zoom=12),
            margin=dict(t=0, b=0, l=0, r=0), height=460,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(bgcolor="rgba(0,15,40,.85)", bordercolor="rgba(0,212,255,.3)",
                        borderwidth=1, font=dict(color="#c8e8ff", size=10))
        )
        return fig

    st.plotly_chart(build_map(map_sample), use_container_width=True)

    # Map status bar
    st.markdown(f"""
    <div style="display:flex;gap:clamp(8px,1.5vw,20px);padding:6px 10px;background:rgba(0,15,40,0.7);
    border:1px solid rgba(0,212,255,0.15);border-radius:6px;flex-wrap:wrap;align-items:center;">
      <span style="font-size:clamp(8px,0.9vw,10px);color:#7ab8e8;font-family:'Share Tech Mono',monospace;">
        📍 Porto, Portugal</span>
      <span style="font-size:clamp(8px,0.9vw,10px);color:#00ff88;">●  {low:,} LOW</span>
      <span style="font-size:clamp(8px,0.9vw,10px);color:#ffc107;">●  {med:,} MED</span>
      <span style="font-size:clamp(8px,0.9vw,10px);color:#ff6b35;">●  {high:,} HIGH</span>
      <span style="font-size:clamp(8px,0.9vw,10px);color:#7ab8e8;margin-left:auto;">
        🕐 {now_str}</span>
    </div>
    """, unsafe_allow_html=True)

# ── ANALYTICS PANEL ───────────────────────────────────────
with analytics_col:
    # Detection Engine
    st.markdown('<div class="sec-head">🔍 AI DETECTION ENGINE</div>', unsafe_allow_html=True)

    detections = [
        ("🅿️", "Long Parking Detected", abnormal),
        ("🚫", "Restricted Zone Entry", rz),
        ("🔁", "Circular Movement", circular),
        ("🛑", "Abnormal Stop Pattern", int((df.get("SPEED_SPIKES", pd.Series(0, index=df.index)) > 3).sum())),
        ("↗️", "Route Deviation", rd),
        ("⚡", "Speed Anomaly", sp),
        ("🤝", "Coordinated Movement", cm),
    ]
    for icon, label, count in detections:
        color = "#ff1744" if count > 1000 else "#ffc107" if count > 100 else "#00ff88"
        st.markdown(f"""<div class="det-row">
          <span class="det-check">✅</span>
          <span style="flex:1;">{icon} {label}</span>
          <span style="font-weight:700;color:{color};font-family:'Share Tech Mono',monospace;">{count:,}</span>
        </div>""", unsafe_allow_html=True)

    # Risk Distribution donut
    st.markdown('<div class="sec-head" style="margin-top:8px;">📊 RISK DISTRIBUTION</div>', unsafe_allow_html=True)
    risk_counts = [low, med, high]
    risk_labels = ["LOW", "MEDIUM", "HIGH"]
    risk_colors = ["#00ff88", "#ffc107", "#ff6b35"]
    fig_donut = go.Figure(go.Pie(
        labels=risk_labels, values=risk_counts,
        marker=dict(colors=risk_colors), hole=0.5,
        textinfo="percent", textfont=dict(color="white", size=10)
    ))
    _plt_donut = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#c8e8ff", size=10),
                      margin=dict(t=5, b=5, l=5, r=5),
                      legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#c8e8ff", size=9)))
    fig_donut.update_layout(**_plt_donut, height=160,
                            annotations=[dict(text=f"<b>{total:,}</b>", x=0.5, y=0.5,
                                              font=dict(size=14, color="#00d4ff", family="Orbitron"),
                                              showarrow=False)])
    st.plotly_chart(fig_donut, use_container_width=True)

    # Recent Alerts
    st.markdown('<div class="sec-head" style="margin-top:4px;">🚨 RECENT ALERTS</div>', unsafe_allow_html=True)
    for trip_id, risk, score in recent_alerts:
        color = CLR.get(risk, "#7ab8e8")
        bg = f"{color}15"
        st.markdown(f"""
        <div class="alert-row" style="background:{bg};border-left-color:{color};">
          <div class="ar-id" style="color:{color};">{str(trip_id)[:12]}</div>
          <div class="ar-badge" style="background:{color}22;color:{color};border:1px solid {color}44;">{risk}</div>
          <div class="ar-time">{score:.0f}/100</div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# ③ ANALYTICS TABS — Full width, 5 tabs
# ══════════════════════════════════════════════════════════
st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)
st.markdown('<div class="sec-head">📊 ADVANCED ANALYTICS ENGINE</div>', unsafe_allow_html=True)

tab_overview, tab_temporal, tab_ml, tab_timeline, tab_explorer = st.tabs([
    "📊  Overview", "⏰  Temporal Analysis", "🤖  ML Engine",
    "🔄  Vehicle Timeline", "🔍  Vehicle Explorer"
])

# ── TAB 1: OVERVIEW — 2×2 grid ───────────────────────────
with tab_overview:
    ov_r1c1, ov_r1c2 = st.columns(2)
    with ov_r1c1:
        st.markdown('<div style="font-size:clamp(9px,1vw,11px);font-weight:700;color:#7ab8e8;letter-spacing:1px;margin-bottom:4px;">RISK LEVEL DISTRIBUTION</div>', unsafe_allow_html=True)
        rc = df["RISK_LEVEL"].value_counts().reset_index()
        rc.columns = ["Risk Level", "Vehicles"]
        fig = px.bar(rc, x="Risk Level", y="Vehicles", color="Risk Level", text="Vehicles",
                     color_discrete_map=CLR)
        fig.update_traces(textposition="outside", textfont=dict(color="white", size=11))
        fig.update_layout(**PLT, height=280, showlegend=False, xaxis={**AX}, yaxis={**AX})
        st.plotly_chart(fig, use_container_width=True)

    with ov_r1c2:
        st.markdown('<div style="font-size:clamp(9px,1vw,11px);font-weight:700;color:#7ab8e8;letter-spacing:1px;margin-bottom:4px;">SPEED VS RISK SCORE</div>', unsafe_allow_html=True)
        samp = df.sample(min(1500, len(df)), random_state=42)
        fig = px.scatter(samp, x="AVG_SPEED_KMH", y="RISK_SCORE", color="RISK_LEVEL",
                         color_discrete_map=CLR, opacity=0.6,
                         labels={"AVG_SPEED_KMH": "Avg Speed (km/h)", "RISK_SCORE": "Risk Score",
                                 "RISK_LEVEL": "Risk Level"})
        fig.update_layout(**PLT, height=280, xaxis={**AX}, yaxis={**AX})
        st.plotly_chart(fig, use_container_width=True)

    ov_r2c1, ov_r2c2 = st.columns(2)
    with ov_r2c1:
        st.markdown('<div style="font-size:clamp(9px,1vw,11px);font-weight:700;color:#7ab8e8;letter-spacing:1px;margin-bottom:4px;">TRIP TYPE BREAKDOWN</div>', unsafe_allow_html=True)
        tt = df["TRIP_TYPE"].value_counts().reset_index()
        tt.columns = ["Type", "Count"]
        fig = px.bar(tt, x="Type", y="Count", color="Count", text="Count",
                     color_continuous_scale="Blues")
        fig.update_traces(textposition="outside", textfont=dict(color="white"))
        fig.update_layout(**PLT, height=280, showlegend=False, xaxis={**AX}, yaxis={**AX})
        st.plotly_chart(fig, use_container_width=True)

    with ov_r2c2:
        st.markdown('<div style="font-size:clamp(9px,1vw,11px);font-weight:700;color:#7ab8e8;letter-spacing:1px;margin-bottom:4px;">ANOMALY FLAGS OVERVIEW</div>', unsafe_allow_html=True)
        fdf = pd.DataFrame(
            {"Flag": ["🅿️ Parking", "⚡ Speed", "🔁 Route Dev", "🚫 Zone", "🤝 Coordinated"],
             "Count": [pk, sp, rd, rz, cm]}).sort_values("Count", ascending=True)
        fig = px.bar(fdf, y="Flag", x="Count", orientation="h", color="Count", text="Count",
                     color_continuous_scale="Reds")
        fig.update_traces(textposition="outside", textfont=dict(color="white"))
        fig.update_layout(**PLT, height=280, showlegend=False, xaxis={**AX}, yaxis={**AX})
        st.plotly_chart(fig, use_container_width=True)

    # KPI metric strip — 2 rows of 3
    mk_r1 = st.columns(3)
    for col, lbl, val, color in [
        (mk_r1[0], "Avg Risk Score", avg_risk, "#ffc107"),
        (mk_r1[1], "Max Risk", max_risk, "#ff1744"),
        (mk_r1[2], "Avg Speed km/h", avg_spd, "#00d4ff")]:
        _v = f"{val:,.1f}" if isinstance(val, float) else f"{val:,}"
        col.markdown(f"""<div class="kpi-metric">
        <div class="km-label">{lbl}</div>
        <div class="km-val" style="color:{color};">{_v}</div></div>""", unsafe_allow_html=True)

    mk_r2 = st.columns(3)
    for col, lbl, val, color in [
        (mk_r2[0], "Avg Max Speed", max_spd, "#ff6b35"),
        (mk_r2[1], "Zone Violations", rz, "#ff1744"),
        (mk_r2[2], "Speed Anomalies", sp, "#ff6b35")]:
        _v = f"{val:,.1f}" if isinstance(val, float) else f"{val:,}"
        col.markdown(f"""<div class="kpi-metric">
        <div class="km-label">{lbl}</div>
        <div class="km-val" style="color:{color};">{_v}</div></div>""", unsafe_allow_html=True)

# ── TAB 2: TEMPORAL ───────────────────────────────────────
with tab_temporal:
    t1, t2 = st.columns(2)
    with t1:
        hourly = df.groupby(["HOUR", "RISK_LEVEL"]).size().reset_index(name="Trips")
        hourly.rename(columns={"RISK_LEVEL": "Risk Level"}, inplace=True)
        fig = px.bar(hourly, x="HOUR", y="Trips", color="Risk Level", color_discrete_map=CLR,
                     labels={"HOUR": "Hour (24h)"})
        fig.update_layout(**PLT, height=300,
                          title=dict(text="Trips by Hour", font=dict(color="white", size=12)),
                          xaxis={**AX}, yaxis={**AX})
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        hr = df.groupby("HOUR")["RISK_SCORE"].mean().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hr["HOUR"], y=hr["RISK_SCORE"], mode="lines+markers",
                                 line=dict(color="#ff6b35", width=2),
                                 marker=dict(color="#ff1744", size=6),
                                 fill="tozeroy", fillcolor="rgba(255,107,53,0.1)", name="Avg Risk"))
        fig.update_layout(**PLT, height=300,
                          title=dict(text="Avg Risk by Hour", font=dict(color="white", size=12)),
                          xaxis={**AX, "dtick": 2}, yaxis={**AX})
        st.plotly_chart(fig, use_container_width=True)

    # Heatmap full width
    day_hour = df.groupby(["DAY", "HOUR"])["RISK_SCORE"].mean().unstack(fill_value=0)
    do = [d for d in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
          if d in day_hour.index]
    fig = px.imshow(day_hour.reindex(do), color_continuous_scale="RdYlGn_r", aspect="auto",
                    labels=dict(x="Hour", y="Day", color="Avg Risk"))
    fig.update_layout(**PLT, height=300,
                      title=dict(text="Day × Hour Risk Heatmap", font=dict(color="white", size=12)))
    st.plotly_chart(fig, use_container_width=True)

# ── TAB 3: ML ENGINE ─────────────────────────────────────
with tab_ml:
    if "IF_LABEL" not in df.columns or "CLUSTER" not in df.columns:
        st.warning("⚠️ Pre-computed ML columns missing. Please re-upload the enriched CSV.")
    else:
        st.markdown('<div class="sec-head">🌲 Isolation Forest Results</div>', unsafe_allow_html=True)
        anom_cnt = (df["IF_LABEL"] == -1).sum()
        norm_cnt = (df["IF_LABEL"] == 1).sum()

        mc1, mc2, mc3 = st.columns([1, 2, 2])
        with mc1:
            st.markdown(f"""<div class="profile-card">
            <div class="profile-label">ANOMALIES DETECTED</div>
            <div class="profile-val" style="color:#ff1744;">{anom_cnt:,}</div>
            <div style="color:#7ab8e8;font-size:clamp(8px,0.9vw,10px);margin-top:4px;">
              Normal: <b style="color:#00ff88;">{norm_cnt:,}</b></div>
            <div style="color:#7ab8e8;font-size:clamp(8px,0.9vw,10px);">
              Contamination: <b style="color:#00d4ff;">5%</b> | Trees: <b style="color:#00d4ff;">50</b></div>
            </div>""", unsafe_allow_html=True)

        with mc2:
            fig = px.scatter(df.sample(min(2000, len(df)), random_state=42),
                             x="AVG_SPEED_KMH", y="RISK_SCORE", color="IF_RESULT",
                             color_discrete_map={"Normal": "#00ff88", "Anomaly": "#ff1744"}, opacity=0.7,
                             labels={"AVG_SPEED_KMH": "Speed (km/h)", "RISK_SCORE": "Risk Score",
                                     "IF_RESULT": "Result"})
            fig.update_layout(**PLT, height=280,
                              title=dict(text="Anomaly Scatter", font=dict(color="white", size=12)),
                              xaxis={**AX}, yaxis={**AX})
            st.plotly_chart(fig, use_container_width=True)

        with mc3:
            rc2 = df["IF_RESULT"].value_counts().reset_index()
            rc2.columns = ["Result", "Count"]
            fig2 = px.pie(rc2, names="Result", values="Count",
                          color="Result",
                          color_discrete_map={"Normal": "#00ff88", "Anomaly": "#ff1744"},
                          hole=0.5)
            fig2.update_layout(**PLT, height=280,
                               title=dict(text="IF Distribution", font=dict(color="white", size=12)))
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<div class="sec-head" style="margin-top:8px;">🧠 DBSCAN Clustering Results</div>',
                    unsafe_allow_html=True)
        cl_cnt = len([c for c in df["CLUSTER"].unique() if c != "Noise"])
        dc1, dc2 = st.columns(2)
        with dc1:
            fig = px.scatter(df.sample(min(3000, len(df)), random_state=42),
                             x="AVG_SPEED_KMH", y="RISK_SCORE", color="CLUSTER", opacity=0.7,
                             labels={"AVG_SPEED_KMH": "Speed", "RISK_SCORE": "Risk"})
            fig.update_layout(**PLT, height=280,
                              title=dict(text=f"DBSCAN — {cl_cnt} Clusters Found",
                                         font=dict(color="white", size=12)),
                              xaxis={**AX}, yaxis={**AX})
            st.plotly_chart(fig, use_container_width=True)
        with dc2:
            cl_dist = df["CLUSTER"].value_counts().reset_index()
            cl_dist.columns = ["CLUSTER", "Count"]
            fig = px.bar(cl_dist.head(10), x="CLUSTER", y="Count", color="Count",
                         color_continuous_scale="Blues", text="Count")
            fig.update_traces(textposition="outside", textfont=dict(color="white"))
            fig.update_layout(**PLT, height=280, showlegend=False,
                              title=dict(text="Cluster Distribution",
                                         font=dict(color="white", size=12)),
                              xaxis={**AX}, yaxis={**AX})
            st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════
# ★ TAB 4: VEHICLE BEHAVIOUR TIMELINE — NEW EXCLUSIVE FEATURE
# ══════════════════════════════════════════════════════════
with tab_timeline:
    st.markdown('<div class="sec-head">🔄 VEHICLE BEHAVIOUR TIMELINE</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:clamp(9px,1vw,11px);color:#7ab8e8;margin-bottom:8px;">Track a vehicle\'s risk escalation across all trips — identify reconnaissance patterns and progressive suspicious behaviour.</div>', unsafe_allow_html=True)

    # Vehicle selector
    veh_ids = sorted(df["VEHICLE_ID"].unique().astype(str).tolist()) if "VEHICLE_ID" in df.columns else []
    tl_c1, tl_c2 = st.columns([2, 4])

    with tl_c1:
        selected_veh = st.selectbox("Select Vehicle ID", veh_ids, index=0, key="veh_timeline")
        veh_df = df[df["VEHICLE_ID"].astype(str) == selected_veh].copy()

        if "TIMESTAMP" in veh_df.columns:
            veh_df = veh_df.sort_values("TIMESTAMP")

        total_trips = len(veh_df)
        high_trips = (veh_df["RISK_LEVEL"] == "HIGH").sum()
        avg_r = round(veh_df["RISK_SCORE"].mean(), 1)
        total_km = round(veh_df["TRIP_DISTANCE"].sum(), 1) if "TRIP_DISTANCE" in veh_df.columns else 0
        total_flags = int(veh_df["FLAG_COUNT"].sum())

        # Most common flag
        flag_cols = [c for c in ["SPEED_ANOMALY", "ROUTE_DEVIATION", "RESTRICTED_ZONE_ENTRY",
                                 "PARKING_ANOMALY", "COORDINATED_MOVEMENT"] if c in veh_df.columns]
        if flag_cols:
            flag_sums = {c.replace("_", " ").title(): int(veh_df[c].sum()) for c in flag_cols}
            top_flag = max(flag_sums, key=flag_sums.get) if flag_sums else "None"
            top_flag_count = max(flag_sums.values()) if flag_sums else 0
        else:
            top_flag, top_flag_count = "N/A", 0

        # Peak hour
        if "HOUR" in veh_df.columns and len(veh_df) > 0:
            peak_hr = int(veh_df["HOUR"].mode().iloc[0])
            peak_hr_str = f"{peak_hr:02d}:00"
        else:
            peak_hr_str = "N/A"

        # Behaviour profile cards
        st.markdown(f"""<div class="profile-card" style="margin-bottom:6px;">
          <div class="profile-label">TOTAL TRIPS</div>
          <div class="profile-val" style="color:#00d4ff;">{total_trips}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""<div class="profile-card" style="margin-bottom:6px;">
          <div class="profile-label">HIGH RISK TRIPS</div>
          <div class="profile-val" style="color:#ff1744;">{high_trips}</div>
          <div style="font-size:clamp(8px,0.9vw,10px);color:#7ab8e8;">
            {(high_trips/max(total_trips,1)*100):.1f}% of all trips</div>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""<div class="profile-card" style="margin-bottom:6px;">
          <div class="profile-label">AVG RISK SCORE</div>
          <div class="profile-val" style="color:{'#ff6b35' if avg_r > 40 else '#ffc107' if avg_r > 25 else '#00ff88'};">{avg_r}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""<div class="profile-card" style="margin-bottom:6px;">
          <div class="profile-label">TOTAL DISTANCE</div>
          <div class="profile-val" style="color:#00d4ff;">{total_km:,.1f} <span style="font-size:10px;">km</span></div>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""<div class="profile-card" style="margin-bottom:6px;">
          <div class="profile-label">TOTAL FLAGS TRIGGERED</div>
          <div class="profile-val" style="color:#ff6b35;">{total_flags}</div>
          <div style="font-size:clamp(8px,0.9vw,10px);color:#7ab8e8;">
            Top: {top_flag} ({top_flag_count})</div>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""<div class="profile-card">
          <div class="profile-label">PEAK ACTIVITY HOUR</div>
          <div class="profile-val" style="color:#ffc107;">{peak_hr_str}</div>
        </div>""", unsafe_allow_html=True)

    with tl_c2:
        # Timeline chart — Risk Score evolution across trips
        if "TIMESTAMP" in veh_df.columns and len(veh_df) > 0:
            veh_plot = veh_df.reset_index(drop=True)
            veh_plot["TRIP_NUM"] = range(1, len(veh_plot) + 1)

            # Color by risk level
            color_map = {"LOW": "#00ff88", "MEDIUM": "#ffc107", "HIGH": "#ff1744"}
            veh_plot["COLOR"] = veh_plot["RISK_LEVEL"].map(color_map).fillna("#7ab8e8")

            fig_tl = go.Figure()

            # Background line connecting all points
            fig_tl.add_trace(go.Scatter(
                x=veh_plot["TRIP_NUM"], y=veh_plot["RISK_SCORE"],
                mode="lines",
                line=dict(color="rgba(0,212,255,0.3)", width=1.5),
                showlegend=False, hoverinfo="skip"
            ))

            # Colored markers for each risk level
            for rl, color in color_map.items():
                mask = veh_plot["RISK_LEVEL"] == rl
                if not mask.any():
                    continue
                subset = veh_plot[mask]

                # Build hover text
                hover_texts = []
                for _, row in subset.iterrows():
                    flags = []
                    if row.get("SPEED_ANOMALY", 0): flags.append("⚡Speed")
                    if row.get("ROUTE_DEVIATION", 0): flags.append("↗️Route")
                    if row.get("RESTRICTED_ZONE_ENTRY", 0): flags.append("🚫Zone")
                    if row.get("PARKING_ANOMALY", 0): flags.append("🅿️Parking")
                    if row.get("COORDINATED_MOVEMENT", 0): flags.append("🤝Coord")
                    flag_str = ", ".join(flags) if flags else "None"
                    ts = row.get("TIMESTAMP", "")
                    ts_str = ts.strftime("%b %d %H:%M") if pd.notna(ts) else "N/A"
                    hover_texts.append(
                        f"Trip #{row['TRIP_NUM']}<br>"
                        f"Time: {ts_str}<br>"
                        f"Risk: {row['RISK_SCORE']:.1f} ({rl})<br>"
                        f"Speed: {row['AVG_SPEED_KMH']:.1f} km/h<br>"
                        f"Flags: {flag_str}"
                    )

                fig_tl.add_trace(go.Scatter(
                    x=subset["TRIP_NUM"], y=subset["RISK_SCORE"],
                    mode="markers", name=rl,
                    marker=dict(color=color, size=8 if rl == "HIGH" else 6,
                                line=dict(color="white", width=1) if rl == "HIGH" else dict(width=0)),
                    text=hover_texts, hoverinfo="text"
                ))

            # Add threshold lines
            fig_tl.add_hline(y=50, line_dash="dash", line_color="rgba(255,107,53,0.3)",
                             annotation_text="HIGH threshold", annotation_font_color="#ff6b35",
                             annotation_font_size=9)
            fig_tl.add_hline(y=30, line_dash="dot", line_color="rgba(255,193,7,0.3)",
                             annotation_text="MEDIUM threshold", annotation_font_color="#ffc107",
                             annotation_font_size=9)

            fig_tl.update_layout(
                **PLT, height=400,
                title=dict(text=f"Risk Escalation — Vehicle {selected_veh}",
                           font=dict(color="white", size=13)),
                xaxis=dict(**AX, title=dict(text="Trip Number", font=dict(color="#7ab8e8", size=10))),
                yaxis=dict(**AX, title=dict(text="Risk Score", font=dict(color="#7ab8e8", size=10)),
                           range=[0, max(veh_plot["RISK_SCORE"].max() * 1.15, 60)])
            )
            st.plotly_chart(fig_tl, use_container_width=True)

            # Flag breakdown bar for this vehicle
            if flag_cols:
                flag_data = pd.DataFrame({
                    "Flag": [c.replace("_", " ").title() for c in flag_cols],
                    "Count": [int(veh_df[c].sum()) for c in flag_cols]
                }).sort_values("Count", ascending=True)
                fig_flags = px.bar(flag_data, y="Flag", x="Count", orientation="h",
                                   color="Count", color_continuous_scale="Reds", text="Count")
                fig_flags.update_traces(textposition="outside", textfont=dict(color="white"))
                fig_flags.update_layout(**PLT, height=200, showlegend=False,
                                        title=dict(text="Flag Breakdown for This Vehicle",
                                                   font=dict(color="white", size=11)),
                                        xaxis={**AX}, yaxis={**AX})
                st.plotly_chart(fig_flags, use_container_width=True)
        else:
            st.info("📊 No timestamp data available for timeline visualization.")

# ── TAB 5: VEHICLE EXPLORER ──────────────────────────────
with tab_explorer:
    ex1, ex2, ex3 = st.columns([3, 1, 1])
    sq = ex1.text_input("🔍 Search Vehicle / Trip ID", "")
    sr2 = ex2.selectbox("Risk", ["All"] + sorted(df["RISK_LEVEL"].unique().tolist()))
    rn = ex3.selectbox("Show", ["25", "50", "100"], index=0)
    exp = df.copy()
    if sq:
        exp = exp[exp["TRIP_ID"].astype(str).str.contains(sq, case=False, na=False)]
    if sr2 != "All":
        exp = exp[exp["RISK_LEVEL"] == sr2]
    if "IF_RESULT" in exp.columns:
        exp = exp[exp["IF_RESULT"].isin(["Normal", "Anomaly"])]
    exp = exp.sort_values("RISK_SCORE", ascending=False)
    dcols = [c for c in
             ["TRIP_ID", "RISK_LEVEL", "RISK_SCORE", "SUSPICION_SCORE", "AVG_SPEED_KMH",
              "TRIP_DISTANCE", "TRAVEL_TIME", "FLAG_COUNT", "RESTRICTED_ZONE_ENTRY", "SPEED_ANOMALY",
              "ROUTE_DEVIATION", "PARKING_ANOMALY", "CIRCUITY_RATIO", "IF_RESULT", "ANOMALY_SCORE",
              "CLUSTER"] if c in exp.columns]
    disp = exp[dcols].head(int(rn)).copy()
    disp.columns = [x.replace("_", " ").title() for x in disp.columns]
    st.dataframe(disp, use_container_width=True, height=400)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    st.download_button("📥 Download Results", exp.to_csv(index=False), f"results_{ts}.csv", "text/csv")

# ══════════════════════════════════════════════════════════
# ④ BOTTOM STRIP — Equal-height info cards
# ══════════════════════════════════════════════════════════
st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)
b1, b2, b3, b4 = st.columns(4)

with b1:
    st.markdown("""<div class="info-card">
    <h4>⚡ SYSTEM ARCHITECTURE</h4>
    <div style="display:flex;align-items:center;gap:clamp(3px,0.5vw,6px);flex-wrap:wrap;">
      <div class="flow-box">Vehicle<br>Simulator</div>
      <div class="flow-arrow">→</div>
      <div class="flow-box">🔗 MQTT<br>Broker</div>
      <div class="flow-arrow">→</div>
      <div class="flow-box" style="border-color:#00ff88;">FastAPI<br>Backend</div>
      <div class="flow-arrow">→</div>
      <div class="flow-box">AI Engine<br>(IF+DBSCAN)</div>
      <div class="flow-arrow">→</div>
      <div class="flow-box" style="border-color:#ffc107;">PostgreSQL<br>DB</div>
    </div>
    <div style="display:flex;align-items:center;gap:clamp(3px,0.5vw,6px);flex-wrap:wrap;margin-top:8px;">
      <div class="flow-box" style="border-color:#00d4ff;">Streamlit<br>Dashboard</div>
      <div class="flow-arrow">←</div>
      <div class="flow-box">Rule-Based<br>Analytics</div>
      <div class="flow-arrow">←</div>
      <div class="flow-box">Risk<br>Scoring</div>
      <div class="flow-arrow">←</div>
      <div class="flow-box">Zone<br>Monitor</div>
    </div>
    </div>""", unsafe_allow_html=True)

with b2:
    st.markdown("""<div class="info-card">
    <h4>🔄 WORKING FLOW</h4>
    <div style="font-size:clamp(8px,0.9vw,10px);color:#c8e8ff;font-family:'Inter',sans-serif;line-height:1.8;">
      <div><span style="color:#00d4ff;font-weight:700;">1.</span> Vehicle Data Generated / Received Every 3–5 sec</div>
      <div><span style="color:#00d4ff;font-weight:700;">2.</span> Data Streamed via MQTT</div>
      <div><span style="color:#00d4ff;font-weight:700;">3.</span> AI Engine Analyses Behaviour</div>
      <div><span style="color:#ffc107;font-weight:700;">4.</span> Risk Score Calculated</div>
      <div><span style="color:#ff6b35;font-weight:700;">5.</span> If Suspicious → Alert Generated</div>
      <div><span style="color:#ff1744;font-weight:700;">6.</span> Dashboard &amp; Map Updated in Real-Time</div>
    </div>
    </div>""", unsafe_allow_html=True)

with b3:
    status_items = "".join([f'<div><span style="color:#00ff88;">●</span> {m}</div>' for m in
                            ["Data Pipeline", "Rule Engine", "Isolation Forest", "DBSCAN Clustering",
                             "Zone Monitor", "Risk Scoring", "Alert Engine"]])
    st.markdown(f"""<div class="info-card">
    <h4>📡 SYSTEM STATUS</h4>
    <div style="font-size:clamp(8px,0.9vw,10px);color:#c8e8ff;font-family:'Inter',sans-serif;line-height:2.0;">
    {status_items}
    </div>
    </div>""", unsafe_allow_html=True)

with b4:
    st.markdown("""<div class="info-card">
    <h4>🛠️ TECHNOLOGIES USED</h4>
    <div style="font-size:clamp(8px,0.9vw,10px);color:#c8e8ff;font-family:'Inter',sans-serif;line-height:1.9;">
      <div><span style="color:#3776ab;">🐍</span> Python (Streamlit / FastAPI)</div>
      <div><span style="color:#ff6b35;">📡</span> MQTT (IoT Streaming)</div>
      <div><span style="color:#41b883;">🗺️</span> Plotly Scattermap</div>
      <div><span style="color:#ff4154;">📊</span> Plotly (Interactive Charts)</div>
      <div><span style="color:#f7931e;">🤖</span> Scikit-Learn (IF + DBSCAN)</div>
      <div><span style="color:#336791;">🗄️</span> PostgreSQL / CSV</div>
      <div><span style="color:#00d4ff;">☁️</span> Cloudflare Tunnel</div>
    </div>
    </div>""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────
st.markdown(f"""<div style="text-align:center;padding:8px;background:rgba(0,10,28,0.8);
border-top:1px solid rgba(0,212,255,0.15);margin-top:8px;">
<span style="font-family:'Orbitron',monospace;font-size:clamp(7px,0.8vw,9px);color:#00d4ff;letter-spacing:2px;">
AI POWERED SMART SURVEILLANCE v5.0 &nbsp;|&nbsp; </span>
<span style="font-size:clamp(7px,0.8vw,9px);color:#7ab8e8;">Python · Streamlit · Plotly · Scikit-Learn</span>
</div>""", unsafe_allow_html=True)

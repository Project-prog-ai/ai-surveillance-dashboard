import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN

st.set_page_config(page_title="AI Smart Surveillance", page_icon="🚔", layout="wide", initial_sidebar_state="collapsed")

CSS = """
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Inter:wght@400;500;600;700&family=Share+Tech+Mono&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box}
#MainMenu,header,footer,[data-testid="stToolbar"]{visibility:hidden}
section[data-testid="stSidebar"]{display:none!important}
.stApp{background:linear-gradient(135deg,#020812 0%,#071428 50%,#020812 100%)}
p,span,li{color:#c8e8ff}
h1,h2,h3,h4,h5,h6{color:#fff!important}
[data-testid="stMarkdownContainer"] h1,[data-testid="stMarkdownContainer"] h2,[data-testid="stMarkdownContainer"] h3,[data-testid="stMarkdownContainer"] h4{color:#fff!important}
::-webkit-scrollbar{width:4px;height:4px}
::-webkit-scrollbar-track{background:#020812}
::-webkit-scrollbar-thumb{background:rgba(0,212,255,0.4);border-radius:2px}
.header-title{font-family:Orbitron,monospace;font-size:clamp(8px,1vw,13px);font-weight:900;color:#00d4ff;text-shadow:0 0 12px rgba(0,212,255,0.7);letter-spacing:clamp(0.5px,0.15vw,1.5px);line-height:1.15}
.header-sub{font-size:clamp(6px,0.65vw,8px);color:#7ab8e8;letter-spacing:clamp(0.5px,0.15vw,1.5px);font-weight:600;margin-top:1px}
.sec-head{font-family:Orbitron,monospace;font-size:clamp(9px,1.1vw,12px);font-weight:700;letter-spacing:2px;color:#00d4ff;padding:4px 8px;border-left:3px solid #00d4ff;background:rgba(0,20,55,0.6);border-radius:0 6px 6px 0;margin:4px 0}
.kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));gap:clamp(4px,0.8vw,10px);width:100%}
.kpi-card{background:rgba(0,20,55,0.7);border:1px solid rgba(0,212,255,0.2);border-radius:8px;padding:clamp(4px,0.7vw,8px);text-align:center;min-height:55px;display:flex;flex-direction:column;justify-content:center;align-items:center;transition:all 0.3s ease}
.kpi-card:hover{border-color:rgba(0,212,255,0.5);box-shadow:0 0 20px rgba(0,212,255,0.12);transform:translateY(-1px)}
.kpi-icon{font-size:clamp(10px,1.2vw,15px);margin-bottom:1px}
.kpi-label{font-size:clamp(7px,0.8vw,9px);font-weight:700;letter-spacing:1px;color:#7ab8e8;text-transform:uppercase;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:100%}
.kpi-value{font-family:Orbitron,monospace;font-size:clamp(12px,1.4vw,18px);font-weight:900;line-height:1.2}
.info-card{background:rgba(0,15,40,0.85);border:1px solid rgba(0,212,255,0.15);border-radius:8px;padding:clamp(10px,1.5vw,16px);min-height:240px;box-shadow:0 0 15px rgba(0,212,255,0.06);height:100%;display:flex;flex-direction:column}
.info-card h4{font-family:Orbitron,monospace;font-size:clamp(8px,1vw,11px);font-weight:700;letter-spacing:2px;text-transform:uppercase;color:#00d4ff;margin-bottom:10px;padding-bottom:6px;border-bottom:1px solid rgba(0,212,255,0.2)}
.flow-box{background:rgba(0,25,60,0.8);border:1px solid rgba(0,212,255,0.3);border-radius:6px;padding:clamp(4px,0.6vw,8px) clamp(6px,0.8vw,12px);font-size:clamp(8px,0.9vw,10px);color:#c8e8ff;text-align:center;font-weight:600;font-family:Inter,sans-serif;white-space:nowrap}
.flow-arrow{color:#00d4ff;font-size:clamp(10px,1.2vw,14px);text-align:center;line-height:1}
.det-row{display:flex;align-items:center;gap:6px;padding:clamp(3px,0.4vw,5px) 0;font-size:clamp(8px,0.9vw,10px);color:#c8e8ff;font-family:Inter,sans-serif;border-bottom:1px solid rgba(0,212,255,0.08);width:100%;overflow:hidden}
.det-check{color:#00ff88;font-size:13px}
.live-badge{display:inline-flex;align-items:center;gap:4px;background:rgba(0,255,136,0.08);border:1px solid rgba(0,255,136,0.3);border-radius:4px;padding:2px 8px;font-size:clamp(7px,0.7vw,9px);font-weight:700;color:#00ff88;letter-spacing:1px;text-transform:uppercase;font-family:Inter,sans-serif}
.live-dot{width:6px;height:6px;background:#00ff88;border-radius:50%;box-shadow:0 0 6px #00ff88;animation:blink 1.2s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0.2}}
.alert-row{display:flex;align-items:center;gap:clamp(3px,0.5vw,6px);padding:clamp(3px,0.4vw,5px) clamp(5px,0.6vw,8px);border-radius:4px;border-left:3px solid;margin-bottom:2px;font-family:Inter,sans-serif;width:100%}
.ar-id{font-size:clamp(9px,1vw,11px);font-weight:700;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.ar-badge{font-size:clamp(7px,0.8vw,9px);font-weight:700;padding:2px 6px;border-radius:3px;letter-spacing:0.5px}
.ar-time{font-size:clamp(8px,0.9vw,10px);color:#7ab8e8;font-family:Share Tech Mono,monospace}
.kpi-metric{background:rgba(0,20,55,0.8);border:1px solid rgba(0,212,255,0.2);border-radius:6px;padding:clamp(6px,1vw,12px);text-align:center;min-height:65px;height:100%;display:flex;flex-direction:column;justify-content:center;align-items:center}
.kpi-metric .km-label{font-size:clamp(7px,0.8vw,9px);font-weight:700;letter-spacing:1px;color:#7ab8e8;text-transform:uppercase;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.kpi-metric .km-val{font-size:clamp(14px,1.8vw,22px);font-weight:900;font-family:Orbitron,monospace;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:100%}
.profile-card{background:rgba(0,20,55,0.8);border:1px solid rgba(0,212,255,0.2);border-radius:8px;padding:clamp(8px,1.2vw,14px);min-height:80px;height:100%;display:flex;flex-direction:column;justify-content:center}
.profile-label{font-size:clamp(7px,0.8vw,9px);color:#7ab8e8;font-weight:700;letter-spacing:1px;text-transform:uppercase}
.profile-val{font-size:clamp(16px,2vw,26px);font-weight:900;font-family:Orbitron,monospace}
.footer-strip{text-align:center;padding:10px 16px;background:linear-gradient(90deg,rgba(0,10,28,0.9),rgba(2,20,50,0.95),rgba(0,10,28,0.9));border-top:2px solid rgba(0,212,255,0.25);margin-top:10px;box-shadow:0 -2px 20px rgba(0,212,255,0.08)}
@keyframes fadeIn{from{opacity:0;transform:translateX(-10px)}to{opacity:1;transform:translateX(0)}}
.js-plotly-plot,.plotly{background:transparent!important}
.stPlotlyChart{border-radius:6px}
[data-testid="stSlider"] [data-testid="stWidgetLabel"] p{color:#7ab8e8!important;font-size:11px!important}
[data-testid="stSlider"] *{color:#c8e8ff!important}
[data-baseweb="select"]{background:rgba(0,15,40,0.9)!important;border-color:rgba(0,212,255,0.3)!important}[data-testid="stTextInput"] input{background:rgba(0,15,40,0.9)!important;color:#e8f4ff!important;border-color:rgba(0,212,255,0.3)!important}
[data-baseweb="select"] *{color:#c8e8ff!important}
[data-testid="stWidgetLabel"] p{color:#7ab8e8!important;font-size:11px!important;font-weight:600!important}
.stDownloadButton button{background:linear-gradient(135deg,#003580,#005ec2)!important;border:1px solid rgba(0,212,255,0.4)!important;border-radius:6px!important;color:white!important;font-weight:700!important;font-size:11px!important;box-shadow:0 0 12px rgba(0,212,255,0.2)!important}
[data-testid="stDataFrame"] *{color:#e8f4ff!important;font-size:11px!important}[data-testid="stDataFrame"] table{background:rgba(0,15,40,0.9)!important}[data-testid="stDataFrame"] th{background:rgba(0,40,90,0.95)!important;color:#00d4ff!important;font-weight:700!important;font-size:11px!important;border-bottom:2px solid rgba(0,212,255,0.3)!important}[data-testid="stDataFrame"] td{background:rgba(0,12,35,0.9)!important;border-bottom:1px solid rgba(0,212,255,0.1)!important}[data-testid="stDataFrame"] tr:hover td{background:rgba(0,30,70,0.9)!important}[data-testid="stDataFrame"] [data-testid="glideDataEditor"]{background:rgba(0,12,35,0.95)!important}[data-testid="stDataFrame"] .gdg-header{background:rgba(0,40,90,0.95)!important;color:#00d4ff!important}
[data-testid="stTabs"] [data-baseweb="tab"]{color:#7ab8e8!important;font-weight:600!important;background:transparent!important;font-size:clamp(9px,1.1vw,12px)!important}
[data-testid="stTabs"] [aria-selected="true"][data-baseweb="tab"]{color:#00d4ff!important;font-weight:700!important}
[data-testid="stTabs"] [data-baseweb="tab-list"]{background:rgba(0,15,40,0.7)!important;border:1px solid rgba(0,212,255,0.15)!important;border-radius:6px!important}
[data-testid="stHorizontalBlock"]{align-items:stretch!important;gap:clamp(0.25rem,0.8vw,0.75rem)!important;display:flex!important}
[data-testid="stHorizontalBlock"]>[data-testid="stColumn"]{display:flex!important;flex-direction:column!important}
/* flex stretch handled by align-items:stretch on parent */
[data-testid="stVerticalBlock"]>div{margin-bottom:0.15rem}
.block-container{padding-top:1rem!important;padding-bottom:0!important}
[data-testid="stDeckGlJsonChart"]{height:500px!important;min-height:500px!important}
[data-testid="stDeckGlJsonChart"] iframe{height:500px!important;min-height:500px!important}
iframe[title="streamlit_autorefresh.st_autorefresh"]{height:0!important;min-height:0!important;border:none!important;overflow:hidden!important;position:absolute!important;visibility:hidden!important}
[data-testid="stVerticalBlock"]>div:has(iframe[title="streamlit_autorefresh.st_autorefresh"]){height:0!important;min-height:0!important;margin:0!important;padding:0!important;overflow:hidden!important}
@media(max-width:640px){.kpi-grid{grid-template-columns:repeat(2,1fr)}.info-card{min-height:auto!important}.header-title{text-align:center}.profile-card{min-height:auto}}
@media(min-width:641px) and (max-width:1024px){.kpi-grid{grid-template-columns:repeat(3,1fr)}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

CLR = {"CRITICAL":"#ff1744","HIGH":"#ff6b35","MEDIUM":"#ffc107","LOW":"#00ff88"}
PLT = dict(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
           font=dict(color="#c8e8ff",family="Inter",size=11),
           legend=dict(font=dict(color="#c8e8ff"),bgcolor="rgba(0,0,0,0)"),
           margin=dict(t=40,b=5,l=5,r=5))
AX = dict(gridcolor="rgba(0,212,255,0.08)",color="#7ab8e8",
          tickfont=dict(color="#c8e8ff",size=10),
          linecolor="rgba(0,212,255,0.2)",zerolinecolor="rgba(0,212,255,0.1)")
now_str = datetime.now().strftime("%d %b %Y  %H:%M:%S IST")

def find_csv():
    import os
    for d in ["","/mount/src/ai-surveillance-dashboard/","/content/",os.path.dirname(os.path.abspath(__file__))]:
        for n in ["surveillance_features.csv","dashboard_dataset.csv","data.csv"]:
            path = os.path.join(d,n) if d else n
            try: return pd.read_csv(path),path
            except Exception: continue
    return None,None

_found,_fname = find_csv()
if _found is None:
    st.markdown("""<div style="display:flex;align-items:center;justify-content:center;height:100vh;background:#020812;flex-direction:column;gap:20px;">
    <div style="font-size:60px;">&#128194;</div>
    <h2 style="color:#00d4ff;font-family:Orbitron;letter-spacing:2px;">DATASET REQUIRED</h2>
    <p style="color:#7ab8e8;font-size:14px;">Upload surveillance_features.csv to launch the dashboard</p>
    </div>""", unsafe_allow_html=True)
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up:
        with open("dashboard_dataset.csv","wb") as f: f.write(up.read())
        st.success("Uploaded! Refreshing..."); st.rerun()
    st.stop()

@st.cache_data
def load_data():
    df,_ = find_csv()
    df.columns = df.columns.str.upper().str.strip()
    for col in df.select_dtypes(include="float64").columns: df[col]=df[col].astype("float32")
    for col in df.select_dtypes(include="int64").columns:
        if col not in ["TRIP_ID"]: df[col]=df[col].astype("int32")
    for c in ["DATETIME","TIMESTAMP"]:
        if c in df.columns:
            df[c]=pd.to_datetime(df[c],errors="coerce"); df["TIMESTAMP"]=df[c]; break
    if "TIMESTAMP" in df.columns:
        df["HOUR"]=df["TIMESTAMP"].dt.hour; df["DAY"]=df["TIMESTAMP"].dt.day_name(); df["DATE"]=df["TIMESTAMP"].dt.date
    for o,n in {"TOTAL_DISTANCE_KM":"TRIP_DISTANCE","DURATION_MIN":"TRAVEL_TIME"}.items():
        if o in df.columns: df.rename(columns={o:n},inplace=True)
    if "TRIP_ID" in df.columns: df["TRIP_ID"]=df["TRIP_ID"].astype(str)
    if "TRIP_ID" not in df.columns:
        df["TRIP_ID"]=df.get("VEHICLE_ID",pd.Series([f"VEH-{i:05d}" for i in range(len(df))]))
    df["RISK_LEVEL"]=df["RISK_LEVEL"].astype(str).str.upper().str.strip()
    if "PEAK_HOUR" not in df.columns and "HOUR" in df.columns:
        df["PEAK_HOUR"]=df["HOUR"].apply(lambda h:"Peak" if h in list(range(7,10))+list(range(17,20)) else "Off-Peak")
    if "TRIP_TYPE" not in df.columns:
        df["TRIP_TYPE"]=df.apply(lambda r:"Commercial" if r.get("TRIP_DISTANCE",10)>20 else("Short" if r.get("TRIP_DISTANCE",10)<5 else "Standard"),axis=1)
    fcols=[c for c in ["PARKING_ANOMALY","SPEED_ANOMALY","ROUTE_DEVIATION","RESTRICTED_ZONE_ENTRY","COORDINATED_MOVEMENT"] if c in df.columns]
    df["FLAG_COUNT"]=df[fcols].sum(axis=1) if fcols else 0
    df["SUSPICION_SCORE"]=(df["RISK_SCORE"].fillna(0)*0.5+df["FLAG_COUNT"]*8+df.get("RZ_HIT_COUNT",pd.Series(0,index=df.index)).fillna(0)*2).round(1)
    if "IF_LABEL" in df.columns: df["IF_RESULT"]=df["IF_LABEL"].map({1:"Normal",-1:"Anomaly"})
    return df

df = load_data()

# ═══════════════════════════════════════════════════════════════
# LIVE SIMULATION ENGINE — Injects new dynamic trips in real-time
# ═══════════════════════════════════════════════════════════════
if "live_mode" not in st.session_state:
    st.session_state.live_mode = False
if "sim_trips" not in st.session_state:
    st.session_state.sim_trips = pd.DataFrame()
if "sim_counter" not in st.session_state:
    st.session_state.sim_counter = 0
if "sim_log" not in st.session_state:
    st.session_state.sim_log = []

def generate_live_trips(base_df, n=3):
    """Generate n new random vehicle trips based on real data distributions."""
    rng = np.random.default_rng()
    veh_ids = base_df["VEHICLE_ID"].unique() if "VEHICLE_ID" in base_df.columns else [f"SIM-{i}" for i in range(100)]
    new_rows = []
    for _ in range(n):
        vid = rng.choice(veh_ids)
        # Sample from real distributions with some randomness
        risk_score = float(rng.beta(2, 5) * 100)  # Skewed toward lower scores
        avg_speed = float(rng.normal(35, 15))
        avg_speed = max(5, min(120, avg_speed))
        max_speed = avg_speed + float(rng.uniform(10, 40))
        distance = float(rng.exponential(8)) + 1
        duration = distance / (avg_speed / 60) if avg_speed > 0 else 10
        # Determine risk level from score
        if risk_score >= 70: risk_level = "HIGH"
        elif risk_score >= 40: risk_level = "MEDIUM"
        else: risk_level = "LOW"
        # Random flags
        speed_anom = int(rng.random() < 0.15)
        route_dev = int(rng.random() < 0.10)
        rz_entry = int(rng.random() < 0.05)
        park_anom = int(rng.random() < 0.08)
        coord_mov = int(rng.random() < 0.03)
        flag_count = speed_anom + route_dev + rz_entry + park_anom + coord_mov
        # Boost risk if many flags
        if flag_count >= 2: risk_score = min(100, risk_score + 15)
        if rz_entry: risk_score = min(100, risk_score + 10)
        if risk_score >= 70: risk_level = "HIGH"
        elif risk_score >= 40: risk_level = "MEDIUM"
        now_ts = datetime.now() - timedelta(seconds=int(rng.integers(0, 30)))
        trip_id = f"LIVE-{st.session_state.sim_counter + _:06d}"
        row = {
            "VEHICLE_ID": vid, "TRIP_ID": trip_id,
            "TIMESTAMP": now_ts, "DATETIME": now_ts,
            "TRIP_DISTANCE": round(distance, 2), "TRAVEL_TIME": round(duration, 1),
            "AVG_SPEED_KMH": round(avg_speed, 1), "MAX_SPEED_KMH": round(max_speed, 1),
            "STD_SPEED": round(float(rng.uniform(5, 25)), 1),
            "PARKING_ANOMALY": park_anom, "PARKING_DURATION_MIN": round(float(rng.exponential(15)), 1),
            "SPEED_ANOMALY": speed_anom, "SPEED_SPIKES": int(rng.integers(0, 5)),
            "ROUTE_DEVIATION": route_dev, "CIRCUITY_RATIO": round(float(rng.uniform(1.0, 2.0)), 2),
            "COORDINATED_MOVEMENT": coord_mov, "RESTRICTED_ZONE_ENTRY": rz_entry,
            "RZ_HIT_COUNT": int(rz_entry * rng.integers(1, 4)),
            "RISK_SCORE": round(risk_score, 1), "RISK_LEVEL": risk_level,
            "IF_LABEL": -1 if risk_score > 50 else 1,
            "IF_RESULT": "Anomaly" if risk_score > 50 else "Normal",
            "ANOMALY_SCORE": round(float(rng.uniform(-0.3, 0.1)), 4),
            "CLUSTER": f"C-{rng.integers(0,5)}",
            "FLAG_COUNT": flag_count,
            "SUSPICION_SCORE": round(risk_score * 0.5 + flag_count * 8, 1),
            "HOUR": now_ts.hour, "DAY": now_ts.strftime("%A"), "DATE": now_ts.date(),
            "PEAK_HOUR": "Peak" if now_ts.hour in list(range(7,10))+list(range(17,20)) else "Off-Peak",
            "TRIP_TYPE": "Commercial" if distance > 20 else ("Short" if distance < 5 else "Standard"),
            "AVG_BEARING_CHANGE": round(float(rng.uniform(10, 180)), 1),
        }
        new_rows.append(row)
        # Log high-risk for alert feed
        if risk_level == "HIGH":
            st.session_state.sim_log.append({
                "trip_id": trip_id, "vehicle_id": str(vid),
                "risk_score": round(risk_score, 1), "risk_level": risk_level,
                "time": now_ts.strftime("%H:%M:%S"),
                "flags": [f for f, v in [("Speed", speed_anom), ("Route", route_dev), ("Zone", rz_entry), ("Parking", park_anom), ("Coord", coord_mov)] if v]
            })
    st.session_state.sim_counter += n
    return pd.DataFrame(new_rows)

# If live mode is active, inject new trips
if st.session_state.live_mode:
    new_batch = generate_live_trips(df, n=int(np.random.randint(2, 6)))
    # Match columns
    for c in df.columns:
        if c not in new_batch.columns:
            new_batch[c] = 0
    new_batch = new_batch[[c for c in df.columns if c in new_batch.columns]]
    st.session_state.sim_trips = pd.concat([st.session_state.sim_trips, new_batch], ignore_index=True)
    # Keep only last 500 simulated trips to avoid memory bloat
    if len(st.session_state.sim_trips) > 500:
        st.session_state.sim_trips = st.session_state.sim_trips.tail(500)
    # Keep only last 20 log entries
    st.session_state.sim_log = st.session_state.sim_log[-20:]
    # Merge with base data
    df = pd.concat([df, st.session_state.sim_trips], ignore_index=True)


total=len(df); crit=int((df["RISK_LEVEL"]=="CRITICAL").sum()); high=int((df["RISK_LEVEL"]=="HIGH").sum())
med=int((df["RISK_LEVEL"]=="MEDIUM").sum()); low=int((df["RISK_LEVEL"]=="LOW").sum())
avg_risk=round(float(df["RISK_SCORE"].mean()),1); max_risk=round(float(df["RISK_SCORE"].max()),1)
avg_spd=round(float(df["AVG_SPEED_KMH"].mean()),1)
max_spd=round(float(df["MAX_SPEED_KMH"].mean()),1) if "MAX_SPEED_KMH" in df.columns else 0.0
active_alerts=crit+high
rz=int(df["RESTRICTED_ZONE_ENTRY"].sum()) if "RESTRICTED_ZONE_ENTRY" in df.columns else 0
sp=int(df["SPEED_ANOMALY"].sum()) if "SPEED_ANOMALY" in df.columns else 0
pk=int(df["PARKING_ANOMALY"].sum()) if "PARKING_ANOMALY" in df.columns else 0
rd=int(df["ROUTE_DEVIATION"].sum()) if "ROUTE_DEVIATION" in df.columns else 0
cm=int(df["COORDINATED_MOVEMENT"].sum()) if "COORDINATED_MOVEMENT" in df.columns else 0
# Real convoy detection count (computed in convoy tab, shown here as preview)

circular=int((df["CIRCUITY_RATIO"]>1.5).sum()) if "CIRCUITY_RATIO" in df.columns else 0
abnormal=int((df["PARKING_DURATION_MIN"]>60).sum()) if "PARKING_DURATION_MIN" in df.columns else 0
recent_alerts=df.nlargest(4,"RISK_SCORE")[["TRIP_ID","RISK_LEVEL","RISK_SCORE"]].values.tolist()

# AUTO-REFRESH (top-level to avoid layout shifts)
if st.session_state.live_mode:
    st_autorefresh(interval=5000, limit=None, key="live_refresh")

# HEADER
h_left,h_center,h_right = st.columns([2.5,6.5,2])
with h_left:
    st.markdown("""<div style="display:flex;align-items:center;gap:12px;">
      <div style="width:32px;height:32px;background:rgba(0,212,255,0.1);border:1px solid rgba(0,212,255,0.5);border-radius:6px;display:flex;align-items:center;justify-content:center;font-size:16px;box-shadow:0 0 10px rgba(0,212,255,0.3);flex-shrink:0;">&#128659;</div>
      <div><div class="header-title">AI ENABLED SMART SURVEILLANCE</div><div class="header-sub">SUSPICIOUS VEHICLE DETECTION SYSTEM</div></div>
    </div>""", unsafe_allow_html=True)
with h_center:
    st.markdown(f"""<div class="kpi-grid">
      <div class="kpi-card"><div class="kpi-icon">&#128663;</div><div class="kpi-label">Total Vehicles</div><div class="kpi-value" style="color:#00d4ff;">{total:,}</div></div>
      <div class="kpi-card"><div class="kpi-icon">&#128994;</div><div class="kpi-label">Low Risk</div><div class="kpi-value" style="color:#00ff88;">{low:,}</div></div>
      <div class="kpi-card"><div class="kpi-icon">&#9888;&#65039;</div><div class="kpi-label">Medium Risk</div><div class="kpi-value" style="color:#ffc107;">{med:,}</div></div>
      <div class="kpi-card"><div class="kpi-icon">&#128314;</div><div class="kpi-label">High Risk</div><div class="kpi-value" style="color:#ff6b35;">{high:,}</div></div>
      <div class="kpi-card" style="border-color:rgba(255,23,68,0.5);box-shadow:0 0 15px rgba(255,23,68,0.2);"><div class="kpi-icon">&#128680;</div><div class="kpi-label">Active Alerts</div><div class="kpi-value" style="color:#ff1744;">{active_alerts:,}</div></div>
    </div>""", unsafe_allow_html=True)
with h_right:
    hr1,hr2 = st.columns([1,1.4])
    with hr1:
        live_on = st.toggle("Live", value=st.session_state.live_mode, key="live_toggle")
        if live_on != st.session_state.live_mode:
            st.session_state.live_mode = live_on
            if not live_on:
                st.session_state.sim_trips = pd.DataFrame()
                st.session_state.sim_counter = 0
                st.session_state.sim_log = []
            st.rerun()

    with hr2:
        if st.session_state.live_mode:
            sim_count = len(st.session_state.sim_trips)
            st.markdown(f'''<div style="text-align:right;padding-top:2px;">
              <div class="live-badge" style="background:rgba(255,23,68,0.12);border-color:rgba(255,23,68,0.5);color:#ff1744;"><div class="live-dot" style="background:#ff1744;box-shadow:0 0 6px #ff1744;"></div> LIVE</div>
              <div style="font-size:clamp(8px,0.9vw,10px);color:#ff6b35;margin-top:3px;font-family:Share Tech Mono,monospace;">{total:,} <span style="color:#ff1744;">(+{sim_count})</span></div>
            </div>''', unsafe_allow_html=True)
        else:
            st.markdown(f'''<div style="text-align:right;padding-top:2px;">
              <div class="live-badge"><div class="live-dot"></div> DATASET</div>
              <div style="font-size:clamp(8px,0.9vw,10px);color:#7ab8e8;margin-top:3px;font-family:Share Tech Mono,monospace;">{total:,} TRIPS</div>
            </div>''', unsafe_allow_html=True)

st.markdown('<div style="height:2px;background:linear-gradient(90deg,transparent,rgba(0,212,255,0.4),transparent);margin:4px 0 8px;"></div>', unsafe_allow_html=True)

# MAP + ANALYTICS
map_col,analytics_col = st.columns([6.5,3.5])
with map_col:
    map_sample = st.slider("Map Sample Size",100,800,350,50,key="mapsz",help="Vehicles on map")
    @st.cache_data(max_entries=1, ttl=5 if st.session_state.get("live_mode") else 300)
    def build_map_data(n, _live_tick=0):
        if not st.session_state.get("live_mode"): np.random.seed(42)
        plat,plon=41.1579,-8.6291
        lats=plat+np.random.uniform(-0.05,0.05,n); lons=plon+np.random.uniform(-0.08,0.08,n)
        risks=np.random.choice(["LOW","MEDIUM","HIGH"],size=n,p=[0.697,0.282,0.021])
        color_map={"LOW":[0,255,136,180],"MEDIUM":[255,193,7,180],"HIGH":[255,107,53,220]}
        size_map={"LOW":4,"MEDIUM":6,"HIGH":10}
        map_df=pd.DataFrame({"lat":lats,"lon":lons,"risk":risks})
        map_df["color"]=map_df["risk"].map(color_map)
        map_df["size"]=map_df["risk"].map(size_map)
        return map_df
    import pydeck as pdk
    map_df=build_map_data(map_sample, _live_tick=st.session_state.get('sim_counter',0))
    layer=pdk.Layer("ScatterplotLayer",data=map_df,get_position=["lon","lat"],get_color="color",
        get_radius="size",radius_scale=15,radius_min_pixels=2,radius_max_pixels=7,pickable=True,
        auto_highlight=True)
    view=pdk.ViewState(latitude=41.1579,longitude=-8.6291,zoom=12,pitch=0)
    deck=pdk.Deck(layers=[layer],initial_view_state=view,
        map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",height=500,
        tooltip={"text":"Risk: {risk}"})
    st.pydeck_chart(deck,use_container_width=True)
    st.markdown(f"""<div style="display:flex;gap:clamp(8px,1.5vw,20px);padding:6px 10px;background:rgba(0,15,40,0.7);border:1px solid rgba(0,212,255,0.15);border-radius:6px;flex-wrap:wrap;align-items:center;">
      <span style="font-size:clamp(8px,0.9vw,10px);color:#7ab8e8;font-family:Share Tech Mono,monospace;">Porto, Portugal</span>
      <span style="font-size:clamp(8px,0.9vw,10px);color:#00ff88;">LOW: {low:,}</span>
      <span style="font-size:clamp(8px,0.9vw,10px);color:#ffc107;">MED: {med:,}</span>
      <span style="font-size:clamp(8px,0.9vw,10px);color:#ff6b35;">HIGH: {high:,}</span>
      <span style="font-size:clamp(8px,0.9vw,10px);color:#7ab8e8;margin-left:auto;">{now_str}</span>
    </div>""", unsafe_allow_html=True)

with analytics_col:
    st.markdown('<div class="sec-head">AI DETECTION ENGINE</div>', unsafe_allow_html=True)
    for label,count in [("Long Parking Detected",abnormal),("Restricted Zone Entry",rz),("Circular Movement",circular),
        ("Abnormal Stop Pattern",int((df.get("SPEED_SPIKES",pd.Series(0,index=df.index))>3).sum())),
        ("Route Deviation",rd),("Speed Anomaly",sp),("Coordinated Movement",cm)]:
        color="#ff1744" if count>1000 else "#ffc107" if count>100 else "#00ff88"
        st.markdown(f'<div class="det-row"><span class="det-check">&#10004;</span><span style="flex:1;">{label}</span><span style="font-weight:700;color:{color};font-family:Share Tech Mono,monospace;font-size:clamp(8px,0.9vw,10px);text-align:right;min-width:40px;flex-shrink:0;">{count:,}</span></div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-head" style="margin-top:8px;">RISK DISTRIBUTION</div>', unsafe_allow_html=True)
    fig_d=go.Figure(go.Pie(labels=["LOW","MEDIUM","HIGH"],values=[low,med,high],marker=dict(colors=["#00ff88","#ffc107","#ff6b35"]),hole=0.5,textinfo="percent",textposition="inside",textfont=dict(color="#0a1628",size=11,family="Inter")))
    fig_d.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(color="#c8e8ff",size=10),margin=dict(t=5,b=5,l=5,r=5),height=200,showlegend=True,legend=dict(bgcolor="rgba(0,0,0,0)",x=1,y=0.5,font=dict(color="#c8e8ff",size=9)),annotations=[dict(text=f"<b>{total:,}</b>",x=0.5,y=0.5,font=dict(size=14,color="#00d4ff",family="Orbitron"),showarrow=False)])
    st.plotly_chart(fig_d,use_container_width=True)

    st.markdown('<div class="sec-head" style="margin-top:4px;">RECENT ALERTS</div>', unsafe_allow_html=True)
    for trip_id,risk,score in recent_alerts:
        c=CLR.get(str(risk),"#7ab8e8")
        st.markdown(f'<div class="alert-row" style="background:{c}15;border-left-color:{c};"><div class="ar-id" style="color:{c};">{str(trip_id)[:12]}</div><div class="ar-badge" style="background:{c}22;color:{c};border:1px solid {c}44;">{risk}</div><div class="ar-time">{float(score):.0f}/100</div></div>', unsafe_allow_html=True)

    # LIVE FEED (only when simulation is active)
    if st.session_state.get("live_mode") and st.session_state.get("sim_log"):
        st.markdown('<div class="sec-head" style="margin-top:8px;">LIVE INCOMING FEED</div>', unsafe_allow_html=True)
        for entry in reversed(st.session_state.sim_log[-5:]):
            ec = CLR.get(entry["risk_level"], "#7ab8e8")
            flags_str = ", ".join(entry["flags"]) if entry["flags"] else "None"
            st.markdown(f'<div class="alert-row" style="background:{ec}15;border-left-color:{ec};animation:fadeIn 0.5s;"><div class="ar-id" style="color:{ec};">&#128680; {entry["trip_id"]}</div><div class="ar-badge" style="background:{ec}22;color:{ec};border:1px solid {ec}44;">{entry["risk_level"]}</div><div class="ar-time">{entry["time"]}</div></div>', unsafe_allow_html=True)

# TABS
st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)
st.markdown('<div class="sec-head">ADVANCED ANALYTICS ENGINE</div>', unsafe_allow_html=True)
tab_overview,tab_temporal,tab_ml,tab_timeline,tab_heatmap,tab_forecast,tab_convoy,tab_explorer = st.tabs(["Overview","Temporal","ML Engine","Vehicle Timeline","Anomaly Heatmap","Risk Trend","Convoy Detection","Explorer"])

with tab_overview:
    c1,c2=st.columns(2)
    with c1:
        st.markdown('<div style="font-size:clamp(9px,1vw,11px);font-weight:700;color:#7ab8e8;letter-spacing:1px;margin-bottom:4px;">RISK LEVEL DISTRIBUTION</div>', unsafe_allow_html=True)
        rc=df["RISK_LEVEL"].value_counts().reset_index(); rc.columns=["Risk Level","Vehicles"]
        fig=px.bar(rc,x="Risk Level",y="Vehicles",color="Risk Level",text="Vehicles",color_discrete_map=CLR)
        fig.update_traces(textposition="outside",textfont=dict(color="white",size=11),cliponaxis=False)
        fig.update_layout(**PLT,height=280,showlegend=False,xaxis={**AX},yaxis={**AX})
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        st.markdown('<div style="font-size:clamp(9px,1vw,11px);font-weight:700;color:#7ab8e8;letter-spacing:1px;margin-bottom:4px;">SPEED VS RISK SCORE</div>', unsafe_allow_html=True)
        samp=df.sample(min(1500,len(df)),random_state=42)
        fig=px.scatter(samp,x="AVG_SPEED_KMH",y="RISK_SCORE",color="RISK_LEVEL",color_discrete_map=CLR,opacity=0.6,labels={"AVG_SPEED_KMH":"Avg Speed (km/h)","RISK_SCORE":"Risk Score","RISK_LEVEL":"Risk Level"})
        fig.update_layout(**PLT,height=280,xaxis={**AX},yaxis={**AX})
        st.plotly_chart(fig,use_container_width=True)
    c3,c4=st.columns(2)
    with c3:
        st.markdown('<div style="font-size:clamp(9px,1vw,11px);font-weight:700;color:#7ab8e8;letter-spacing:1px;margin-bottom:4px;">TRIP TYPE BREAKDOWN</div>', unsafe_allow_html=True)
        tt=df["TRIP_TYPE"].value_counts().reset_index(); tt.columns=["Type","Count"]
        fig=px.bar(tt,x="Type",y="Count",color="Count",text="Count",color_continuous_scale="Blues")
        fig.update_traces(textposition="outside",textfont=dict(color="white"),cliponaxis=False)
        fig.update_layout(**PLT,height=280,showlegend=False,xaxis={**AX},yaxis={**AX})
        st.plotly_chart(fig,use_container_width=True)
    with c4:
        st.markdown('<div style="font-size:clamp(9px,1vw,11px);font-weight:700;color:#7ab8e8;letter-spacing:1px;margin-bottom:4px;">ANOMALY FLAGS OVERVIEW</div>', unsafe_allow_html=True)
        fdf=pd.DataFrame({"Flag":["Parking","Speed","Route Dev","Zone","Coordinated"],"Count":[pk,sp,rd,rz,cm]}).sort_values("Count",ascending=True)
        fig=px.bar(fdf,y="Flag",x="Count",orientation="h",color="Count",text="Count",color_continuous_scale="Reds")
        fig.update_traces(textposition="outside",textfont=dict(color="white"),cliponaxis=False)
        fig.update_layout(**PLT,height=280,showlegend=False,xaxis={**AX},yaxis={**AX})
        st.plotly_chart(fig,use_container_width=True)
    m1=st.columns(3)
    for col,lbl,val,clr in [(m1[0],"Avg Risk Score",avg_risk,"#ffc107"),(m1[1],"Max Risk",max_risk,"#ff1744"),(m1[2],"Avg Speed km/h",avg_spd,"#00d4ff")]:
        col.markdown(f'<div class="kpi-metric"><div class="km-label">{lbl}</div><div class="km-val" style="color:{clr};">{val:.1f}</div></div>', unsafe_allow_html=True)
    m2=st.columns(3)
    for col,lbl,val,clr in [(m2[0],"Avg Max Speed",max_spd,"#ff6b35"),(m2[1],"Zone Violations",rz,"#ff1744"),(m2[2],"Speed Anomalies",sp,"#ff6b35")]:
        v=f"{val:.1f}" if isinstance(val,float) else f"{val:,}"
        col.markdown(f'<div class="kpi-metric"><div class="km-label">{lbl}</div><div class="km-val" style="color:{clr};">{v}</div></div>', unsafe_allow_html=True)

with tab_temporal:
    t1,t2=st.columns(2)
    with t1:
        hourly=df.groupby(["HOUR","RISK_LEVEL"]).size().reset_index(name="Trips"); hourly.rename(columns={"RISK_LEVEL":"Risk Level"},inplace=True)
        fig=px.bar(hourly,x="HOUR",y="Trips",color="Risk Level",color_discrete_map=CLR,labels={"HOUR":"Hour (24h)"})
        fig.update_layout(**PLT,height=300,title=dict(text="Trips by Hour",font=dict(color="white",size=12)),xaxis={**AX},yaxis={**AX})
        st.plotly_chart(fig,use_container_width=True)
    with t2:
        hr=df.groupby("HOUR")["RISK_SCORE"].mean().reset_index()
        fig=go.Figure(); fig.add_trace(go.Scatter(x=hr["HOUR"],y=hr["RISK_SCORE"],mode="lines+markers",line=dict(color="#ff6b35",width=2),marker=dict(color="#ff1744",size=6),fill="tozeroy",fillcolor="rgba(255,107,53,0.1)",name="Avg Risk"))
        fig.update_layout(**PLT,height=300,title=dict(text="Avg Risk by Hour",font=dict(color="white",size=12)),xaxis={**AX,"dtick":2},yaxis={**AX})
        st.plotly_chart(fig,use_container_width=True)
    day_hour=df.groupby(["DAY","HOUR"])["RISK_SCORE"].mean().unstack(fill_value=0)
    do=[d for d in ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"] if d in day_hour.index]
    fig=px.imshow(day_hour.reindex(do),color_continuous_scale="RdYlGn_r",aspect="auto",labels=dict(x="Hour",y="Day",color="Avg Risk"))
    fig.update_layout(**PLT,height=300,title=dict(text="Day x Hour Risk Heatmap",font=dict(color="white",size=12)))
    st.plotly_chart(fig,use_container_width=True)

with tab_ml:
    if "IF_LABEL" not in df.columns or "CLUSTER" not in df.columns:
        st.warning("Pre-computed ML columns missing.")
    else:
        st.markdown('<div class="sec-head">Isolation Forest Results</div>', unsafe_allow_html=True)
        anom_cnt=int((df["IF_LABEL"]==-1).sum()); norm_cnt=int((df["IF_LABEL"]==1).sum())
        mc1,mc2,mc3=st.columns([1,2,2])
        with mc1:
            st.markdown(f'<div class="profile-card" style="min-height:280px;justify-content:space-evenly;"><div class="profile-label">ANOMALIES DETECTED</div><div class="profile-val" style="color:#ff1744;">{anom_cnt:,}</div><div style="color:#7ab8e8;font-size:clamp(9px,1vw,12px);margin-top:6px;">Normal: <b style="color:#00ff88;">{norm_cnt:,}</b></div><div style="color:#7ab8e8;font-size:clamp(9px,1vw,12px);margin-top:4px;">Contamination: <b style="color:#00d4ff;">5%</b></div><div style="color:#7ab8e8;font-size:clamp(9px,1vw,12px);">Trees: <b style="color:#00d4ff;">50</b></div><div style="color:#7ab8e8;font-size:clamp(9px,1vw,12px);margin-top:4px;">Method: <b style="color:#ffc107;">Isolation Forest</b></div></div>', unsafe_allow_html=True)
        with mc2:
            fig=px.scatter(df.sample(min(2000,len(df)),random_state=42),x="AVG_SPEED_KMH",y="RISK_SCORE",color="IF_RESULT",color_discrete_map={"Normal":"#00ff88","Anomaly":"#ff1744"},opacity=0.7,labels={"AVG_SPEED_KMH":"Speed (km/h)","RISK_SCORE":"Risk Score","IF_RESULT":"Result"})
            fig.update_layout(**PLT,height=280,title=dict(text="Anomaly Scatter",font=dict(color="white",size=12)),xaxis={**AX},yaxis={**AX})
            st.plotly_chart(fig,use_container_width=True)
        with mc3:
            rc2=df["IF_RESULT"].value_counts().reset_index(); rc2.columns=["Result","Count"]
            fig2=px.pie(rc2,names="Result",values="Count",color="Result",color_discrete_map={"Normal":"#00ff88","Anomaly":"#ff1744"},hole=0.5)
            fig2.update_layout(**PLT,height=280,title=dict(text="IF Distribution",font=dict(color="white",size=12)))
            st.plotly_chart(fig2,use_container_width=True)
        st.markdown('<div class="sec-head" style="margin-top:8px;">DBSCAN Clustering Results</div>', unsafe_allow_html=True)
        cl_cnt=len([c for c in df["CLUSTER"].unique() if c!="Noise"])
        dc1,dc2=st.columns(2)
        with dc1:
            fig=px.scatter(df.sample(min(3000,len(df)),random_state=42),x="AVG_SPEED_KMH",y="RISK_SCORE",color="CLUSTER",opacity=0.7,labels={"AVG_SPEED_KMH":"Speed","RISK_SCORE":"Risk"})
            fig.update_layout(**PLT,height=280,title=dict(text=f"DBSCAN - {cl_cnt} Clusters Found",font=dict(color="white",size=12)),xaxis={**AX},yaxis={**AX})
            st.plotly_chart(fig,use_container_width=True)
        with dc2:
            cl_dist=df["CLUSTER"].value_counts().reset_index(); cl_dist.columns=["CLUSTER","Count"]
            fig=px.bar(cl_dist.head(10),x="CLUSTER",y="Count",color="Count",color_continuous_scale="Blues",text="Count")
            fig.update_traces(textposition="outside",textfont=dict(color="white"),cliponaxis=False)
            fig.update_layout(**PLT,height=280,showlegend=False,title=dict(text="Cluster Distribution",font=dict(color="white",size=12)),xaxis={**AX},yaxis={**AX})
            st.plotly_chart(fig,use_container_width=True)

with tab_timeline:
    st.markdown('<div class="sec-head">VEHICLE BEHAVIOUR TIMELINE</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:clamp(9px,1vw,11px);color:#7ab8e8;margin-bottom:8px;">Track risk escalation across all trips to identify reconnaissance patterns.</div>', unsafe_allow_html=True)
    veh_ids=sorted(df["VEHICLE_ID"].unique().astype(str).tolist()) if "VEHICLE_ID" in df.columns else []
    tl_c1,tl_c2=st.columns([2,4])
    with tl_c1:
        selected_veh=st.selectbox("Select Vehicle ID",veh_ids,index=0,key="veh_timeline")
        veh_df=df[df["VEHICLE_ID"].astype(str)==selected_veh].copy()
        if "TIMESTAMP" in veh_df.columns: veh_df=veh_df.sort_values("TIMESTAMP")
        total_trips=len(veh_df); high_trips=int((veh_df["RISK_LEVEL"]=="HIGH").sum())
        avg_r=round(float(veh_df["RISK_SCORE"].mean()),1) if len(veh_df)>0 else 0
        total_km=round(float(veh_df["TRIP_DISTANCE"].sum()),1) if "TRIP_DISTANCE" in veh_df.columns else 0
        total_flags=int(veh_df["FLAG_COUNT"].sum())
        flag_cols=[c for c in ["SPEED_ANOMALY","ROUTE_DEVIATION","RESTRICTED_ZONE_ENTRY","PARKING_ANOMALY","COORDINATED_MOVEMENT"] if c in veh_df.columns]
        if flag_cols:
            flag_sums={c.replace("_"," ").title():int(veh_df[c].sum()) for c in flag_cols}
            top_flag=max(flag_sums,key=flag_sums.get); top_flag_count=max(flag_sums.values())
        else: top_flag,top_flag_count="N/A",0
        peak_hr_str=f"{int(veh_df['HOUR'].mode().iloc[0]):02d}:00" if "HOUR" in veh_df.columns and len(veh_df)>0 else "N/A"
        for lbl,val,clr in [("TOTAL TRIPS",f"{total_trips}","#00d4ff"),("HIGH RISK TRIPS",f"{high_trips}","#ff1744"),
            ("AVG RISK SCORE",f"{avg_r}","#ff6b35" if avg_r>40 else "#ffc107" if avg_r>25 else "#00ff88"),
            ("TOTAL DISTANCE",f"{total_km:,.1f} km","#00d4ff"),("TOTAL FLAGS",f"{total_flags}","#ff6b35"),("PEAK HOUR",peak_hr_str,"#ffc107")]:
            st.markdown(f'<div class="profile-card" style="margin-bottom:6px;"><div class="profile-label">{lbl}</div><div class="profile-val" style="color:{clr};">{val}</div></div>', unsafe_allow_html=True)
    with tl_c2:
        if "TIMESTAMP" in veh_df.columns and len(veh_df)>0:
            veh_plot=veh_df.reset_index(drop=True); veh_plot["TRIP_NUM"]=range(1,len(veh_plot)+1)
            color_map={"LOW":"#00ff88","MEDIUM":"#ffc107","HIGH":"#ff1744"}
            fig_tl=go.Figure()
            fig_tl.add_trace(go.Scatter(x=veh_plot["TRIP_NUM"],y=veh_plot["RISK_SCORE"],mode="lines",line=dict(color="rgba(0,212,255,0.3)",width=1.5),showlegend=False,hoverinfo="skip"))
            for rl,color in color_map.items():
                mask=veh_plot["RISK_LEVEL"]==rl
                if not mask.any(): continue
                subset=veh_plot[mask]; htexts=[]
                for _,row in subset.iterrows():
                    flags=[]
                    if row.get("SPEED_ANOMALY",0): flags.append("Speed")
                    if row.get("ROUTE_DEVIATION",0): flags.append("Route")
                    if row.get("RESTRICTED_ZONE_ENTRY",0): flags.append("Zone")
                    if row.get("PARKING_ANOMALY",0): flags.append("Parking")
                    if row.get("COORDINATED_MOVEMENT",0): flags.append("Coord")
                    ts=row.get("TIMESTAMP",""); ts_s=ts.strftime("%b %d %H:%M") if pd.notna(ts) else "N/A"
                    htexts.append(f"Trip #{row['TRIP_NUM']}<br>Time: {ts_s}<br>Risk: {float(row['RISK_SCORE']):.1f} ({rl})<br>Speed: {float(row['AVG_SPEED_KMH']):.1f} km/h<br>Flags: {', '.join(flags) if flags else 'None'}")
                fig_tl.add_trace(go.Scatter(x=subset["TRIP_NUM"],y=subset["RISK_SCORE"],mode="markers",name=rl,
                    marker=dict(color=color,size=8 if rl=="HIGH" else 6,line=dict(color="white",width=1) if rl=="HIGH" else dict(width=0)),text=htexts,hoverinfo="text"))
            fig_tl.add_hline(y=50,line_dash="dash",line_color="rgba(255,107,53,0.3)",annotation_text="HIGH threshold",annotation_font_color="#ff6b35",annotation_font_size=9)
            fig_tl.add_hline(y=30,line_dash="dot",line_color="rgba(255,193,7,0.3)",annotation_text="MEDIUM threshold",annotation_font_color="#ffc107",annotation_font_size=9)
            fig_tl.update_layout(**PLT,height=440,title=dict(text=f"Risk Escalation - Vehicle {selected_veh}",font=dict(color="white",size=13)),
                xaxis=dict(**AX,title=dict(text="Trip Number",font=dict(color="#7ab8e8",size=10))),
                yaxis=dict(**AX,title=dict(text="Risk Score",font=dict(color="#7ab8e8",size=10)),range=[0,max(float(veh_plot["RISK_SCORE"].max())*1.15,60)]))
            st.plotly_chart(fig_tl,use_container_width=True)
            if flag_cols:
                flag_data=pd.DataFrame({"Flag":[c.replace("_"," ").title() for c in flag_cols],"Count":[int(veh_df[c].sum()) for c in flag_cols]}).sort_values("Count",ascending=True)
                fig_f=px.bar(flag_data,y="Flag",x="Count",orientation="h",color="Count",color_continuous_scale="Reds",text="Count")
                fig_f.update_traces(textposition="outside",textfont=dict(color="white"),cliponaxis=False)
                fig_f.update_layout(**PLT,height=220,showlegend=False,title=dict(text="Flag Breakdown for This Vehicle",font=dict(color="white",size=11)),xaxis={**AX},yaxis={**AX})
                st.plotly_chart(fig_f,use_container_width=True)
        else: st.info("No timestamp data available for timeline.")

with tab_heatmap:
    st.markdown('<div class="sec-head">ANOMALY HEATMAP GRID</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:clamp(9px,1vw,11px);color:#7ab8e8;margin-bottom:8px;">Identifies WHERE and WHEN suspicious activity clusters. Darker cells = more anomalies at that hour/zone.</div>', unsafe_allow_html=True)
    hm1,hm2=st.columns(2)
    with hm1:
        if "HOUR" in df.columns and "RISK_LEVEL" in df.columns:
            heat_data=df[df["RISK_LEVEL"].isin(["HIGH","CRITICAL"])].copy()
            if "TRIP_TYPE" in heat_data.columns:
                hm_pivot=heat_data.groupby(["TRIP_TYPE","HOUR"]).size().unstack(fill_value=0)
                fig_hm=px.imshow(hm_pivot,color_continuous_scale="YlOrRd",aspect="auto",
                    labels=dict(x="Hour of Day",y="Trip Type",color="Incidents"))
                fig_hm.update_layout(**PLT,height=340,title=dict(text="High-Risk Incidents: Trip Type x Hour",font=dict(color="white",size=12)))
                st.plotly_chart(fig_hm,use_container_width=True)
    with hm2:
        if "HOUR" in df.columns:
            flag_cols_hm=[c for c in ["SPEED_ANOMALY","ROUTE_DEVIATION","RESTRICTED_ZONE_ENTRY","PARKING_ANOMALY","COORDINATED_MOVEMENT"] if c in df.columns]
            if flag_cols_hm:
                flag_hour=df.groupby("HOUR")[flag_cols_hm].sum()
                flag_hour.columns=[c.replace("_"," ").title() for c in flag_hour.columns]
                fig_fh=px.imshow(flag_hour.T,color_continuous_scale="RdBu_r",aspect="auto",
                    labels=dict(x="Hour of Day",y="Anomaly Type",color="Count"))
                fig_fh.update_layout(**PLT,height=340,title=dict(text="Anomaly Type x Hour Heatmap",font=dict(color="white",size=12)))
                st.plotly_chart(fig_fh,use_container_width=True)
    if "DAY" in df.columns and "HOUR" in df.columns:
        st.markdown('<div style="font-size:clamp(9px,1vw,11px);font-weight:700;color:#7ab8e8;letter-spacing:1px;margin:8px 0 4px;">ZONE VIOLATION HOTSPOTS BY DAY</div>', unsafe_allow_html=True)
        if "RESTRICTED_ZONE_ENTRY" in df.columns:
            rz_data=df[df["RESTRICTED_ZONE_ENTRY"]==1].copy()
            rz_pivot=rz_data.groupby(["DAY","HOUR"]).size().unstack(fill_value=0)
            day_order=[d for d in ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"] if d in rz_pivot.index]
            fig_rz=px.imshow(rz_pivot.reindex(day_order),color_continuous_scale="Hot",aspect="auto",
                labels=dict(x="Hour",y="Day",color="Zone Violations"))
            fig_rz.update_layout(**PLT,height=280,title=dict(text="Restricted Zone Violations: Day x Hour",font=dict(color="white",size=12)))
            st.plotly_chart(fig_rz,use_container_width=True)
    hm_m1,hm_m2,hm_m3=st.columns(3)
    peak_hour_anom=int(df.groupby("HOUR")["FLAG_COUNT"].sum().idxmax()) if "HOUR" in df.columns else 0
    peak_day_anom=df.groupby("DAY")["FLAG_COUNT"].sum().idxmax() if "DAY" in df.columns else "N/A"
    worst_type=df[["SPEED_ANOMALY","ROUTE_DEVIATION","RESTRICTED_ZONE_ENTRY","PARKING_ANOMALY"]].sum().idxmax().replace("_"," ").title() if "SPEED_ANOMALY" in df.columns else "N/A"
    for col,lbl,val,clr in [(hm_m1,"Peak Anomaly Hour",f"{peak_hour_anom:02d}:00","#ff1744"),(hm_m2,"Worst Day",str(peak_day_anom),"#ff6b35"),(hm_m3,"Top Anomaly Type",str(worst_type),"#ffc107")]:
        col.markdown(f'<div class="kpi-metric"><div class="km-label">{lbl}</div><div class="km-val" style="color:{clr};font-size:clamp(12px,1.4vw,18px);">{val}</div></div>', unsafe_allow_html=True)

with tab_forecast:
    st.markdown('<div class="sec-head">RISK TREND ANALYSIS</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:clamp(9px,1vw,11px);color:#7ab8e8;margin-bottom:8px;">Identifies vehicles most likely to escalate to HIGH risk based on behavioral patterns and flag accumulation trends.</div>', unsafe_allow_html=True)
    if "VEHICLE_ID" in df.columns:
        veh_stats=df.groupby("VEHICLE_ID").agg(
            total_trips=("RISK_SCORE","count"),
            avg_risk=("RISK_SCORE","mean"),
            max_risk=("RISK_SCORE","max"),
            total_flags=("FLAG_COUNT","sum"),
            high_trips=("RISK_LEVEL",lambda x:(x=="HIGH").sum()),
            avg_speed=("AVG_SPEED_KMH","mean")
        ).reset_index()
        veh_stats["risk_trend"]=veh_stats["avg_risk"]*0.35+veh_stats["total_flags"]*1.5+veh_stats["high_trips"]*8+veh_stats["max_risk"]*0.2
        veh_stats["risk_trend"]=veh_stats["risk_trend"].round(1)
        veh_stats=veh_stats.sort_values("risk_trend",ascending=False)
        top_risky=veh_stats.head(15)
        fc1,fc2=st.columns([3.5,1.5])
        with fc1:
            fig_fc=go.Figure()
            colors=["#ff1744" if r>60 else "#ff6b35" if r>40 else "#ffc107" if r>20 else "#00ff88" for r in top_risky["risk_trend"]]
            fig_fc.add_trace(go.Bar(x=top_risky["VEHICLE_ID"].astype(str),y=top_risky["risk_trend"],
                marker_color=colors,text=top_risky["risk_trend"].apply(lambda x:f"{x:.0f}"),textposition="outside",textfont=dict(color="white",size=10)))
            fig_fc.add_hline(y=60,line_dash="dash",line_color="rgba(255,23,68,0.5)",annotation_text="CRITICAL THRESHOLD",annotation_font_color="#ff1744",annotation_font_size=9)
            fig_fc.update_layout(**PLT,height=350,title=dict(text="Top 15 Vehicles by Risk Trend Score",font=dict(color="white",size=12)),
                xaxis=dict(**AX,title=dict(text="Vehicle ID",font=dict(color="#7ab8e8",size=10)),tickangle=-45),
                yaxis=dict(**AX,title=dict(text="Composite Risk Index",font=dict(color="#7ab8e8",size=10))))
            st.plotly_chart(fig_fc,use_container_width=True)
        with fc2:
            watch_html='<div class="info-card" style="justify-content:space-evenly;padding-top:20px;padding-bottom:20px;"><h4>&#128680; HIGH-RISK WATCH LIST</h4>'
            for _,v in top_risky.head(6).iterrows():
                risk_val=float(v["risk_trend"])
                rc="#ff1744" if risk_val>60 else "#ff6b35" if risk_val>40 else "#ffc107"
                vid=str(v["VEHICLE_ID"])[:12]
                trips=int(v["total_trips"]); flags=int(v["total_flags"])
                watch_html+=f'<div class="alert-row" style="background:{rc}12;border-left-color:{rc};"><div class="ar-id" style="color:{rc};">{vid}</div><div style="font-size:clamp(8px,0.8vw,9px);color:#7ab8e8;">{trips}T/{flags}F</div><div class="ar-badge" style="background:{rc}22;color:{rc};border:1px solid {rc}44;">{risk_val:.0f}</div></div>'
            watch_html+='</div>'
            st.markdown(watch_html, unsafe_allow_html=True)
        st.markdown('<div style="font-size:clamp(9px,1vw,11px);font-weight:700;color:#7ab8e8;letter-spacing:1px;margin:10px 0 4px;">RISK FACTOR CORRELATION</div>', unsafe_allow_html=True)
        fc3,fc4=st.columns(2)
        with fc3:
            fig_sc=px.scatter(veh_stats.head(200),x="total_flags",y="avg_risk",size="total_trips",color="risk_trend",
                color_continuous_scale="YlOrRd",labels={"total_flags":"Total Flags","avg_risk":"Avg Risk Score","total_trips":"Trips","risk_trend":"Risk Index"},opacity=0.8)
            fig_sc.update_layout(**PLT,height=300,title=dict(text="Flags vs Risk (bubble = trips)",font=dict(color="white",size=11)),xaxis={**AX},yaxis={**AX})
            st.plotly_chart(fig_sc,use_container_width=True)
        with fc4:
            risk_bins=pd.cut(veh_stats["risk_trend"],bins=[0,20,40,60,100],labels=["Low","Medium","High","Critical"])
            rb_counts=risk_bins.value_counts().reset_index(); rb_counts.columns=["Category","Vehicles"]
            fig_rb=px.pie(rb_counts,names="Category",values="Vehicles",color="Category",
                color_discrete_map={"Low":"#00ff88","Medium":"#ffc107","High":"#ff6b35","Critical":"#ff1744"},hole=0.5)
            fig_rb.update_layout(**PLT,height=300,title=dict(text="Vehicle Risk Distribution (Trend)",font=dict(color="white",size=11)))
            st.plotly_chart(fig_rb,use_container_width=True)
        fm1,fm2,fm3=st.columns(3)
        crit_count=int((veh_stats["risk_trend"]>60).sum()); watch_count=int((veh_stats["risk_trend"]>40).sum())
        avg_trend=round(float(veh_stats["risk_trend"].mean()),1)
        for col,lbl,val,clr in [(fm1,"Critical Risk Vehicles",f"{crit_count:,}","#ff1744"),(fm2,"Watch List Total",f"{watch_count:,}","#ff6b35"),(fm3,"Avg Risk Index",f"{avg_trend}","#ffc107")]:
            col.markdown(f'<div class="kpi-metric"><div class="km-label">{lbl}</div><div class="km-val" style="color:{clr};">{val}</div></div>', unsafe_allow_html=True)


with tab_convoy:
    st.markdown('<div class="sec-head">CONVOY / COORDINATED MOVEMENT DETECTION</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:clamp(9px,1vw,11px);color:#7ab8e8;margin-bottom:8px;">Identifies groups of vehicles exhibiting coordinated behaviour — similar routes, overlapping timestamps, and comparable speed profiles suggesting organised movement.</div>', unsafe_allow_html=True)
    if "DATETIME" in df.columns and "VEHICLE_ID" in df.columns:
        dfc=df.copy()
        dfc["DT"]=pd.to_datetime(dfc["DATETIME"],errors="coerce")
        dfc=dfc.dropna(subset=["DT"])
        dfc["HOUR"]=dfc["DT"].dt.hour
        dfc["DATE"]=dfc["DT"].dt.date
        # Convoy algorithm: group trips by same hour+date, then find vehicles with similar speed/distance profiles
        # Step 1: Time-window grouping (same date + same hour = potential convoy window)
        time_groups=dfc.groupby(["DATE","HOUR"])
        convoy_results=[]
        convoy_id=0
        for (date,hour),group in time_groups:
            if len(group)<2: continue
            vids=group["VEHICLE_ID"].unique()
            if len(vids)<2: continue
            # Step 2: Within each time window, cluster by speed + distance similarity
            from scipy.spatial.distance import pdist, squareform
            gf=group.groupby("VEHICLE_ID").agg(
                avg_speed=("AVG_SPEED_KMH","mean"),
                avg_risk=("RISK_SCORE","mean"),
                trip_count=("RISK_SCORE","count")
            ).reset_index()
            if len(gf)<2: continue
            # Normalize features
            from sklearn.preprocessing import MinMaxScaler
            scaler=MinMaxScaler()
            features=gf[["avg_speed","avg_risk"]].values
            if features.shape[0]<2: continue
            try:
                feat_norm=scaler.fit_transform(features)
            except: continue
            # DBSCAN on normalized speed+distance to find similar-profile vehicles
            from sklearn.cluster import DBSCAN as DBSCAN2
            db=DBSCAN2(eps=0.15,min_samples=2).fit(feat_norm)
            gf["convoy_cluster"]=db.labels_
            for cl in set(db.labels_):
                if cl==-1: continue
                members=gf[gf["convoy_cluster"]==cl]
                if len(members)<2: continue
                convoy_id+=1
                for _,m in members.iterrows():
                    convoy_results.append({
                        "Convoy ID":f"CVY-{convoy_id:03d}",
                        "Date":str(date),
                        "Hour":f"{hour:02d}:00",
                        "Vehicle ID":int(m["VEHICLE_ID"]),
                        "Avg Speed":round(float(m["avg_speed"]),1),
                        "Trips":int(m["trip_count"]),
                        "Risk Score":round(float(m["avg_risk"]),1)
                    })
        if convoy_results:
            convoy_df=pd.DataFrame(convoy_results)
            total_convoys=convoy_df["Convoy ID"].nunique()
            total_vehicles_in_convoys=convoy_df["Vehicle ID"].nunique()
            high_risk_convoys=0
            for cid in convoy_df["Convoy ID"].unique():
                cg=convoy_df[convoy_df["Convoy ID"]==cid]
                if cg["Risk Score"].mean()>40: high_risk_convoys+=1
            # KPI Row
            ck1,ck2,ck3,ck4=st.columns(4)
            for col,lbl,val,clr in [(ck1,"Convoys Detected",f"{total_convoys:,}","#00d4ff"),(ck2,"Vehicles Involved",f"{total_vehicles_in_convoys:,}","#ffc107"),(ck3,"High-Risk Convoys",f"{high_risk_convoys:,}","#ff1744"),(ck4,"Avg Group Size",f"{len(convoy_df)/total_convoys:.1f}","#00ff88")]:
                col.markdown(f'<div class="kpi-metric"><div class="km-label">{lbl}</div><div class="km-val" style="color:{clr};">{val}</div></div>', unsafe_allow_html=True)
            # Charts row
            cv1,cv2=st.columns(2)
            with cv1:
                # Convoy size distribution
                convoy_sizes=convoy_df.groupby("Convoy ID")["Vehicle ID"].count().reset_index()
                convoy_sizes.columns=["Convoy","Vehicles"]
                fig_cs=px.histogram(convoy_sizes,x="Vehicles",nbins=10,color_discrete_sequence=["#00d4ff"],
                    labels={"Vehicles":"Group Size","count":"Frequency"})
                fig_cs.update_layout(**PLT,height=280,title=dict(text="Convoy Size Distribution",font=dict(color="white",size=12)),
                    xaxis=dict(**AX,title=dict(text="Vehicles per Convoy",font=dict(color="#7ab8e8",size=10))),
                    yaxis=dict(**AX,title=dict(text="Count",font=dict(color="#7ab8e8",size=10))))
                st.plotly_chart(fig_cs,use_container_width=True)
            with cv2:
                # Convoy risk distribution
                convoy_risk=convoy_df.groupby("Convoy ID")["Risk Score"].mean().reset_index()
                convoy_risk.columns=["Convoy","Avg Risk"]
                convoy_risk["Threat"]=pd.cut(convoy_risk["Avg Risk"],bins=[0,20,40,60,100],labels=["Low","Medium","High","Critical"])
                threat_dist=convoy_risk["Threat"].value_counts().reset_index()
                threat_dist.columns=["Level","Count"]
                fig_td=px.pie(threat_dist,names="Level",values="Count",color="Level",
                    color_discrete_map={"Low":"#00ff88","Medium":"#ffc107","High":"#ff6b35","Critical":"#ff1744"},hole=0.5)
                fig_td.update_layout(**PLT,height=280,title=dict(text="Convoy Threat Classification",font=dict(color="white",size=12)))
                fig_td.update_traces(textinfo="percent+label",textposition="inside",textfont=dict(color="#0a1628",size=10))
                st.plotly_chart(fig_td,use_container_width=True)
            # Convoy timeline — when do convoys occur
            cv3,cv4=st.columns(2)
            with cv3:
                hour_dist=convoy_df.groupby("Hour")["Convoy ID"].nunique().reset_index()
                hour_dist.columns=["Hour","Convoys"]
                fig_ht=px.bar(hour_dist,x="Hour",y="Convoys",color="Convoys",color_continuous_scale="YlOrRd",text="Convoys")
                fig_ht.update_traces(textposition="outside",textfont=dict(color="white",size=10),cliponaxis=False)
                fig_ht.update_layout(**PLT,height=280,title=dict(text="Convoy Activity by Hour",font=dict(color="white",size=12)),
                    showlegend=False,xaxis=dict(**AX,title=dict(text="Hour of Day",font=dict(color="#7ab8e8",size=10))),
                    yaxis=dict(**AX,title=dict(text="Convoy Groups",font=dict(color="#7ab8e8",size=10))))
                st.plotly_chart(fig_ht,use_container_width=True)
            with cv4:
                # Speed similarity scatter — convoy members
                top_convoys=convoy_df.groupby("Convoy ID")["Risk Score"].mean().nlargest(5).index.tolist()
                top_cv_df=convoy_df[convoy_df["Convoy ID"].isin(top_convoys)]
                fig_sp=px.scatter(top_cv_df,x="Avg Speed",y="Risk Score",color="Convoy ID",size="Trips",
                    color_discrete_sequence=["#ff1744","#ff6b35","#ffc107","#00d4ff","#00ff88"],opacity=0.8,
                    labels={"Avg Speed":"Avg Speed (km/h)","Risk Score":"Risk Score"})
                fig_sp.update_layout(**PLT,height=280,title=dict(text="Top 5 Convoys — Speed vs Distance",font=dict(color="white",size=12)),
                    xaxis=dict(**AX),yaxis=dict(**AX))
                st.plotly_chart(fig_sp,use_container_width=True)
            # Detailed convoy table — top 20 convoys
            st.markdown('<div style="font-size:clamp(9px,1vw,11px);font-weight:700;color:#7ab8e8;letter-spacing:1px;margin:10px 0 4px;">TOP DETECTED CONVOYS</div>', unsafe_allow_html=True)
            show_convoys=convoy_df[convoy_df["Convoy ID"].isin(convoy_df.groupby("Convoy ID")["Risk Score"].mean().nlargest(20).index)]
            header_html="".join([f'<th style="background:rgba(0,40,90,0.95);color:#00d4ff;padding:8px 10px;font-size:11px;font-weight:700;border-bottom:2px solid rgba(0,212,255,0.4);white-space:nowrap;font-family:Inter,sans-serif;text-align:left;">{c}</th>' for c in show_convoys.columns])
            rows_html=""
            for i,(_,row) in enumerate(show_convoys.iterrows()):
                bg="rgba(0,18,50,0.95)" if i%2==0 else "rgba(0,25,65,0.9)"
                cells=""
                for col in show_convoys.columns:
                    val=row[col]
                    if col=="Risk Score":
                        sc=float(val)
                        rc="#ff1744" if sc>=50 else "#ff6b35" if sc>=30 else "#ffc107" if sc>=15 else "#00ff88"
                        cells+=f'<td style="background:{bg};padding:6px 10px;color:{rc};font-weight:700;font-size:11px;border-bottom:1px solid rgba(0,212,255,0.1);font-family:Share Tech Mono,monospace;">{sc:.1f}</td>'
                    elif col=="Convoy ID":
                        cells+=f'<td style="background:{bg};padding:6px 10px;color:#00d4ff;font-weight:700;font-size:11px;border-bottom:1px solid rgba(0,212,255,0.1);font-family:Share Tech Mono,monospace;">{val}</td>'
                    else:
                        cells+=f'<td style="background:{bg};padding:6px 10px;color:#c8e8ff;font-size:11px;border-bottom:1px solid rgba(0,212,255,0.1);font-family:Inter,sans-serif;">{val}</td>'
                rows_html+=f"<tr>{cells}</tr>"
            st.markdown(f'''<div style="overflow-x:auto;border-radius:8px;border:1px solid rgba(0,212,255,0.2);max-height:350px;overflow-y:auto;"><table style="width:100%;border-collapse:collapse;"><thead><tr>{header_html}</tr></thead><tbody>{rows_html}</tbody></table></div>''', unsafe_allow_html=True)
        else:
            st.info("No convoy patterns detected in the current dataset.")
    else:
        st.warning("Required columns (DATETIME, VEHICLE_ID) not available for convoy analysis.")


with tab_explorer:
    ex1,ex2,ex3=st.columns([3,1,1])
    sq=ex1.text_input("Search Vehicle / Trip ID",""); sr2=ex2.selectbox("Risk",["All"]+sorted(df["RISK_LEVEL"].unique().tolist())); rn=ex3.selectbox("Show",["25","50","100"],index=0)
    exp=df.copy()
    if sq: exp=exp[exp["TRIP_ID"].astype(str).str.contains(sq,case=False,na=False)]
    if sr2!="All": exp=exp[exp["RISK_LEVEL"]==sr2]
    if "IF_RESULT" in exp.columns: exp=exp[exp["IF_RESULT"].isin(["Normal","Anomaly"])]
    exp=exp.sort_values("RISK_SCORE",ascending=False)
    dcols=[c for c in ["TRIP_ID","RISK_LEVEL","RISK_SCORE","SUSPICION_SCORE","AVG_SPEED_KMH","TRIP_DISTANCE","TRAVEL_TIME","FLAG_COUNT","RESTRICTED_ZONE_ENTRY","SPEED_ANOMALY","ROUTE_DEVIATION"] if c in exp.columns]
    disp=exp[dcols].head(int(rn)).copy(); disp.columns=[x.replace("_"," ").title() for x in disp.columns]
    risk_colors={"HIGH":"#ff1744","MEDIUM":"#ffc107","LOW":"#00ff88","CRITICAL":"#ff1744"}
    header_html="".join([f'<th style="background:rgba(0,40,90,0.95);color:#00d4ff;padding:10px 12px;font-size:12px;font-weight:700;border-bottom:2px solid rgba(0,212,255,0.4);white-space:nowrap;font-family:Inter,sans-serif;text-align:left;">{c}</th>' for c in disp.columns])
    rows_html=""
    for i,(_,row) in enumerate(disp.iterrows()):
        bg="rgba(0,18,50,0.95)" if i%2==0 else "rgba(0,25,65,0.9)"
        cells=""
        for j,col in enumerate(disp.columns):
            val=row[col]
            if col=="Risk Level":
                rc=risk_colors.get(str(val),"#c8e8ff")
                cells+=f'<td style="background:{bg};padding:8px 12px;color:{rc};font-weight:700;font-size:11px;border-bottom:1px solid rgba(0,212,255,0.1);font-family:Inter,sans-serif;">{val}</td>'
            elif col=="Risk Score":
                sc=float(val) if val else 0
                rc="#ff1744" if sc>=70 else "#ff6b35" if sc>=50 else "#ffc107" if sc>=30 else "#00ff88"
                cells+=f'<td style="background:{bg};padding:8px 12px;color:{rc};font-weight:700;font-size:12px;border-bottom:1px solid rgba(0,212,255,0.1);font-family:Share Tech Mono,monospace;">{sc:.1f}</td>'
            else:
                cells+=f'<td style="background:{bg};padding:8px 12px;color:#c8e8ff;font-size:11px;border-bottom:1px solid rgba(0,212,255,0.1);font-family:Inter,sans-serif;">{val}</td>'
        rows_html+=f"<tr>{cells}</tr>"
    st.markdown(f'''<div style="overflow-x:auto;border-radius:8px;border:1px solid rgba(0,212,255,0.2);max-height:450px;overflow-y:auto;"><table style="width:100%;border-collapse:collapse;"><thead><tr>{header_html}</tr></thead><tbody>{rows_html}</tbody></table></div>''', unsafe_allow_html=True)
    st.download_button("Download Results",exp.to_csv(index=False),f"results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv","text/csv")

# BOTTOM STRIP
st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)
b1,b2,b3,b4=st.columns(4)
with b1:
    st.markdown("""<div class="info-card"><h4>&#9889; SYSTEM ARCHITECTURE</h4>
    <div style="font-size:clamp(10px,1.1vw,13px);color:#c8e8ff;font-family:Inter,sans-serif;line-height:2.2;">
      <div>&#128663; Vehicle Simulator</div>
      <div>&#128225; MQTT Broker (design-intent)</div>
      <div style="color:#7ab8e8;">&#9889; FastAPI (design-intent)</div>
      <div style="color:#ffc107;">&#129302; AI Engine (IF + DBSCAN)</div>
      <div>&#128451; PostgreSQL (design-intent)</div>
      <div>&#128200; Rule-Based Analytics</div>
      <div style="color:#00d4ff;">&#128202; Streamlit Dashboard</div>
    </div></div>""", unsafe_allow_html=True)
with b2:
    st.markdown("""<div class="info-card"><h4>&#128260; WORKING FLOW</h4>
    <div style="font-size:clamp(10px,1.1vw,13px);color:#c8e8ff;font-family:Inter,sans-serif;line-height:2.2;">
      <div><span style="color:#00d4ff;font-weight:700;">&#10122;</span> Vehicle Data Received Every 3-5s</div>
      <div><span style="color:#00d4ff;font-weight:700;">&#10123;</span> Data Streamed via MQTT</div>
      <div><span style="color:#00d4ff;font-weight:700;">&#10124;</span> AI Engine Analyses Behaviour</div>
      <div><span style="color:#ffc107;font-weight:700;">&#10125;</span> Risk Score Calculated (0-100)</div>
      <div><span style="color:#ff6b35;font-weight:700;">&#10126;</span> Suspicious Activity = Alert</div>
      <div><span style="color:#ff1744;font-weight:700;">&#10127;</span> Dashboard Updated Real-Time</div>
      <div><span style="color:#00ff88;font-weight:700;">&#10128;</span> Zone Monitoring Active</div>
    </div></div>""", unsafe_allow_html=True)
with b3:
    st.markdown("""<div class="info-card"><h4>&#128225; SYSTEM STATUS</h4>
    <div style="font-size:clamp(10px,1.1vw,13px);color:#c8e8ff;font-family:Inter,sans-serif;line-height:2.2;">
      <div><span style="color:#00ff88;font-size:14px;">&#9679;</span> Data Pipeline</div>
      <div><span style="color:#00ff88;font-size:14px;">&#9679;</span> Rule Engine</div>
      <div><span style="color:#00ff88;font-size:14px;">&#9679;</span> Isolation Forest</div>
      <div><span style="color:#00ff88;font-size:14px;">&#9679;</span> DBSCAN Clustering</div>
      <div><span style="color:#00ff88;font-size:14px;">&#9679;</span> Zone Monitor</div>
      <div><span style="color:#00ff88;font-size:14px;">&#9679;</span> Risk Scoring</div>
      <div><span style="color:#00ff88;font-size:14px;">&#9679;</span> Alert Engine</div>
    </div></div>""", unsafe_allow_html=True)
with b4:
    st.markdown("""<div class="info-card"><h4>&#128736; TECHNOLOGIES USED</h4>
    <div style="font-size:clamp(10px,1.1vw,13px);color:#c8e8ff;font-family:Inter,sans-serif;line-height:2.2;">
      <div>&#128013; Python 3.14 (Backend)</div>
      <div>&#127760; Streamlit (Dashboard)</div>
      <div>&#128225; MQTT (IoT Streaming)</div>
      <div>&#127758; Plotly Maps + Charts</div>
      <div>&#129302; Scikit-Learn (ML Models)</div>
      <div>&#128451; PostgreSQL / CSV</div>
      <div>&#9729;&#65039; Cloudflare Tunnel</div>
    </div></div>""", unsafe_allow_html=True)

st.markdown("""<div class="footer-strip">
<span style="font-family:Orbitron,monospace;font-size:clamp(7px,0.8vw,9px);color:#00d4ff;letter-spacing:2px;">AI ENABLED SMART SURVEILLANCE v5.0</span>
<span style="font-size:clamp(7px,0.8vw,9px);color:#7ab8e8;"> | Python - Streamlit - Plotly - Scikit-Learn</span>
</div>""", unsafe_allow_html=True)

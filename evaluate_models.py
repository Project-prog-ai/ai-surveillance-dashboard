"""
evaluate_models.py — Reproducible Evaluation of the AI Vehicle Threat Assessment Framework
==========================================================================================

This script evaluates every component of the surveillance framework against
a rule-based reference label set.  IMPORTANT: the reference labels are
constructed from the same rule-based scoring logic (Section 3.6 of the
dissertation), so they are NOT an independent ground truth.  All metrics
should be interpreted as measuring agreement with that rule-based proxy,
not against a verified external standard.

Components evaluated:
    1. Majority-class baseline
    2. Rule-based / reference component (self-evaluation)
    3. Isolation Forest (standalone)
    4. DBSCAN spatial component (standalone)
    5. Integrated framework (50/30/20 weighting)
    6. Threshold sensitivity analysis (thresholds 40, 50, 60)
    7. Indicator ablation study

Outputs are written to results/ as CSV tables.

Usage:
    python evaluate_models.py
"""

import os, sys, time, warnings, tracemalloc
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

warnings.filterwarnings("ignore")

# ─── REPRODUCIBILITY ──────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ─── CONFIGURATION ────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "surveillance_features.csv")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

W_RULE = 0.50; W_IF = 0.30; W_DB = 0.20
DEFAULT_THRESHOLD = 50
IF_CONTAMINATION = 0.05
DBSCAN_EPS = 0.30; DBSCAN_MIN_SAMPLES = 15

FLAG_COLS = ["PARKING_ANOMALY","SPEED_ANOMALY","ROUTE_DEVIATION","RESTRICTED_ZONE_ENTRY","COORDINATED_MOVEMENT"]
FLAG_WEIGHTS = {"PARKING_ANOMALY":25,"SPEED_ANOMALY":20,"ROUTE_DEVIATION":20,"RESTRICTED_ZONE_ENTRY":25,"COORDINATED_MOVEMENT":10}

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# ═══════════════════════════════════════════════════════════════════
# 1. DATA LOADING & VALIDATION
# ═══════════════════════════════════════════════════════════════════
def load_and_validate(path):
    log("Loading dataset ...")
    df = pd.read_csv(path)
    log(f"  Shape: {df.shape}")
    missing = df.isnull().sum().sum()
    log(f"  Missing values: {missing}")
    n_dup = df.duplicated().sum()
    log(f"  Duplicate rows: {n_dup}")
    if n_dup > 0:
        df = df.drop_duplicates().reset_index(drop=True)
    if "AVG_SPEED_KMH" in df.columns:
        log(f"  Extreme speeds (>300 km/h): {(df['AVG_SPEED_KMH']>300).sum()}")
    for col in FLAG_COLS:
        if col in df.columns and not set(df[col].unique()).issubset({0,1}):
            log(f"  WARNING: {col} has non-binary values")
    # NOTE: Reference labels derived from SAME rule-based scoring — NOT independent ground truth
    df["GROUND_TRUTH"] = (df["RISK_SCORE"] >= DEFAULT_THRESHOLD).astype(int)
    pos = df["GROUND_TRUTH"].sum()
    log(f"  Reference positives (score>={DEFAULT_THRESHOLD}): {pos} ({pos/len(df)*100:.2f}%)")
    log("  Validation complete.\n")
    return df

# ═══════════════════════════════════════════════════════════════════
# 2. METRICS
# ═══════════════════════════════════════════════════════════════════
def compute_metrics(y_true, y_pred, name):
    tn,fp,fn,tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    return {"Model":name, "Accuracy":round(accuracy_score(y_true,y_pred),4),
            "Precision":round(precision_score(y_true,y_pred,zero_division=0),4),
            "Recall":round(recall_score(y_true,y_pred,zero_division=0),4),
            "F1":round(f1_score(y_true,y_pred,zero_division=0),4),
            "TP":int(tp),"FP":int(fp),"FN":int(fn),"TN":int(tn)}

# ═══════════════════════════════════════════════════════════════════
# 3. MAJORITY-CLASS BASELINE
# ═══════════════════════════════════════════════════════════════════
def evaluate_majority_baseline(df):
    log("Evaluating majority-class baseline (all negative) ...")
    y_true = df["GROUND_TRUTH"]; y_pred = np.zeros(len(df), dtype=int)
    m = compute_metrics(y_true, y_pred, "Majority Baseline")
    log(f"  Acc={m['Accuracy']}, Rec={m['Recall']}, F1={m['F1']}")
    return m

# ═══════════════════════════════════════════════════════════════════
# 4. RULE-BASED (SELF-REFERENTIAL — NOT INDEPENDENT)
# ═══════════════════════════════════════════════════════════════════
def evaluate_rule_based(df):
    """NOTE: Self-evaluation. Labels ARE the rule-based scores."""
    log("Evaluating rule-based component (self-referential, NOT independent) ...")
    score = pd.Series(0.0, index=df.index)
    for f,w in FLAG_WEIGHTS.items():
        if f in df.columns: score += df[f]*w
    y_pred = (score >= DEFAULT_THRESHOLD).astype(int)
    m = compute_metrics(df["GROUND_TRUTH"], y_pred, "Rule-Based (self-referential)")
    log(f"  Acc={m['Accuracy']}, Prec={m['Precision']}, Rec={m['Recall']}, F1={m['F1']}")
    return m

# ═══════════════════════════════════════════════════════════════════
# 5. ISOLATION FOREST
# ═══════════════════════════════════════════════════════════════════
def evaluate_isolation_forest(df):
    log("Evaluating Isolation Forest ...")
    feat = [c for c in ["TOTAL_DISTANCE_KM","AVG_SPEED_KMH","MAX_SPEED_KMH","STD_SPEED",
            "CIRCUITY_RATIO","AVG_BEARING_CHANGE","PARKING_DURATION_MIN","RZ_HIT_COUNT"] if c in df.columns]
    X = StandardScaler().fit_transform(df[feat].values)
    iso = IsolationForest(contamination=IF_CONTAMINATION, random_state=RANDOM_SEED, n_estimators=100)
    iso.fit(X)
    labels = iso.predict(X)
    y_pred = (labels == -1).astype(int)
    m = compute_metrics(df["GROUND_TRUTH"], y_pred, "Isolation Forest")
    log(f"  Anomalies: {y_pred.sum()}, Acc={m['Accuracy']}, Prec={m['Precision']}, Rec={m['Recall']}, F1={m['F1']}")
    return m, y_pred, iso.decision_function(X)

# ═══════════════════════════════════════════════════════════════════
# 6. DBSCAN
# ═══════════════════════════════════════════════════════════════════
def evaluate_dbscan(df):
    log("Evaluating DBSCAN ...")
    X = StandardScaler().fit_transform(df[["AVG_SPEED_KMH","TOTAL_DISTANCE_KM"]].values)
    db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit_predict(X)
    y_pred = (db == -1).astype(int)
    n_clusters = len(set(db)) - (1 if -1 in db else 0)
    m = compute_metrics(df["GROUND_TRUTH"], y_pred, "DBSCAN")
    log(f"  Clusters: {n_clusters}, Noise: {y_pred.sum()}, Acc={m['Accuracy']}, F1={m['F1']}")
    return m, y_pred

# ═══════════════════════════════════════════════════════════════════
# 7. INTEGRATED FRAMEWORK
# ═══════════════════════════════════════════════════════════════════
def evaluate_integrated(df, if_scores, db_pred, threshold=DEFAULT_THRESHOLD):
    log(f"Evaluating integrated framework (threshold={threshold}) ...")
    rule_score = pd.Series(0.0, index=df.index)
    for f,w in FLAG_WEIGHTS.items():
        if f in df.columns: rule_score += df[f]*w
    if_min,if_max = if_scores.min(), if_scores.max()
    if_norm = 100*(1-(if_scores-if_min)/(if_max-if_min)) if if_max!=if_min else np.zeros_like(if_scores)
    integrated = W_RULE*rule_score.values + W_IF*if_norm + W_DB*db_pred*100
    y_pred = (integrated >= threshold).astype(int)
    m = compute_metrics(df["GROUND_TRUTH"], y_pred, f"Integrated (t={threshold})")
    log(f"  Flagged: {y_pred.sum()}, Acc={m['Accuracy']}, Prec={m['Precision']}, Rec={m['Recall']}, F1={m['F1']}")
    return m, integrated

# ═══════════════════════════════════════════════════════════════════
# 8. THRESHOLD SENSITIVITY
# ═══════════════════════════════════════════════════════════════════
def threshold_sensitivity(df, scores):
    log("Threshold sensitivity analysis ...")
    results = []
    for t in [40, 50, 60]:
        y_pred = (scores >= t).astype(int)
        m = compute_metrics(df["GROUND_TRUTH"], y_pred, f"Threshold={t}")
        m["Threshold"] = t
        results.append(m)
        log(f"  t={t}: Acc={m['Accuracy']}, Prec={m['Precision']}, Rec={m['Recall']}, F1={m['F1']}")
    return results

# ═══════════════════════════════════════════════════════════════════
# 9. INDICATOR ABLATION
# ═══════════════════════════════════════════════════════════════════
def indicator_ablation(df, if_scores, db_pred):
    log("Indicator ablation study ...")
    results = []
    # Full model
    full,_ = evaluate_integrated(df, if_scores, db_pred)
    full["Ablation"] = "Full"; results.append(full)
    # Remove one at a time
    for flag in FLAG_COLS:
        if flag not in df.columns: continue
        score = pd.Series(0.0, index=df.index)
        for f,w in FLAG_WEIGHTS.items():
            if f in df.columns and f != flag: score += df[f]*w
        if_min,if_max = if_scores.min(), if_scores.max()
        if_norm = 100*(1-(if_scores-if_min)/(if_max-if_min)) if if_max!=if_min else np.zeros_like(if_scores)
        integrated = W_RULE*score.values + W_IF*if_norm + W_DB*db_pred*100
        y_pred = (integrated >= DEFAULT_THRESHOLD).astype(int)
        m = compute_metrics(df["GROUND_TRUTH"], y_pred, f"Without {flag}")
        m["Ablation"] = f"Remove {flag}"; results.append(m)
    # Individual indicators
    for flag in FLAG_COLS:
        if flag not in df.columns: continue
        m = compute_metrics(df["GROUND_TRUTH"], df[flag].astype(int), f"Only {flag}")
        m["Ablation"] = f"Only {flag}"; results.append(m)
    return results

# ═══════════════════════════════════════════════════════════════════
# 10. TRAIN/TEST SPLIT
# ═══════════════════════════════════════════════════════════════════
def train_test_evaluation(df):
    log("Train/test split evaluation (70/30) ...")
    feat = [c for c in ["TOTAL_DISTANCE_KM","AVG_SPEED_KMH","MAX_SPEED_KMH","STD_SPEED",
            "CIRCUITY_RATIO","AVG_BEARING_CHANGE","PARKING_DURATION_MIN","RZ_HIT_COUNT"] if c in df.columns]
    X = df[feat].values; y = df["GROUND_TRUTH"].values
    X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.30,random_state=RANDOM_SEED,stratify=y)
    sc = StandardScaler(); X_tr_s = sc.fit_transform(X_tr); X_te_s = sc.transform(X_te)
    iso = IsolationForest(contamination=IF_CONTAMINATION, random_state=RANDOM_SEED, n_estimators=100)
    iso.fit(X_tr_s)
    m_tr = compute_metrics(y_tr, (iso.predict(X_tr_s)==-1).astype(int), "IF Train (70%)")
    m_te = compute_metrics(y_te, (iso.predict(X_te_s)==-1).astype(int), "IF Test (30%)")
    log(f"  Train: Acc={m_tr['Accuracy']}, F1={m_tr['F1']}")
    log(f"  Test:  Acc={m_te['Accuracy']}, F1={m_te['F1']}")
    return [m_tr, m_te]

# ═══════════════════════════════════════════════════════════════════
# 11. PERFORMANCE PROFILING
# ═══════════════════════════════════════════════════════════════════
def performance_check(df):
    log("Performance profiling ...")
    results = {"Total Records": len(df)}
    feat = [c for c in ["TOTAL_DISTANCE_KM","AVG_SPEED_KMH","MAX_SPEED_KMH","STD_SPEED",
            "CIRCUITY_RATIO","AVG_BEARING_CHANGE","PARKING_DURATION_MIN","RZ_HIT_COUNT"] if c in df.columns]

    t0=time.time(); pd.read_csv(DATA_PATH); results["Data Loading (s)"]=round(time.time()-t0,3)
    t0=time.time(); X=StandardScaler().fit_transform(df[feat].values); results["Scaling (s)"]=round(time.time()-t0,4)

    tracemalloc.start()
    t0=time.time()
    iso=IsolationForest(contamination=IF_CONTAMINATION,random_state=RANDOM_SEED,n_estimators=100)
    iso.fit(X); iso.predict(X)
    results["IF Time (s)"]=round(time.time()-t0,3)
    _,peak=tracemalloc.get_traced_memory(); tracemalloc.stop()
    results["IF Memory (MB)"]=round(peak/1024/1024,2)

    X_db=StandardScaler().fit_transform(df[["AVG_SPEED_KMH","TOTAL_DISTANCE_KM"]].values)
    tracemalloc.start()
    t0=time.time(); DBSCAN(eps=DBSCAN_EPS,min_samples=DBSCAN_MIN_SAMPLES).fit_predict(X_db)
    results["DBSCAN Time (s)"]=round(time.time()-t0,3)
    _,peak=tracemalloc.get_traced_memory(); tracemalloc.stop()
    results["DBSCAN Memory (MB)"]=round(peak/1024/1024,2)

    for k,v in results.items(): log(f"  {k}: {v}")
    return results

# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    log("="*70)
    log("AI Vehicle Threat Assessment — Model Evaluation Suite")
    log(f"Random Seed: {RANDOM_SEED} | Dataset: {DATA_PATH}")
    log("="*70+"\n")
    total_start = time.time()
    df = load_and_validate(DATA_PATH)
    all_results = []

    all_results.append(evaluate_majority_baseline(df)); print()
    all_results.append(evaluate_rule_based(df)); print()
    if_m,if_pred,if_scores = evaluate_isolation_forest(df); all_results.append(if_m); print()
    db_m,db_pred = evaluate_dbscan(df); all_results.append(db_m); print()
    int_m,int_scores = evaluate_integrated(df, if_scores, db_pred); all_results.append(int_m); print()

    # Comparison table
    log("="*70); log("COMPARISON TABLE"); log("="*70)
    comp = pd.DataFrame(all_results)
    print(comp.to_string(index=False)); comp.to_csv(os.path.join(RESULTS_DIR,"model_comparison.csv"),index=False)
    log("Saved: results/model_comparison.csv\n")

    # Threshold sensitivity
    thresh = threshold_sensitivity(df, int_scores)
    pd.DataFrame(thresh).to_csv(os.path.join(RESULTS_DIR,"threshold_sensitivity.csv"),index=False)
    log("Saved: results/threshold_sensitivity.csv\n")

    # Ablation
    abl = indicator_ablation(df, if_scores, db_pred)
    pd.DataFrame(abl).to_csv(os.path.join(RESULTS_DIR,"indicator_ablation.csv"),index=False)
    log("Saved: results/indicator_ablation.csv\n")

    # Train/test
    split = train_test_evaluation(df)
    pd.DataFrame(split).to_csv(os.path.join(RESULTS_DIR,"train_test_split.csv"),index=False)
    log("Saved: results/train_test_split.csv\n")

    # Performance
    perf = performance_check(df)
    pd.DataFrame([perf]).to_csv(os.path.join(RESULTS_DIR,"performance_metrics.csv"),index=False)
    log("Saved: results/performance_metrics.csv\n")

    log(f"Total time: {round(time.time()-total_start,2)}s")
    log("All results in results/ directory.")

if __name__ == "__main__":
    main()

"""
test_pipeline.py — Unit and integration tests for the AI Vehicle Threat Assessment Framework.
Run with: python -m pytest tests/ -v
"""

import os, sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "surveillance_features.csv")

# ─── FIXTURES ──────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def df():
    return pd.read_csv(DATA_PATH)

@pytest.fixture
def sample_trip():
    """A synthetic trip for unit testing."""
    return pd.Series({
        "PARKING_ANOMALY": 1, "SPEED_ANOMALY": 1, "ROUTE_DEVIATION": 0,
        "RESTRICTED_ZONE_ENTRY": 1, "COORDINATED_MOVEMENT": 0,
        "AVG_SPEED_KMH": 45.0, "MAX_SPEED_KMH": 90.0, "STD_SPEED": 15.0,
        "TOTAL_DISTANCE_KM": 12.5, "DURATION_MIN": 30.0,
        "CIRCUITY_RATIO": 1.8, "AVG_BEARING_CHANGE": 120.0,
        "PARKING_DURATION_MIN": 45.0, "RZ_HIT_COUNT": 2, "RISK_SCORE": 65.0
    })

# ─── 1. DATA INTEGRITY TESTS ──────────────────────────────────────
class TestDataIntegrity:
    def test_dataset_loads(self, df):
        assert len(df) > 0, "Dataset should not be empty"

    def test_no_missing_values(self, df):
        assert df.isnull().sum().sum() == 0, "No missing values expected"

    def test_no_duplicates(self, df):
        dup = df.duplicated().sum()
        assert dup == 0, f"Found {dup} duplicate rows"

    def test_required_columns_exist(self, df):
        required = ["VEHICLE_ID","TRIP_ID","RISK_SCORE","RISK_LEVEL",
                     "AVG_SPEED_KMH","PARKING_ANOMALY","SPEED_ANOMALY",
                     "ROUTE_DEVIATION","RESTRICTED_ZONE_ENTRY"]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_flag_columns_binary(self, df):
        flags = ["PARKING_ANOMALY","SPEED_ANOMALY","ROUTE_DEVIATION",
                 "RESTRICTED_ZONE_ENTRY","COORDINATED_MOVEMENT"]
        for f in flags:
            if f in df.columns:
                assert set(df[f].unique()).issubset({0,1}), f"{f} should be binary"

    def test_risk_score_range(self, df):
        assert df["RISK_SCORE"].min() >= 0, "Risk scores should be >= 0"
        assert df["RISK_SCORE"].max() <= 100, "Risk scores should be <= 100"

    def test_risk_levels_valid(self, df):
        valid = {"LOW","MEDIUM","HIGH"}
        actual = set(df["RISK_LEVEL"].unique())
        assert actual.issubset(valid), f"Unexpected risk levels: {actual - valid}"

    def test_speed_non_negative(self, df):
        assert (df["AVG_SPEED_KMH"] >= 0).all(), "Speeds should be non-negative"

# ─── 2. FEATURE GENERATION TESTS ──────────────────────────────────
class TestFeatureGeneration:
    def test_flag_count_computation(self, df):
        flags = ["PARKING_ANOMALY","SPEED_ANOMALY","ROUTE_DEVIATION",
                 "RESTRICTED_ZONE_ENTRY","COORDINATED_MOVEMENT"]
        existing = [f for f in flags if f in df.columns]
        flag_count = df[existing].sum(axis=1)
        assert flag_count.min() >= 0
        assert flag_count.max() <= len(existing)

    def test_trip_type_derivable(self, df):
        if "TOTAL_DISTANCE_KM" in df.columns:
            types = pd.cut(df["TOTAL_DISTANCE_KM"], bins=[0,5,15,float("inf")],
                          labels=["Short","Standard","Commercial"])
            assert types.notna().sum() > 0

# ─── 3. RISK SCORE CALCULATION TESTS ──────────────────────────────
class TestRiskScoring:
    def test_rule_based_score(self, sample_trip):
        weights = {"PARKING_ANOMALY":25,"SPEED_ANOMALY":20,"ROUTE_DEVIATION":20,
                   "RESTRICTED_ZONE_ENTRY":25,"COORDINATED_MOVEMENT":10}
        score = sum(sample_trip[f]*w for f,w in weights.items())
        assert score == 70, f"Expected 70, got {score}"  # 25+20+0+25+0=70

    def test_zero_flags_zero_score(self):
        weights = {"PARKING_ANOMALY":25,"SPEED_ANOMALY":20,"ROUTE_DEVIATION":20,
                   "RESTRICTED_ZONE_ENTRY":25,"COORDINATED_MOVEMENT":10}
        score = sum(0*w for w in weights.values())
        assert score == 0

    def test_all_flags_max_score(self):
        weights = {"PARKING_ANOMALY":25,"SPEED_ANOMALY":20,"ROUTE_DEVIATION":20,
                   "RESTRICTED_ZONE_ENTRY":25,"COORDINATED_MOVEMENT":10}
        score = sum(1*w for w in weights.values())
        assert score == 100

# ─── 4. RISK CLASSIFICATION TESTS ─────────────────────────────────
class TestRiskClassification:
    def test_low_risk_classification(self):
        assert _classify(10) == "LOW"
        assert _classify(29.9) == "LOW"

    def test_medium_risk_classification(self):
        assert _classify(30) == "MEDIUM"
        assert _classify(54.9) == "MEDIUM"

    def test_high_risk_classification(self):
        assert _classify(55) == "HIGH"
        assert _classify(85) == "HIGH"

    def test_classification_matches_dataset(self, df):
        for _,row in df.sample(100, random_state=42).iterrows():
            expected = row["RISK_LEVEL"]
            computed = _classify(row["RISK_SCORE"])
            assert computed == expected, f"Score {row['RISK_SCORE']}: expected {expected}, got {computed}"

def _classify(score):
    if score >= 55: return "HIGH"
    elif score >= 30: return "MEDIUM"
    else: return "LOW"

# ─── 5. ISOLATION FOREST TESTS ─────────────────────────────────────
class TestIsolationForest:
    def test_if_runs(self, df):
        from sklearn.ensemble import IsolationForest
        feat = ["AVG_SPEED_KMH","TOTAL_DISTANCE_KM"]
        X = df[feat].values[:500]
        iso = IsolationForest(contamination=0.05, random_state=42, n_estimators=50)
        iso.fit(X)
        labels = iso.predict(X)
        assert set(labels).issubset({-1, 1})
        assert (labels == -1).sum() > 0, "IF should detect some anomalies"

    def test_if_reproducible(self, df):
        from sklearn.ensemble import IsolationForest
        X = df[["AVG_SPEED_KMH","TOTAL_DISTANCE_KM"]].values[:500]
        r1 = IsolationForest(contamination=0.05,random_state=42).fit_predict(X)
        r2 = IsolationForest(contamination=0.05,random_state=42).fit_predict(X)
        assert np.array_equal(r1, r2), "IF should be reproducible with same seed"

# ─── 6. DBSCAN TESTS ──────────────────────────────────────────────
class TestDBSCAN:
    def test_dbscan_runs(self, df):
        from sklearn.cluster import DBSCAN
        from sklearn.preprocessing import StandardScaler
        X = StandardScaler().fit_transform(df[["AVG_SPEED_KMH","TOTAL_DISTANCE_KM"]].values[:500])
        labels = DBSCAN(eps=0.3, min_samples=15).fit_predict(X)
        assert len(labels) == 500
        assert -1 in labels, "DBSCAN should find some noise points"

    def test_dbscan_deterministic(self, df):
        from sklearn.cluster import DBSCAN
        from sklearn.preprocessing import StandardScaler
        X = StandardScaler().fit_transform(df[["AVG_SPEED_KMH","TOTAL_DISTANCE_KM"]].values[:500])
        r1 = DBSCAN(eps=0.3, min_samples=15).fit_predict(X)
        r2 = DBSCAN(eps=0.3, min_samples=15).fit_predict(X)
        assert np.array_equal(r1, r2), "DBSCAN should be deterministic"

# ─── 7. INTEGRATED PREDICTION TESTS ───────────────────────────────
class TestIntegratedPrediction:
    def test_weighted_combination(self):
        rule=50; if_score=80; db_score=100
        integrated = 0.50*rule + 0.30*if_score + 0.20*db_score
        assert integrated == 69.0  # 25+24+20

    def test_threshold_logic(self):
        assert (69.0 >= 50) == True   # flagged
        assert (30.0 >= 50) == False  # not flagged

# ─── 8. DASHBOARD DATASET COMPATIBILITY ───────────────────────────
class TestDashboardCompat:
    def test_dashboard_columns_available(self, df):
        """Verify columns needed by app.py exist."""
        needed = ["VEHICLE_ID","RISK_SCORE","RISK_LEVEL","AVG_SPEED_KMH",
                  "PARKING_ANOMALY","SPEED_ANOMALY","ROUTE_DEVIATION",
                  "RESTRICTED_ZONE_ENTRY","IF_LABEL","IF_RESULT","CLUSTER"]
        for c in needed:
            assert c in df.columns, f"Dashboard needs {c}"

    def test_risk_level_distribution(self, df):
        dist = df["RISK_LEVEL"].value_counts()
        assert "LOW" in dist.index
        assert dist["LOW"] > dist.get("HIGH", 0), "LOW should be majority"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])


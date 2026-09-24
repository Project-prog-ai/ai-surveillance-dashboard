# AI-Enabled Smart Surveillance Dashboard

A behavioural analytics and machine learning framework for smart city vehicle surveillance, built as part of a dissertation project.

## Project Structure

```
├── app.py                    # Streamlit dashboard (main application)
├── evaluate_models.py        # Reproducible model evaluation suite
├── run_pipeline.py           # Full pipeline runner (evaluation + tests)
├── requirements.txt          # Python dependencies
├── surveillance_features.csv # Processed dataset (19,599 trips)
├── tests/
│   └── test_pipeline.py      # Unit and integration tests
├── results/                  # Generated evaluation results (CSV)
│   ├── model_comparison.csv
│   ├── threshold_sensitivity.csv
│   ├── indicator_ablation.csv
│   ├── train_test_split.csv
│   └── performance_metrics.csv
├── src/                      # Source modules (reserved)
└── docs/
    ├── AI_Surveillance_Dashboard_Feature_Report.docx
    ├── AI_Surveillance_Dashboard_Technical_Report.docx
    └── AI_Vehicle_Threat_Assessment_Dissertation_v3.docx
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Dashboard

```bash
streamlit run app.py
```

### 3. Run Model Evaluation

```bash
python evaluate_models.py
```

This produces reproducible evaluation results in `results/`:

| File | Contents |
|---|---|
| `model_comparison.csv` | Comparison of all models (Majority Baseline, Rule-Based, IF, DBSCAN, Integrated) |
| `threshold_sensitivity.csv` | Integrated framework at thresholds 40, 50, 60 |
| `indicator_ablation.csv` | Impact of removing/isolating each threat indicator |
| `train_test_split.csv` | 70/30 held-out evaluation of Isolation Forest |
| `performance_metrics.csv` | Execution time and memory usage |

### 4. Run Tests

```bash
python -m pytest tests/ -v
```

### 5. Run Full Pipeline (Evaluation + Tests)

```bash
python run_pipeline.py
```

## Evaluation Components

### Models Evaluated
1. **Majority-class baseline** — Predicts every trip as non-suspicious
2. **Rule-based component** — 5-indicator weighted scoring (self-referential*)
3. **Isolation Forest** — Unsupervised anomaly detection on 8 features
4. **DBSCAN** — Density-based spatial clustering for outlier detection
5. **Integrated framework** — 50% rule-based + 30% IF + 20% DBSCAN

*\*Important: The reference labels are constructed from the same rule-based scoring logic. This is a self-referential evaluation, not an independent validation.*

### Additional Analyses
- **Threshold sensitivity** — Tests thresholds at 40, 50, and 60
- **Indicator ablation** — Removes each of the 5 indicators individually
- **Train/test split** — 70/30 stratified split with no data leakage
- **Performance profiling** — Execution time and memory for each component

## Dashboard Tabs

| Tab | Purpose |
|---|---|
| Overview | Risk distribution, speed vs risk, trip types, flags |
| Temporal | Hourly patterns, day-hour heatmap |
| ML Engine | Interactive IF + DBSCAN analysis |
| Vehicle Timeline | Per-vehicle risk escalation tracking |
| Anomaly Heatmap | Temporal anomaly concentration patterns |
| Risk Trend Analysis | Vehicle risk trend scoring and watch lists |
| Convoy Detection | Multi-vehicle coordinated movement analysis |
| Explorer | Searchable/filterable vehicle database |

## Architecture

The submitted prototype is a **single-page Streamlit application** that reads from a pre-processed CSV dataset. The following components described in the dissertation are **design-intent for production deployment** and are **not implemented** in the submitted code:

- MQTT streaming broker
- FastAPI backend API
- PostgreSQL database
- Real-time vehicle simulator

The Live Simulation Mode in the dashboard demonstrates dynamic data ingestion behaviour using programmatically generated trips, but does not use an actual MQTT broker.

## Reproducibility

- All random operations use `RANDOM_SEED = 42`
- Re-running `evaluate_models.py` produces identical results
- The `run_pipeline.py` script automates the full evaluation sequence
- Test suite validates data integrity, feature generation, scoring logic, and model behaviour

## Dataset

- **Source**: Porto taxi GPS trajectories (public dataset)
- **Records**: 19,599 processed trips
- **Vehicles**: 415 unique identifiers
- **Features**: 24 columns including 5 behavioural threat indicators

# Bijli-Insight Pakistan — Full Project Flow (Start to End)

> This document traces every step from raw data on disk to the live dashboard.  
> Read top to bottom — each section picks up exactly where the previous one left off.

---

## Step 0 — What You Start With

Before any code runs, four files already exist on disk:

```
data/raw/
├── open-meteo-24.85N66.99E7m.csv     ← Karachi  hourly weather, Jan 2024 – Dec 2025
├── open-meteo-31.53N74.39E220m.csv   ← Lahore   hourly weather, Jan 2024 – Dec 2025
├── open-meteo-33.64N73.03E527m.csv   ← Islamabad hourly weather, Jan 2024 – Dec 2025
└── nepra_tariff_slabs_2024.csv       ← NEPRA 2024 residential tariff slab rates
```

Each weather CSV has ~17,544 rows (one per hour for 2 years).  
Each row contains: `time`, `temperature_2m`, `relative_humidity_2m`, `wind_speed_10m`.

These files were fetched from:
- **Open-Meteo** archive API (`archive-api.open-meteo.com`) — CC-BY 4.0
- **NEPRA** tariff schedule (`nepra.org.pk`) — Public Domain

---

## Step 1 — Build the Manifest

**Command:** `python -m backend.pipeline.generate_manifest`  
**File:** `backend/pipeline/generate_manifest.py` → class `ManifestBuilder`

This script answers the question: *"Can we prove these files haven't been tampered with?"*

**What it does:**
1. Scans `data/raw/` for all `open-meteo-*.csv`, `pk_*_weather_*.csv`, and `nepra_*.csv` files.
2. For each file, reads it in 8 KB chunks and computes a **SHA-256 cryptographic hash**.
3. Reads the file to count rows.
4. Assigns the correct `publisher` and `licence` based on the filename prefix.
5. Writes one row per file into `data/raw/manifest.csv`.

**What comes out:**

```
data/raw/manifest.csv
┌─────────────────────────────────┬───────────────────────────────────────────┬──────────┬──────────────────────────────────┐
│ filename                        │ publisher                                 │ row_count│ sha256_hash                      │
├─────────────────────────────────┼───────────────────────────────────────────┼──────────┼──────────────────────────────────┤
│ open-meteo-24.85N66.99E7m.csv   │ Open-Meteo / PMD Weather Archive          │ 17544    │ ab3f...                          │
│ open-meteo-31.53N74.39E220m.csv │ Open-Meteo / PMD Weather Archive          │ 17544    │ c92d...                          │
│ open-meteo-33.64N73.03E527m.csv │ Open-Meteo / PMD Weather Archive          │ 17544    │ 71ae...                          │
│ nepra_tariff_slabs_2024.csv     │ National Electric Power Regulatory Auth.  │ 5        │ 88ff...                          │
└─────────────────────────────────┴───────────────────────────────────────────┴──────────┴──────────────────────────────────┘
```

If any byte in a raw file changes after this point, the next pipeline run will catch it.

---

## Step 2 — Build the Warehouse

**Command:** `python -m backend.pipeline.warehouse_pipeline`  
**File:** `backend/pipeline/warehouse_pipeline.py` → class `DataWarehousePipeline`

This is the core `make data` step. It takes raw CSVs and produces a clean, feature-rich DuckDB database ready for ML.

---

### Step 2a — Layer 1: Verify Raw Files

**Method:** `verify_raw_manifest()`

Before touching a single row of data, the pipeline re-hashes every file in `manifest.csv` and compares the result to the stored hash.

```
manifest.csv row: sha256 = "ab3f..."
Re-hash of file:  sha256 = "ab3f..."  ✅ match → continue
                  sha256 = "XXXX..."  ❌ mismatch → raises ValueError, pipeline stops
```

This guarantees that what you train on is exactly what was originally downloaded.

---

### Step 2b — Layer 2: Cleaned Weather Table

**Method:** `build_cleaned_layer()`  
**Output:** DuckDB table `cleaned_weather`

Reads all three weather CSVs (skipping the Open-Meteo 3-line header), concatenates them, and normalises everything into a consistent schema:

```
Raw column names (messy, vary by file)    →    Standardised names
─────────────────────────────────────────────────────────────────
temperature_2m                            →    temperature_c     (float32)
relative_humidity_2m                      →    humidity_pct      (float32)
wind_speed_10m                            →    wind_speed_kmh    (float32)
time                                      →    timestamp_utc     (UTC timestamp)
(derived from filename)                   →    filename          (city identifier)
```

Two audit columns are added:
- `is_imputed = False` — will be flipped to True if the DQA fills a gap
- `is_anomalous = False` — will be flipped to True if a temperature is physically impossible

**Result:** ~52,632 rows in `cleaned_weather` (3 cities × ~17,544 rows each).

---

### Step 2c — Layer 3: Gold Feature Table

**Method:** `build_processed_gold_layer()`  
**Output:** DuckDB table `gold_weather_features`

This is the table that ML models train on. It adds a stable primary key and computes all lag/rolling features using DuckDB window functions — strictly partitioned per city so Karachi lags never bleed into Lahore rows.

```sql
SELECT
    md5(filename || timestamp_utc::VARCHAR) AS row_id,   ← unique, deterministic primary key
    filename,
    timestamp_utc,
    temperature_c,
    humidity_pct,
    wind_speed_kmh,
    EXTRACT(HOUR FROM timestamp_utc)  AS hour,
    EXTRACT(DOW  FROM timestamp_utc)  AS day_of_week,

    LAG(temperature_c,   1) OVER (PARTITION BY filename ORDER BY timestamp_utc) AS temp_lag_1,
    LAG(temperature_c,  24) OVER (PARTITION BY filename ORDER BY timestamp_utc) AS temp_lag_24,
    LAG(temperature_c, 168) OVER (PARTITION BY filename ORDER BY timestamp_utc) AS temp_lag_168,

    AVG(temperature_c) OVER (
        PARTITION BY filename ORDER BY timestamp_utc
        ROWS BETWEEN 24 PRECEDING AND CURRENT ROW
    ) AS temp_rolling_24
FROM cleaned_weather
```

`temp_lag_168` is the same-hour-last-week value — this doubles as the **t-168 persistence baseline** used in evaluation.

---

## Step 3 — Data Quality Audit (runs inside the hourly pipeline)

**File:** `backend/agents/data_quality_agent.py` → class `DataQualityAgent`

Every new row arriving from the live API is inspected before being stored. The agent runs three rules in sequence — **FIX → FILL → FLAG** — and logs every decision.

```
Incoming DataFrame (1 new observation)
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  ACTION: FIX                                        │
│  Checks if numeric columns are stored as strings.   │
│  "25.4" → 25.4 (float32). Logs TypeConsistencyRule │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│  ACTION: FILL                                       │
│  Counts consecutive NaN values per numeric column.  │
│  If 1 or 2 consecutive NaNs → linear interpolation  │
│  → sets is_imputed = True on filled rows.           │
│  If 3+ NaNs → leaves as NaN (too uncertain to fill) │
│  Logs ShortGapImputationRule                        │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│  ACTION: FLAG                                       │
│  Checks: temp > 55°C or temp < -10°C               │
│  If true → sets is_anomalous = True                 │
│  The original value is KEPT — not altered.          │
│  Logs PhysicalBoundaryRule                          │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
              Cleaned DataFrame returned
              All decisions appended to data/quality_log.json
```

Every entry in `quality_log.json` records: timestamp, column name, rule triggered, action taken, rows affected, and a plain-English rationale.

---

## Step 4 — Train the Quantile Model (first run only)

**File:** `backend/ml/quantile_model.py` → class `QuantileEnergyForecaster`

This runs automatically the first time `backend/pipeline/flow.py` or the Streamlit dashboard is started.

### Why three models instead of one?

A single point forecast (e.g. "tomorrow will be 33°C") is dangerous for grid operators — it gives no warning of uncertainty. Instead, the system trains three separate LightGBM boosters, each minimising a different **pinball loss** function:

```
Pinball Loss for quantile α:
  L(y, ŷ) = max( α × (y − ŷ),  (1−α) × (ŷ − y) )

q10 (α=0.10): penalises over-prediction heavily → learns the lower bound
q50 (α=0.50): symmetric penalty → learns the median (same as MAE minimisation)
q90 (α=0.90): penalises under-prediction heavily → learns the upper bound
```

### Train/test split

```
gold_weather_features (52,632 rows, sorted by time)
│
├── First 80% = Training set   (41,702 rows — Jan 2024 to ~Oct 2025)
└── Last  20% = Test set       (10,426 rows — ~Oct 2025 to Dec 2025)
```

No shuffling. The model never sees future data during training.

### Features used

| Feature | What it captures |
|---|---|
| `hour` | Daily AC usage cycle (peaks 18–22 PKT) |
| `day_of_week` | Weekend vs. weekday demand patterns |
| `humidity_pct` | Felt temperature / cooling load amplifier |
| `wind_speed_kmh` | Natural ventilation (reduces AC need) |
| `temp_lag_1` | Previous hour — inertia in temperature |
| `temp_lag_24` | Same hour yesterday — daily cycle |
| `temp_lag_168` | Same hour last week — weekly pattern |
| `temp_rolling_24` | 24-hour moving average — heatwave context |

### After training

Three model files are saved:
```
models/
├── lgb_quantile_10.txt
├── lgb_quantile_50.txt
└── lgb_quantile_90.txt
```

On every subsequent run, these files are loaded directly — no retraining.

### Non-crossing guarantee

Raw quantile models can occasionally produce `q50 > q90` due to gradient noise. A two-line post-processing step prevents this:
```python
q_mid  = np.maximum(q_low, q_mid)   # q50 can never be below q10
q_high = np.maximum(q_mid, q_high)  # q90 can never be below q50
```

---

## Step 5 — Evaluate the Model

**File:** `backend/ml/model_evaluator.py` → class `ModelFairnessEvaluator`

Three evaluations run on the test set:

### 5a — Baseline Comparison

The declared persistence baseline is: **"predict this hour = same hour last week"** (`temp_lag_168`).

```
Baseline MAE (t-168):  1.8795 °C   ← what a naive rule gives
Model MAE    (q50):    0.4023 °C   ← what LightGBM gives
Improvement:           78.6%
```

The model must beat the baseline. This comparison was declared before training (see `Dataset_Declaration.md`).

### 5b — Quantile Calibration

Checks two things:
1. Are intervals non-crossing for every row? (`q10 ≤ q50 ≤ q90`) → must be True
2. What percentage of real observations fall inside `[q10, q90]`? → target ~80%

### 5c — Group Slice Errors

Breaks the MAE down by operational slice to expose hidden weaknesses:

```
Peak hours   (18:00–22:00 PKT)  vs  Off-peak hours
Heatwave days (temp ≥ 35°C)     vs  Moderate days
Weekdays                         vs  Weekends
```

A single overall MAE can hide a group where the model is dangerously inaccurate.

---

## Step 6 — Hourly Automated Pipeline

**Command:** `python -m backend.pipeline.flow`  
**File:** `backend/pipeline/flow.py`  
**Cron schedule:** `5 * * * *` (minute 5 of every hour, when deployed via Prefect)

This ties everything together into a live, repeating loop:

```
Every hour at :05
        │
        ▼
task_fetch_live_observation()
    Calls: https://api.open-meteo.com/v1/forecast?...
    Returns: 1-row DataFrame with current temperature, humidity, wind
    Fallback: realistic hardcoded values if the API is offline
        │
        ▼
task_run_quality_audit()
    Passes the new row through DataQualityAgent (FIX → FILL → FLAG)
    Appends any decisions to data/quality_log.json
        │
        ▼
task_append_warehouse()
    Inserts the cleaned row into cleaned_weather in DuckDB
    Generates a deterministic row_id = md5(filename + timestamp)
        │
        ▼
task_run_inference_and_log_status()
    Loads gold_weather_features from DuckDB
    Splits 80/20 chronologically
    Trains model on first run (saves to models/) / loads from disk on subsequent runs
    Runs predict_intervals() → gets real q50 predictions on test set
    Computes model_mae, baseline_mae (t-168), improvement_pct
    Writes real computed values to data/pipeline_status.json
    Appends run record to data/prefect_runs.csv
        │
        ▼
     DONE — waits for next hour
```

---

## Step 7 — RAG Knowledge Indexing (first dashboard load)

**File:** `backend/agents/insight_rag_agent.py` → class `InsightAgentRAG`  
**Triggered by:** `get_rag_agent()` cache helper in `frontend/app.py` (runs once per session)

```
knowledge_base/
├── neeca_policy_2022.pdf        ← NEECA National Energy Policy 2022
└── neeca_residential_ecbc.pdf  ← NEECA Residential ECBC-2023
        │
        ▼
ingest_regulatory_pdf()
    pypdf extracts text from each page
    Splits into ~250-word overlapping chunks
    Stores each chunk in ChromaDB with metadata:
        { title, publisher, page_number }
    ChromaDB auto-generates sentence embeddings
    All chunks saved to data/chroma_db/ (persistent — only runs once)
```

From that point on, every user query hits ChromaDB:

```
User query: "What temperature should I set my AC?"
        │
        ▼
retrieve_grounding_context()
    ChromaDB vector similarity search → top 2 matching policy chunks
        │
        ▼
synthesize_verified_advice()
    Found relevant chunks → builds answer:
        "According to NEECA Residential ECBC-2023 (NEECA), evidence indicates:
         '...Section 5.2.1 minimum thermostat setpoint of 26°C...'"
        + DocumentCitation: { title, section, publisher }
        + confidence_score: 0.92

    No relevant chunks found → strict fallback:
        "I don't know based on current verified data and official Pakistan
         energy advisories. Please refer to NEECA ECBC guidelines..."
        + fallback_triggered: True
```

No numbers are invented. Every cited claim comes from a real document chunk stored in ChromaDB.

---

## Step 8 — Streamlit Dashboard

**Command:** `streamlit run frontend/app.py`  
**File:** `frontend/app.py`

The dashboard is a live window into everything built above. It reads from DuckDB, the pipeline status file, the quality log, the manifest, the trained model, and the RAG agent — all in real time.

---

### Sidebar (loads on every render)
- City selector → sets `selected_filename` used by all data queries
- Tariff category selector → feeds Tab 4 calculator
- DuckDB connection status: queries `SHOW TABLES` + `SELECT COUNT(*)` from `cleaned_weather`
- Refresh button: clears `@st.cache_data` and forces re-read

---

### Tab 1 — Operational Health

Reads `data/pipeline_status.json` → shows real computed metrics:
```
Data freshness    → minutes since last pipeline run
FIX / FILL / FLAG → counts from data/quality_log.json
Model MAE         → 0.4023 °C  (real, from last flow run)
Baseline MAE      → 1.8795 °C  (real t-168 persistence)
Improvement       → 78.6%      (real computation)
```

Reads `data/raw/manifest.csv` → renders SHA-256 checksum table with source links.

---

### Tab 2 — DuckDB Warehouse Explorer

Runs a live DuckDB query:
```sql
SELECT * FROM gold_weather_features
WHERE filename = '<selected_city_file>'
ORDER BY timestamp_utc DESC
LIMIT <user_slider>
```

Shows summary stats (avg temp, max temp, avg humidity), the raw dataframe, and a CSV download button.

---

### Tab 3 — 7-Day Quantile Forecasting Studio

```
get_forecaster()            ← loads lgb_quantile_*.txt from models/
        │
load_duckdb_data()          ← fetches last N rows of gold_weather_features for selected city
        │
df.sort_values("timestamp_utc").dropna(subset=FEATURE_COLS)
        │
forecaster.predict_intervals(X_f)
        │
        ├── q_low  → real lower bound (q10) array
        ├── q_mid  → real median (q50) array
        └── q_high → real upper bound (q90) array
        │
st.line_chart(chart_df)     ← three real curves, no random noise
        │
ModelFairnessEvaluator.compute_group_error_slices()
        └── Peak / Off-peak / Heatwave MAEs shown from real predictions
```

---

### Tab 4 — NEPRA Bill Calculator

```
load_tariff_slabs()         ← reads data/raw/nepra_tariff_slabs_2024.csv once
        │
User inputs: monthly_units, num_ac, ac_temp, peak_hours_ac
        │
Lookup slab row where min_units ≤ monthly_units ≤ max_units
        │
base_bill = monthly_units × rate_per_unit_pkr
total_bill = base_bill × 1.22  (+ ~22% govt taxes / FPA surcharges)
        │
NEECA 26°C savings model:
    degrees_below_neeca = max(0, 26 - ac_temp)
    savings_pct = degrees_below_neeca × 6%  (per °C above setpoint)
    potential_saved = total_bill × savings_pct × 0.65  (AC ≈ 65% of summer bill)
```

All rates come from the registered NEPRA CSV — no hardcoded numbers.

---

### Tab 5 — NEECA & NEPRA Policy AI

```
get_rag_agent()             ← ChromaDB client + InsightAgentRAG (cached per session)
        │
User types query
        │
rag.retrieve_grounding_context(query)
        │   ChromaDB vector search → top 2 policy chunks
        │
rag.synthesize_verified_advice(query, context)
        │
        ├── fallback_triggered = True  → st.warning("I don't know...")
        └── fallback_triggered = False → renders narrative + citation badges
                                          with real document title, page, publisher
```

---

## Folder Structure

```
Bijli-Insight-Pakistan/
├── backend/                        ← all Python backend code (importable as backend.*)
│   ├── __init__.py
│   ├── pipeline/
│   │   ├── data_fetcher.py         Step 0/1 — downloads raw CSVs, generates manifest
│   │   ├── generate_manifest.py    Step 1   — SHA-256 audit ledger builder
│   │   ├── warehouse_pipeline.py   Step 2   — 3-layer DuckDB build
│   │   └── flow.py                 Step 6   — Prefect hourly automation
│   ├── agents/
│   │   ├── data_quality_agent.py   Step 3   — FIX/FILL/FLAG DQA protocol
│   │   └── insight_rag_agent.py    Step 7   — ChromaDB RAG for policy queries
│   └── ml/
│       ├── quantile_model.py       Step 4   — LightGBM q10/q50/q90 trainer
│       └── model_evaluator.py      Step 5   — baseline comparison + group slices
├── frontend/
│   └── app.py                      Step 8   — Streamlit dashboard (5 tabs)
├── data/raw/                       source CSVs + manifest.csv
├── models/                         trained lgb_quantile_*.txt files
├── knowledge_base/                 NEECA/NEPRA PDF documents for RAG
├── tests/                          7 contract tests
├── docs/                           current_flow.md, fix.md, hackathonguide.md
├── AI_Use_Log.md                   required submission doc
├── Dataset_Declaration.md          required submission doc
└── Makefile                        make data / make test / make pipeline / make app
```

---

## Full End-to-End Summary

```
DISK                  CODE                          OUTPUT
────────────────────────────────────────────────────────────────────────────
data/raw/*.csv
                →  generate_manifest.py         →  manifest.csv (SHA-256)
                →  warehouse_pipeline.py         →  warehouse.duckdb
                                                      cleaned_weather
                                                      gold_weather_features
                →  quantile_model.py (train)     →  models/lgb_quantile_*.txt
                →  model_evaluator.py            →  real MAE metrics
                →  insight_rag_agent.py (index)  →  data/chroma_db/

EVERY HOUR (Prefect)
Open-Meteo API  →  flow.py task 1               →  1 new observation
                →  flow.py task 2 (DQA)         →  quality_log.json
                →  flow.py task 3               →  cleaned_weather += 1 row
                →  flow.py task 4 (inference)   →  pipeline_status.json
                                                    prefect_runs.csv

DASHBOARD (on demand)
DuckDB          →  app.py Tab 1                 →  live health metrics
DuckDB          →  app.py Tab 2                 →  warehouse data explorer
DuckDB + Model  →  app.py Tab 3                 →  real q10/q50/q90 forecast chart
NEPRA CSV       →  app.py Tab 4                 →  bill calculation + savings advice
ChromaDB        →  app.py Tab 5                 →  grounded policy advisory
```

---

## Step 9 — FastAPI Backend + HTML Frontend (added Session 3)

**Files:** `backend/api.py`, `index.html`

The Streamlit dashboard works great locally but can't be embedded in Vercel's static hosting. To make the polished HTML demo fully functional, a FastAPI layer was added between the Python ML/RAG code and the browser.

### How the FastAPI backend works

**Run:** `uvicorn backend.api:app --port 8000`

```
Browser (index.html on Vercel)
        │
        │  fetch("http://localhost:8000/api/...")
        │  (CORS: allow_origins=["*"])
        │
        ▼
backend/api.py  ─────────────────────────────────────────────────────────
        │
        ├── GET /api/forecast?city=karachi
        │       1. Calls Open-Meteo forecast API → 168 h of temperature/humidity/wind
        │       2. Queries DuckDB gold_weather_features → last 200 rows for lag features
        │       3. Builds feature matrix (hour, dow, humidity, wind, lag1/24/168, roll24)
        │       4. QuantileEnergyForecaster.predict_intervals(X)
        │       5. Returns {lo: [168 floats], md: [168 floats], hi: [168 floats]}
        │
        ├── GET /api/weather?city=karachi
        │       DuckDB query → timestamp, temp, humidity, wind for last 25 rows
        │       + aggregate stats (avg_temp, max_temp, avg_humidity)
        │
        ├── GET /api/manifest
        │       Reads data/raw/manifest.csv → JSON array of {filename, publisher, ...}
        │
        ├── GET /api/status
        │       Reads data/pipeline_status.json → {mae, baseline_mae, total_rows, ...}
        │
        ├── GET /api/tariffs
        │       Reads data/raw/nepra_tariff_slabs_2024.csv
        │       Returns {prot: [[cap, rate], ...], unprot: [...], a2: [...], ind: [...]}
        │
        └── POST /api/chat  {question: "..."}
                InsightAgentRAG.retrieve_grounding_context(question)
                InsightAgentRAG.synthesize_verified_advice(question, context)
                Returns {answer, source, confidence, fallback}
```

### HTTPS → HTTP localhost: why it works

`index.html` is hosted on Vercel (HTTPS). It calls `http://localhost:8000`. Normally HTTPS→HTTP is "mixed content" and browsers block it — **but localhost is a "potentially trustworthy origin"** per the W3C Secure Contexts spec, so all modern browsers (Chrome, Firefox, Safari) allow it without warnings or flags.

### index.html fetch pattern

Every fake data source in the HTML was replaced with a real `fetch()`. The pattern:

```javascript
const API = 'http://localhost:8000';

async function loadForecast() {
    const city = $('#city').value;
    try {
        const r = await fetch(`${API}/api/forecast?city=${encodeURIComponent(city)}`);
        const data = await r.json();
        window.F = {lo: data.lo, md: data.md, hi: data.hi};
    } catch(e) {
        // fallback: sine-wave demo if backend not running
        ...
    }
    draw(); hero(); g3();
}
```

The try/catch fallback means the page still renders a demo chart if the backend is offline — it just shows fake data instead of real data.

---

*Bijli-Insight Pakistan · HackInsight 2026 · Updated 2026-10-21 Session 3*

# Bijli-Insight Pakistan — Current Flow

> Accurate as of 2026-10-21 (Session 3). Everything marked ✅ is verified end-to-end. See `fix.md` for what changed in Sessions 1–2. Session 3 added the FastAPI layer and replaced the Streamlit dashboard with `index.html`.

---

## The Big Picture

HackInsight 2026 submission — national energy intelligence platform for Pakistan's three major grid zones: Karachi (K-Electric), Lahore (LESCO), Islamabad (IESCO). Forecasts electricity demand with calibrated uncertainty bounds and gives NEECA/NEPRA-grounded conservation advice.

**Stack:** Python · FastAPI · DuckDB · LightGBM · ChromaDB · Prefect · HTML/CSS/JS (Vercel) · (Streamlit kept but not primary demo frontend)

**Verified metrics (real, not hardcoded):**
- Model MAE: **0.4023 °C** | Baseline MAE (t-168): **1.8795 °C** | Improvement: **78.6%**
- Tests: **7 / 7 pass**

---

## Data Sources

| File | Publisher | Rows | On track list | Licence |
|---|---|---|---|---|
| `open-meteo-24.85N66.99E7m.csv` | Open-Meteo | 17,544 | Yes | CC-BY 4.0 |
| `open-meteo-31.53N74.39E220m.csv` | Open-Meteo | 17,544 | Yes | CC-BY 4.0 |
| `open-meteo-33.64N73.03E527m.csv` | Open-Meteo | 17,544 | Yes | CC-BY 4.0 |
| `nepra_tariff_slabs_2024.csv` | NEPRA (nepra.org.pk) | 5 | **No** | Public Domain |

All four files are registered in `data/raw/manifest.csv` with SHA-256 checksums.

---

## Data Flow — Step by Step

```
Open-Meteo Archive (3 cities) + NEPRA tariff CSV
       │
       ▼
data/raw/*.csv          ← 4 registered source files
       │
       ▼
ManifestBuilder         ← backend/pipeline/generate_manifest.py
       │                   Scans data/raw/ for open-meteo-*.csv, pk_*_weather_*.csv,
       │                   and nepra_*.csv. Computes SHA-256 per file.
       │                   Writes data/raw/manifest.csv with correct publisher per source.
       ▼
DataWarehousePipeline   ← backend/pipeline/warehouse_pipeline.py
       │
       ├── Layer 1: verify_raw_manifest()
       │     Re-hashes every CSV, compares against manifest.csv.
       │     Raises ValueError on any tamper (byte-level detection).
       │
       ├── Layer 2: build_cleaned_layer()  →  DuckDB table: cleaned_weather
       │     - Reads all CSVs, handles Open-Meteo header (skiprows=3)
       │     - Renames columns to: temperature_c, humidity_pct, wind_speed_kmh
       │     - Parses timestamps to UTC, casts to float32
       │     - Adds is_imputed=False, is_anomalous=False flags
       │
       └── Layer 3: build_processed_gold_layer()  →  DuckDB table: gold_weather_features
             - Primary key: md5(filename || timestamp_utc) → row_id
             - Window features (partitioned per city):
                 temp_lag_1      = previous hour
                 temp_lag_24     = same hour yesterday
                 temp_lag_168    = same hour last week  ← persistence baseline
                 temp_rolling_24 = 24h moving average
             - Extracts hour, day_of_week for ML
```

---

## Hourly Pipeline (Prefect Flow)

**File:** `backend/pipeline/flow.py`
**Trigger:** `5 * * * *` — when deployed to Prefect

```
task_fetch_live_observation(city="karachi")
       │   Calls Open-Meteo forecast API. Falls back to a
       │   realistic hardcoded observation if offline.
       ▼
task_run_quality_audit(df_raw, log_file)
       │   Runs DataQualityAgent.inspect_and_remediate():
       │     1. FIX  — string columns → float32 (TypeConsistencyRule)
       │     2. FILL — ≤2 consecutive NaN → linear interpolation, is_imputed=True
       │     3. FLAG — temp > 55°C or < -10°C → is_anomalous=True (value kept)
       │   Appends all decisions to data/quality_log.json
       ▼
task_append_warehouse(df_clean, db_file)
       │   Inserts the cleaned row into cleaned_weather in DuckDB.
       │   (gold_weather_features is not rebuilt here — cleaned_weather grows live)
       ▼
task_run_inference_and_log_status(db_file, status_file, run_log_file)
       │   ✅ REAL — loads gold_weather_features from DuckDB
       │   Calls QuantileEnergyForecaster.prepare_time_based_split() (80/20)
       │   Trains LightGBM models on first run → saves to models/
       │   Loads saved models on subsequent runs
       │   Calls ModelFairnessEvaluator.compute_declared_baseline()
       │   Computes real model_mae, baseline_mae, improvement_pct
       │   Writes data/pipeline_status.json with actual numbers
       │   Appends to data/prefect_runs.csv
       ▼
     DONE
```

---

## Data Quality Agent — Tri-Action Protocol

**File:** `backend/agents/data_quality_agent.py`

| Action | Rule | Trigger | What happens |
|---|---|---|---|
| **FIX** | TypeConsistencyRule | Column dtype is `object` | Cast to float32, log decision |
| **FILL** | ShortGapImputationRule | 1–2 consecutive NaN | Linear interpolation + bfill/ffill, `is_imputed=True` |
| **FLAG** | PhysicalBoundaryRule | temp > 55°C or < -10°C | `is_anomalous=True`, value preserved unchanged |

Every decision is appended to `data/quality_log.json` as a `QualityDecisionLog` record.

---

## Machine Learning — Quantile Forecaster

**File:** `backend/ml/quantile_model.py`

Three separate LightGBM boosters trained with **pinball loss** (quantile regression):

| Quantile | Meaning | Saved model |
|---|---|---|
| q10 | Floor — minimum demand | `models/lgb_quantile_10.txt` |
| q50 | Median — expected demand | `models/lgb_quantile_50.txt` |
| q90 | Ceiling — peak AC surge | `models/lgb_quantile_90.txt` |

**Features:** `hour`, `day_of_week`, `humidity_pct`, `wind_speed_kmh`, `temp_lag_1`, `temp_lag_24`, `temp_lag_168`, `temp_rolling_24`

**Target:** `temperature_c` (proxy for cooling load)

**Split:** Chronological 80/20 (41,702 train / 10,426 test). No shuffle, no leakage.

**Non-crossing:** `q50 = max(q10, q50)`, `q90 = max(q50, q90)` enforced post-prediction.

**Status:** ✅ Trained. Model files exist in `models/`. Real MAE = 0.4023 °C.

---

## Model Evaluator

**File:** `backend/ml/model_evaluator.py`

1. **Persistence Baseline (t-168):** `ŷ(t) = y(t-168)`. Real baseline MAE = 1.8795 °C.
2. **Quantile Calibration:** Validates `q10 ≤ q50 ≤ q90` + empirical coverage (~80% target).
3. **Group Slice Errors:** Breaks MAE by:
   - Peak (18:00–22:00 PKT) vs. Off-Peak
   - Weekday vs. Weekend
   - Heatwave (≥35°C) vs. Moderate

---

## RAG Insight Agent

**File:** `backend/agents/insight_rag_agent.py`

```
PDF files in knowledge_base/
       │
       ▼
ingest_regulatory_pdf()         ← called once via get_rag_agent() in app.py
       │   pypdf extracts text page by page
       │   Chunks into ~250-word blocks with page metadata
       │   Adds to ChromaDB PersistentClient at data/chroma_db/
       │   Skips if collection.count() > 0 (idempotent)
       ▼
retrieve_grounding_context(query)
       │   ChromaDB vector similarity → top 2 matching policy chunks
       │   Returns: is_grounded, documents, metadatas
       ▼
synthesize_verified_advice(query, context)
       │   Not grounded → "I don't know based on current verified data..."
       │   Grounded → narrative + DocumentCitation list + confidence_score
       └── Returns GroundedInsightPayload (fallback_triggered, cited_documents, ...)
```

**Knowledge base:**
- `knowledge_base/neeca_policy_2022.pdf`
- `knowledge_base/neeca_residential_ecbc.pdf`

**ChromaDB store:** `data/chroma_db/` (created on first `streamlit run`)

---

## FastAPI Backend

**File:** `backend/api.py`  
**Run:** `uvicorn backend.api:app --port 8000`

| Endpoint | Method | What it returns |
|---|---|---|
| `/api/forecast?city=karachi` | GET | Real Open-Meteo + LightGBM q10/q50/q90 for next 168 h |
| `/api/weather?city=karachi` | GET | DuckDB `gold_weather_features` rows + avg/max/humidity |
| `/api/manifest` | GET | `manifest.csv` parsed to JSON array |
| `/api/status` | GET | `pipeline_status.json` (MAE 0.4023, 52,632 rows, freshness) |
| `/api/tariffs` | GET | `nepra_tariff_slabs_2024.csv` parsed as `{prot, unprot, a2, ind}` |
| `/api/chat` | POST `{question}` | `InsightAgentRAG` answer + source + confidence + fallback flag |

CORS is wildcard (`allow_origins=["*"]`) so the Vercel-hosted `index.html` can call it.

**Forecast pipeline inside `/api/forecast`:**
```
Open-Meteo forecast API (free, no key)
    → 168h of future temperature / humidity / wind for requested city
    → Build lag features using DuckDB history (last 200 rows)
    → QuantileEnergyForecaster.predict_intervals(X)
    → Return {lo: [168 floats], md: [168 floats], hi: [168 floats]}
```

---

## HTML Frontend (index.html)

**File:** `index.html` (project root)  
**Deployed:** Vercel (static, no build step)  
**API target:** `http://localhost:8000` (browser localhost exemption allows HTTPS→HTTP for localhost)

| Section | Data source | Status |
|---|---|---|
| 7-day forecast chart (q10/q50/q90 bands) | `/api/forecast` → real LightGBM | ✅ Real |
| Warehouse data table | `/api/weather` → live DuckDB | ✅ Real |
| Pipeline KPIs (MAE, rows, freshness) | `/api/status` → pipeline_status.json | ✅ Real |
| NEPRA bill calculator | `/api/tariffs` → NEPRA CSV | ✅ Real |
| Policy AI chatbot | `/api/chat` → InsightAgentRAG | ✅ Real |
| Manifest / data lineage table | `/api/manifest` → manifest.csv | ✅ Real |

Fallback: if `/api/forecast` is unreachable (backend not running), chart falls back to sine-wave demo so the page doesn't break.

---

## Streamlit Dashboard (kept, legacy)

**File:** `frontend/app.py`  
**Run:** `streamlit run frontend/app.py`  
Still functional. All 5 tabs still real. Not the primary demo frontend anymore — `index.html` is — but useful for local exploration without the FastAPI layer running.

---

## Everything Is Real Now

### ✅ Fully working end-to-end
- `make data` / `python -m backend.pipeline.warehouse_pipeline` — builds all 3 DuckDB layers from scratch
- SHA-256 manifest with 4 registered sources (2 independent, 1 not on track list)
- `PakistanDataFetcher` — all fetch methods implemented, live API download works
- Data Quality Agent — FIX/FILL/FLAG + `quality_log.json`
- LightGBM quantile model — trained, saved to `models/`, loaded on reuse
- Real MAE + baseline metrics in `pipeline_status.json`
- FastAPI backend — all 6 endpoints return real data
- 7-day forecast — real Open-Meteo + LightGBM (verified: Karachi q50 h0 = 17.71 °C)
- `index.html` — all chart/table/chatbot sections call real API endpoints
- NEPRA bill calculator — reads from `nepra_tariff_slabs_2024.csv`
- Policy AI — real ChromaDB RAG, real fallback on unsupported queries
- `pytest tests/ -v` — 7 / 7 pass

### ⏳ Requires user action (not code)
- **Vercel deploy** — import `Arsalna-Shaikh/Bijli-Insight-Pakistan` at `vercel.com/new`, static site, no build command
- **Run backend during demo** — `uvicorn backend.api:app --port 8000` must be running on the presenter's machine
- **Prefect scheduler** — `python -m backend.pipeline.flow` works locally; needs a live machine for the Tue 4PM–Wed 4PM evidence window
- **Hugging Face Space** — deploy `frontend/app.py` as a public HF Space (required deliverable)
- **Git tag** — `git tag v1.0-final && git push origin v1.0-final`
- **`Summary.pdf`** — 1-page PDF (write and export)
- **Demo video** — 3-minute screen recording, public link

---

## How to Run Each Component

```powershell
# 1. Rebuild DuckDB warehouse + retrain model from scratch
python -m backend.pipeline.warehouse_pipeline

# 2. Rebuild manifest (after adding/changing raw CSVs)
python -m backend.pipeline.generate_manifest

# 3. Run one full hourly pipeline cycle (trains model on first run)
python -m backend.pipeline.flow

# 4. Run all unit tests
pytest tests/ -v

# 5. Launch Streamlit dashboard
streamlit run frontend/app.py
```

---

## File Map

```
Bijli-Insight-Pakistan/
├── data/
│   ├── raw/
│   │   ├── open-meteo-24.85N66.99E7m.csv      ← Karachi weather (17,544 rows)
│   │   ├── open-meteo-31.53N74.39E220m.csv    ← Lahore weather  (17,544 rows)
│   │   ├── open-meteo-33.64N73.03E527m.csv    ← Islamabad weather (17,544 rows)
│   │   ├── nepra_tariff_slabs_2024.csv         ← NEPRA 2024 tariff slabs (5 rows) NEW
│   │   └── manifest.csv                        ← SHA-256 audit ledger (4 sources)
│   ├── warehouse.duckdb                         ← cleaned_weather + gold_weather_features
│   ├── chroma_db/                               ← ChromaDB vector store (created on first run)
│   ├── pipeline_status.json                     ← real MAE metrics (written by flow.py)
│   ├── quality_log.json                         ← DQA audit trail
│   └── prefect_runs.csv                         ← pipeline run history
├── models/
│   ├── lgb_quantile_10.txt                      ← trained q10 booster
│   ├── lgb_quantile_50.txt                      ← trained q50 booster
│   └── lgb_quantile_90.txt                      ← trained q90 booster
├── backend/
│   ├── __init__.py
│   ├── pipeline/
│   │   ├── data_fetcher.py       ← PakistanDataFetcher (all methods implemented)
│   │   ├── generate_manifest.py  ← ManifestBuilder (scans Open-Meteo + NEPRA CSVs)
│   │   ├── warehouse_pipeline.py ← DataWarehousePipeline (3-layer build)
│   │   └── flow.py               ← Prefect hourly flow (real inference, real metrics)
│   ├── agents/
│   │   ├── data_quality_agent.py ← DataQualityAgent (FIX/FILL/FLAG)
│   │   └── insight_rag_agent.py  ← InsightAgentRAG (ChromaDB + DuckDB)
│   └── ml/
│       ├── quantile_model.py     ← QuantileEnergyForecaster (LightGBM pinball)
│       └── model_evaluator.py    ← ModelFairnessEvaluator (baseline + slices)
├── frontend/
│   └── app.py                    ← Streamlit dashboard (legacy, all 5 tabs still real)
├── index.html                    ← primary demo frontend (deploy to Vercel, calls localhost:8000)
├── knowledge_base/
│   ├── neeca_policy_2022.pdf
│   └── neeca_residential_ecbc.pdf
├── tests/
│   ├── test_dqa.py           ← PASS
│   ├── test_fetcher.py       ← PASS
│   ├── test_model.py         ← PASS
│   ├── test_rag.py           ← PASS
│   └── test_warehouse.py     ← PASS
├── docs/
│   ├── current_flow.md       ← this file
│   ├── project_flow.md
│   ├── fix.md
│   └── hackathonguide.md
├── AI_Use_Log.md             ← required submission doc
├── Dataset_Declaration.md    ← required submission doc
├── render.yaml               ← Render blueprint for the API
├── Makefile
└── requirements.txt
```

---

*Bijli-Insight Pakistan · HackInsight 2026 · Updated 2026-10-21 post-fix*

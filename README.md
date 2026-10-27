# Bijli-Insight Pakistan · بجلی بصیرت پاکستان

**National Energy Intelligence & Quantile Load Forecasting System**
HackInsight 2026, Pakistan Energy Track · **Team: Void**

Bijli-Insight turns verified weather telemetry, NEPRA/NEECA policy documents and quantile forecasts into a decision studio for the K-Electric, LESCO and IESCO service areas.

| | |
|---|---|
| **Live dashboard** | https://bijli-insight-pakistan.vercel.app |
| **Live API** | https://bijli-insight-api.onrender.com/api/status |
| **Note** | The API runs on Render's free tier; the first request after idle takes about 50 s |

---

## Team

| Member | Full name | Department | Year | Role |
| --- | --- | --- | --- | --- |
| 1 | Arsalna Shaikh | CSIT | 3rd Year | Backend Development & ML |
| 2 | Amna Faisal | CSIT | 3rd Year | AI, ML and Deployment |
| 3 | Kashan Baig | CSIT | 3rd Year | AI Engineer |
| 4 | Ahmed Raza | CSIT | 3rd Year | Full Stack |

---

## Project Abstract / Overview

Pakistan's grid is strained by heatwaves, evening peak demand (18:00–22:00 PKT) and steep NEPRA residential slabs. Operators and households need **trustworthy data**, **uncertainty-aware forecasts** and **policy-grounded advice**, not generic chatbot answers.

The system is a full stack:

1. Ingest two years (2024–2025) of hourly Open-Meteo weather for Karachi, Lahore and Islamabad: **52,632 rows**.
2. Prove file integrity with SHA-256 hashes in `data/raw/manifest.csv`.
3. Load a three-layer **DuckDB** warehouse: raw → `cleaned_weather` → `gold_weather_features`.
4. Run a **Data Quality Agent** (FIX / FILL / FLAG) with an audit log, `quality_log.json`.
5. Train **LightGBM** pinball-loss models at q10, q50 and q90, evaluated against a declared t−168 persistence baseline.
6. Answer policy questions through a **RAG agent**:
   - It retrieves from ChromaDB over NEECA documents and NEPRA tariffs.
   - A Gemini LLM writes the answer only from the retrieved passages.
   - When the documents don't fully cover an on-topic question, it searches the web (Tavily), prefers NEPRA, DISCO and government sources, and cites them with links. Every answer is labelled: project documents, web research, both, calculation, insufficient evidence, or out of scope.
   - When neither source gives reliable evidence, it says so instead of guessing.
7. Refresh hourly with **Prefect**, serve everything through a **FastAPI** backend, and present results in a **web dashboard**. A Streamlit version is also included.

The dashboard covers:
- operational health and data provenance
- 7-day quantile forecasts with a 3D hourly heat map
- a NEPRA slab bill estimator
- a policy advisor
- a warehouse explorer

---

## Problem Statement

- **Peak-hour stress:** cooling load concentrates between 18:00 and 22:00 PKT and strains DISCO transformers.
- **Uncertain weather–load coupling:** a single-point forecast hides worst-case heat and under-prepares operators.
- **Tariff cliffs:** crossing 200 kWh/month moves a household from protected to unprotected NEPRA slabs.
- **Policy disconnect:** NEECA energy-efficiency policy and building-sector guidance are not wired into everyday tools.
- **Untrusted data:** downstream ML and billing advice fail if raw CSVs are incomplete, mistyped or tampered with.

---

## Objectives

1. Build a reproducible, cryptographically verified raw layer for three Pakistani cities.
2. Enforce deterministic data-quality rules with an audit trail:
   - cast types
   - interpolate gaps of two hours or less
   - flag physical outliers
3. Produce gold features without future leakage:
   - lags of 1, 24 and 168 hours
   - a 24-hour rolling temperature
   - a chronological 80/20 split
4. Forecast with **80% intervals** (q10–q90) and a non-crossing constraint.
5. Beat a declared **t−168 persistence baseline**, with error slices for peak hours, weekends and heatwaves.
6. Ground advice in official documents and refuse unsupported claims.
7. Ship a dashboard that a judge or DISCO analyst can open in a browser, or run locally.

---

## Solution Design & Agentic AI Integration

The system is **multi-agent**, not a single LLM wrapper. Two specialised agents sit on top of a verified warehouse.

### Architecture

```
Open-Meteo API  →  data/raw/*.csv  →  SHA-256 manifest
                                      ↓
                         Data Quality Agent (FIX / FILL / FLAG)
                                      ↓
                    DuckDB: cleaned_weather → gold_weather_features
                                      ↓
              LightGBM q10/q50/q90  +  evaluator (vs t−168 baseline)
                                      ↓
     Insight RAG Agent (ChromaDB + NEECA docs + NEPRA tariffs + Gemini LLM)
                                      ↓
     Prefect hourly flow  →  FastAPI (Render)  →  Web dashboard (Vercel)
                                                  Streamlit studio (alternate)
```

### Agentic components

| Agent | Role | How it decides |
| --- | --- | --- |
| **Data Quality Agent** (`backend/agents/data_quality_agent.py`) | Autonomous remediation | **FIX** object→numeric casts. **FILL** linear interpolation for gaps of at most two hours (`is_imputed`). **FLAG** temperatures outside −10 °C to 55 °C (`is_anomalous`) without rewriting the measurement. Every action is appended to `data/quality_log.json`. |
| **Insight RAG Agent** (`backend/agents/insight_rag_agent.py`) | Grounded policy advisor | Retrieves the most relevant chunks from ChromaDB. A Gemini LLM (default `gemini-3.8-flash`) writes a structured answer **only** from those passages. Citations (document, publisher, page) come from retrieval metadata, never from the model. Gold `row_id`s are included only if DuckDB verifies them. If the documents don't fully cover an on-topic question, it falls back to **Tavily web search** (`backend/agents/web_search.py`): official domains rank first, forums and social media are excluded, and only results the answer actually cites are returned as `web_sources` (URLs come from the search API, never the model). Off-topic questions are declined without searching. With no reliable evidence it says so (`source_pathway: insufficient`), and an LLM outage is reported as an outage, not as missing evidence. |

The agents are **custom Python** with typed payloads (`QualityDecisionLog`, `GroundedInsightPayload`). The stack does **not** use CrewAI, AutoGPT, BabyAGI, LangChain or LangGraph. **Prefect** orchestrates fetch → audit → warehouse → status.

### Knowledge base (`backend/pipeline/build_knowledge_base.py`)

| Source | Format | Note |
| --- | --- | --- |
| NEECA National Energy Policy 2022 | PDF (`knowledge_base/neeca_policy_2022.pdf`) | Text layer, 27 pages |
| NEECA: Energy-Efficient Practices in Building Sector | Scanned PDF, indexed from its OCR transcription (`knowledge_base/neeca_residential_ecbc.ocr.txt`) | 11 pages, page numbers preserved |
| NEPRA Residential Tariff Schedule 2024 | `data/raw/nepra_tariff_slabs_2024.csv`, rendered as text | 5 slabs |

Any source missing from the store is indexed automatically when the API starts.

### Core modules

| Module | Path | Function |
| --- | --- | --- |
| Acquisition | `backend/pipeline/data_fetcher.py` | Open-Meteo archive fetch, SHA-256 |
| Manifest | `backend/pipeline/generate_manifest.py` | Audit ledger `manifest.csv` |
| Warehouse | `backend/pipeline/warehouse_pipeline.py` | Verify hashes; build cleaned + gold tables |
| Orchestration | `backend/pipeline/flow.py` | Hourly Prefect flow (`5 * * * *`) |
| Knowledge base | `backend/pipeline/build_knowledge_base.py` | Index RAG sources into ChromaDB |
| DQA | `backend/agents/data_quality_agent.py` | Tri-action quality protocol |
| RAG | `backend/agents/insight_rag_agent.py` | Retrieval, grounded synthesis, fallback |
| Quantile ML | `backend/ml/quantile_model.py` | LightGBM pinball loss, q10/q50/q90 |
| Evaluation | `backend/ml/model_evaluator.py` | Baseline, coverage, peak/weekend/heatwave MAE |
| API | `backend/api.py` | FastAPI: forecast, weather, status, manifest, tariffs, chat |
| Web dashboard | `index.html` | Deployed on Vercel |
| Streamlit studio | `frontend/app.py` | Alternate UI (Hugging Face Space) |

---

## Data Orchestration Pipeline

**Schedule:** the Prefect flow `bijli_insight_hourly_flow` runs on cron `5 * * * *` (minute 5 of every hour).

**Steps** (`backend/pipeline/flow.py`):

1. **Fetch** the current hour from the Open-Meteo forecast API.
2. **Audit** it with `DataQualityAgent.inspect_and_remediate`, and persist the decisions to `data/quality_log.json`.
3. **Append** the cleaned row to DuckDB `cleaned_weather`, with a deterministic MD5 `row_id`.
4. **Infer and log:** run the quantile models, then write `data/pipeline_status.json` (MAE vs baseline, freshness, record count) and append a row to `data/prefect_runs.csv`.

**Batch rebuild from scratch:**

```bash
python -m backend.pipeline.generate_manifest       # hash raw CSVs → manifest.csv
python -m backend.pipeline.warehouse_pipeline      # verify hashes; build cleaned + gold tables
python -m backend.pipeline.build_knowledge_base    # rebuild the RAG index
```

Gold features are `temp_lag_1`, `temp_lag_24`, `temp_lag_168`, `temp_rolling_24`, `hour` and `day_of_week`. They are partitioned by source file, so one city never leaks into another.

---

## Tools, APIs & Technology Stack Used

Submission checklist: only items **used in this repository** are marked.

| Category | Options from brief | Used in this project |
| --- | --- | --- |
| **Programming languages** | Python, R, Java, C++ | **Python 3.12** (deployed) / 3.13 (local), plus HTML/CSS/JavaScript for the dashboard |
| **Frameworks / libraries** | TensorFlow, PyTorch, LangChain, pandas, numpy, LightGBM, scikit-learn, pydantic, pypdf | **pandas, numpy, LightGBM, pypdf, FastAPI, OpenAI SDK (Gemini OpenAI-compatible endpoint), requests/httpx**. Not TensorFlow, PyTorch or LangChain. |
| **Databases** | PostgreSQL, MySQL, MongoDB | **DuckDB** (analytical warehouse) and **ChromaDB** (vector store). Not PostgreSQL, MySQL or MongoDB. |
| **Cloud platforms** | AWS, Azure, Google Cloud Platform | None of the three. Hosting: **Render** (API) and **Vercel** (dashboard). LLM inference: **Google Gemini API**. |
| **BI / visualization** | Power BI, Tableau, Streamlit | **Custom web dashboard** (SVG charts, three.js 3D heat map) and **Streamlit** |
| **Orchestration** | Apache Airflow, Other | **Prefect** (not Airflow) |
| **Agentic AI frameworks** | CrewAI, AutoGPT, BabyAGI, LangGraph | **Custom agents** (DQA + Insight RAG Agent). Not CrewAI, AutoGPT, BabyAGI or LangGraph. |

### APIs and data sources

- [Open-Meteo Archive API](https://archive-api.open-meteo.com/v1/archive): historical hourly temperature, humidity and wind (CC-BY 4.0).
- [Open-Meteo Forecast API](https://api.open-meteo.com/v1/forecast): live observations for the Prefect flow, and forecast inputs for `/api/forecast`.
- [NEPRA tariff schedule](https://nepra.org.pk/tariff/schedule): 2024 residential slabs (`data/raw/nepra_tariff_slabs_2024.csv`).
- NEECA documents in `knowledge_base/`.
- Google Gemini API: LLM synthesis for the RAG agent (falls back to `gemini-3.5-flash-lite` when the main model is rate-limited).
- [Tavily Search API](https://tavily.com): web-search fallback for questions the knowledge base doesn't cover.

### Project API (`backend/api.py`)

| Endpoint | Returns |
| --- | --- |
| `GET /api/forecast?city=karachi` | 168 hourly q10 / q50 / q90 values |
| `GET /api/weather?city=&table=&limit=` | Warehouse rows and summary statistics |
| `GET /api/status` | Last pipeline run: MAE, baseline, improvement, freshness, records |
| `GET /api/manifest` | Source files with licence, row count, SHA-256 |
| `GET /api/tariffs` | NEPRA slab rates |
| `POST /api/chat` `{question}` | Markdown answer, document sources, `pathway`, `web_sources` (title, URL, date, official), fallback flag |
| `GET /health` | Liveness |

---

## Results & Findings

| Result | Evidence |
| --- | --- |
| **Model MAE (q50): 0.4023 °C** | `data/pipeline_status.json`, served live at `/api/status` |
| **Baseline MAE (t−168): 1.8795 °C** | Same file; baseline declared before training (`Dataset_Declaration.md`) |
| **78.6% error reduction vs baseline** | Same file |
| **52,632 verified hourly rows** | 17,544 hours × 3 cities, in `manifest.csv` and DuckDB |
| **Leak-free split** | Chronological 80/20: 41,702 train / 10,426 test rows |
| **Integrity gate** | The warehouse refuses to build if any SHA-256 differs from the manifest |
| **Quantile contract** | Tests assert q10 ≤ q50 ≤ q90 |
| **RAG grounding** | In-scope questions answered with citations (e.g. protected slabs → NEPRA schedule; home insulation → NEECA building guidance). Out-of-scope questions (e.g. "electricity cost in 2045") return "I don't know". |
| **Test suite** | 23 tests passing (`pytest tests/`) |

The evaluator slices error for Pakistan's operating conditions: peak hours (18:00–22:00 PKT), weekends vs weekdays, and heatwave days (≥ 35 °C).

---

## How does your project enhance business intelligence or decision-making?

- **DISCO / grid planners** see a **range** (q10–q90), not a single temperature, so they can plan peak AC stress against an 80% band instead of a falsely precise point.
- **Slice evaluation** shows whether error grows on heatwave evenings, the hours that actually drive load-shedding risk.
- **Households and SMEs** get a NEPRA slab estimate that shows the protected-to-unprotected cliff at 200 units, plus an appliance-based usage builder. Together these link kWh behaviour to rupee outcomes.
- **Compliance and audit** can trace every raw file (hash), every data-quality action (JSON log) and every pipeline run (CSV). That answers the BI question "why did the number change?"
- **Policy answers** cite the NEECA or NEPRA document and page, so consumer-facing advice does not contain hallucinated tariffs.

The product is a **decision studio**: warehouse, forecast, tariff and regulation on one screen, not a disconnected notebook.

---

## Future Scope / Improvements

1. Retrain the LightGBM models on a schedule from new `gold_weather_features` rows, with automatic model versioning.
2. Join actual DISCO load / feeder MW (where licensed) instead of using weather as a proxy for cooling stress.
3. Add Time-of-Use tariffs and full NEPRA S.R.O. tables as structured data.
4. Run the Prefect flow on a managed scheduler (VM or GitHub Actions), and add authentication for DISCO analyst groups.
5. Expand the RAG agent to Urdu queries and more NEECA/NEPRA circulars, and log citation usage for audit.
6. Alerting: SMS or email when the q90 path implies peak-hour transformer overload in a selected city.
7. Drift monitors: if residuals between live Open-Meteo data and the gold layer exceed a threshold, auto-FLAG and pause model publishing.

---

## Project layout

```
Bijli-Insight_Pakistan/
├── index.html                 # Web dashboard (Vercel)
├── backend/
│   ├── api.py                 # FastAPI service (Render)
│   ├── agents/                # Data Quality Agent + Insight RAG Agent
│   ├── pipeline/              # fetch, manifest, warehouse, Prefect flow, knowledge base
│   └── ml/                    # quantile forecaster + evaluator
├── frontend/app.py            # Streamlit studio (Hugging Face Space)
├── data/
│   ├── raw/                   # Open-Meteo CSVs, NEPRA tariffs, manifest.csv
│   ├── warehouse.duckdb       # cleaned + gold layers
│   ├── chroma_db/             # RAG vector index
│   └── pipeline_status.json · prefect_runs.csv · quality_log.json
├── models/                    # trained LightGBM q10 / q50 / q90
├── knowledge_base/            # NEECA documents (+ OCR transcription)
├── tests/                     # pytest suite
├── docs/                      # architecture walkthroughs, product brief, design system
├── render.yaml · vercel.json  # deployment
├── requirements.txt           # full deps (incl. Streamlit) · requirements-render.txt: API only
└── Makefile
```

---

## How to run

**Online:** open https://bijli-insight-pakistan.vercel.app. Nothing to install.

**Locally:**

```powershell
git clone https://github.com/AmnaFaisal1305/Bijli-Insight_Pakistan.git
cd Bijli-Insight_Pakistan
python -m venv .venv
.\.venv\Scripts\Activate.ps1            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt pytest prefect
echo GEMINI_API_KEY=your_key > .env       # optional: enables LLM answers in the advisor
echo TAVILY_API_KEY=your_key >> .env    # optional: enables the web-search fallback
# optional tuning: RAG_MAX_DISTANCE (Chroma distance cutoff), WEB_MIN_SCORE (default 0.3), GEMINI_MODEL, GEMINI_FALLBACK_MODEL

python -m backend.pipeline.generate_manifest
python -m backend.pipeline.warehouse_pipeline
python -m pytest tests/ -v
python -m backend.pipeline.flow
uvicorn backend.api:app --port 8000     # then open index.html in a browser
streamlit run frontend/app.py           # alternate Streamlit studio
```

Makefile targets: `make data`, `make test`, `make pipeline`, `make app`.

---

## Submission documents

- [`Dataset_Declaration.md`](Dataset_Declaration.md): data sources, licences, declared baseline
- [`AI_Use_Log.md`](AI_Use_Log.md): how AI tools were used and what the team verified
- [`docs/`](docs/): architecture and file-by-file walkthroughs, product brief, design system

---

## License and data notice

Weather series are from Open-Meteo (CC-BY 4.0). Policy documents remain the property of NEECA and NEPRA / the Government of Pakistan; this repository uses them for retrieval and citation only. Forecasts are decision support, not a substitute for official NEPRA tariff notifications.

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import duckdb
import pandas as pd
import numpy as np
import httpx
import csv
import json
from pathlib import Path

from backend.ml.quantile_model import QuantileEnergyForecaster
from backend.agents.insight_rag_agent import InsightAgentRAG
from backend.pipeline.build_knowledge_base import ensure_knowledge_base
import chromadb

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def allow_private_network(request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Private-Network"] = "true"
    return response

DB_PATH    = Path("data/warehouse.duckdb")
MODELS_DIR = Path("models")
CHROMA_DIR = Path("data/chroma_db")
RAW_DIR    = Path("data/raw")
FORECAST_INPUTS = Path("data/forecast_inputs")

CITY_MAP = {
    "karachi":   {"file": "open-meteo-24.85N66.99E7m.csv",   "lat": 24.85, "lon": 66.99},
    "lahore":    {"file": "open-meteo-31.53N74.39E220m.csv",  "lat": 31.53, "lon": 74.39},
    "islamabad": {"file": "open-meteo-33.64N73.03E527m.csv",  "lat": 33.64, "lon": 73.03},
}

forecaster    = QuantileEnergyForecaster(models_dir=MODELS_DIR)
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
rag_agent     = InsightAgentRAG(chroma_client=chroma_client, max_policy_results=6)
ensure_knowledge_base(rag_agent)  # Render's disk is fresh on each deploy


@app.get("/health")
def health():
    return {"status": "ok"}


def _conn():
    return duckdb.connect(str(DB_PATH), read_only=True)


def _city_key(city: str) -> str:
    return city.lower().split(" (")[0].split()[0]


@app.get("/api/forecast")
async def forecast(city: str = "karachi"):
    cfg = CITY_MAP.get(_city_key(city), CITY_MAP["karachi"])

    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={cfg['lat']}&longitude={cfg['lon']}"
        f"&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
        f"&forecast_days=7&timezone=Asia%2FKarachi"
    )
    conn = _conn()
    lag_rows = conn.execute(
        "SELECT temperature_c, humidity_pct, wind_speed_kmh FROM gold_weather_features "
        "WHERE filename = ? ORDER BY timestamp_utc DESC LIMIT 200",
        [cfg["file"]]
    ).fetchall()
    conn.close()
    hist = [row[0] for row in reversed(lag_rows)]  # oldest first

    hourly = None
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            for _ in range(2):
                hourly = (await client.get(url)).json().get("hourly")
                if hourly:
                    break
    except (httpx.HTTPError, ValueError):
        pass
    snapshot = FORECAST_INPUTS / f"{cfg['file']}.json"
    if not hourly and snapshot.exists():
        # Open-Meteo rate-limits Render's shared IP; use the latest committed 16-day Open-Meteo forecast,
        # starting from the current PKT hour so a snapshot from an earlier day stays aligned.
        snap  = json.loads(snapshot.read_text(encoding="utf-8"))
        now   = pd.Timestamp.now("Asia/Karachi").strftime("%Y-%m-%dT%H:00")
        start = next((i for i, t in enumerate(snap["time"]) if t >= now), len(snap["time"]))
        if len(snap["time"]) - start >= 168:
            hourly = {k: v[start:] for k, v in snap.items()}
            source = "open-meteo-snapshot"
    elif hourly:
        source = "open-meteo"
    if hourly:
        temps    = hourly["temperature_2m"][:168]
        humidity = hourly["relative_humidity_2m"][:168]
        wind     = hourly["wind_speed_10m"][:168]
    else:
        # no forecast at all: reuse the last observed week as weather inputs
        source   = "warehouse"
        week     = list(reversed(lag_rows[:168]))
        temps    = [float(r[0]) for r in week]
        humidity = [float(r[1]) for r in week]
        wind     = [float(r[2]) for r in week]

    rows = []
    for i in range(168):
        hr  = i % 24
        dow = (i // 24) % 7

        def prev_temp(offset):
            if i >= offset:
                return temps[i - offset]
            idx = len(hist) - (offset - i)
            return float(hist[idx]) if 0 <= idx < len(hist) else float(temps[0])

        lag1   = prev_temp(1)
        lag24  = prev_temp(24)
        lag168 = prev_temp(168)
        roll24 = float(np.mean([prev_temp(j) for j in range(1, min(25, i + 2))]))
        rows.append([hr, dow, humidity[i], wind[i], lag1, lag24, lag168, roll24])

    X     = pd.DataFrame(rows, columns=forecaster.FEATURE_COLS)
    preds = forecaster.predict_intervals(X)

    return {
        "lo": [round(v, 2) for v in preds["q_low"].tolist()],
        "md": [round(v, 2) for v in preds["q_mid"].tolist()],
        "hi": [round(v, 2) for v in preds["q_high"].tolist()],
        "source": source,
    }


@app.get("/api/weather")
def weather(city: str = "karachi", table: str = "gold_weather_features", limit: int = 25):
    cfg = CITY_MAP.get(_city_key(city), CITY_MAP["karachi"])
    db_table = {
        "gold_weather_features": "gold_weather_features",
        "silver_weather_clean":  "cleaned_weather",
        "raw_open_meteo":        "cleaned_weather",
    }.get(table, "gold_weather_features")

    conn  = _conn()
    rows  = conn.execute(
        f"SELECT timestamp_utc, temperature_c, humidity_pct, wind_speed_kmh "
        f"FROM {db_table} WHERE filename = ? ORDER BY timestamp_utc DESC LIMIT ?",
        [cfg["file"], limit]
    ).fetchall()
    stats = conn.execute(
        f"SELECT AVG(temperature_c), MAX(temperature_c), AVG(humidity_pct) "
        f"FROM {db_table} WHERE filename = ?",
        [cfg["file"]]
    ).fetchone()
    conn.close()

    return {
        "rows": [
            {
                "time":     str(r[0]),
                "temp":     round(float(r[1]), 1),
                "humidity": round(float(r[2]), 0),
                "wind":     round(float(r[3]), 1),
            }
            for r in rows
        ],
        "stats": {
            "avg_temp":     round(float(stats[0]), 1),
            "max_temp":     round(float(stats[1]), 1),
            "avg_humidity": round(float(stats[2]), 0),
        },
    }


@app.get("/api/manifest")
def manifest():
    with open(RAW_DIR / "manifest.csv") as f:
        return list(csv.DictReader(f))


@app.get("/api/status")
def status():
    status_file = Path("data/pipeline_status.json")
    if status_file.exists():
        with open(status_file) as f:
            return json.load(f)
    conn  = _conn()
    total = conn.execute("SELECT COUNT(*) FROM gold_weather_features").fetchone()[0]
    conn.close()
    # No pipeline run recorded yet: report that honestly instead of placeholder metrics.
    return {"status": "NO_RUN_YET", "active_records_count": total}


@app.get("/api/quality_log")
def quality_log(limit: int = 250):
    log_file = Path("data/quality_log.json")
    records = json.loads(log_file.read_text(encoding="utf-8")) if log_file.exists() else []
    return records[::-1][:limit]  # newest first


@app.get("/api/tariffs")
def tariffs():
    slabs: dict = {"prot": [], "unprot": [], "a2": [[10**9, 42]], "ind": [[10**9, 36]]}
    with open(RAW_DIR / "nepra_tariff_slabs_2024.csv") as f:
        for row in csv.DictReader(f):
            cap  = int(row["max_units"]) if int(row["max_units"]) < 99000 else 10**9
            rate = float(row["rate_per_unit_pkr"])
            key  = "prot" if int(row["max_units"]) <= 200 else "unprot"
            slabs[key].append([cap, rate])
    return slabs


@app.post("/api/chat")
async def chat(body: dict):
    question = body.get("question", "").strip()
    if not question:
        return {"answer": "Please ask a question.", "source": "", "confidence": "0%", "fallback": True}
    result = rag_agent.answer(question)
    sources = [c["document_title"] for c in result["cited_documents"]]
    return {
        "answer":      result["narrative_explanation"],
        # every document the answer was grounded on, not just the top hit
        "source":      " · ".join(dict.fromkeys(sources)),
        "confidence":  f"{result['confidence_score'] * 100:.1f}%",
        "fallback":    result["fallback_triggered"],
        "pathway":     result["source_pathway"],
        "web_sources": result["web_sources"],
    }

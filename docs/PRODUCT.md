# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Single static `index.html` (vanilla HTML/CSS/JS, CDN libraries only) deployed on Vercel with no build step. Data comes from the FastAPI backend on Render (`https://bijli-insight-api.onrender.com`; `http://localhost:8000` when served locally).

## Users

Primary: HackInsight 2026 judges (Energy Demand track) evaluating the submission, typically in a ~3-minute live demo. They judge technical rigour and credibility: real data, provenance, model-vs-baseline evidence, a live pipeline, grounded AI answers. The product's eventual end users (Pakistani households planning around heat and electricity bills) are the story the judges are asked to believe in, so every screen must still read sensibly to a non-expert.

## Product Purpose

National energy intelligence for Pakistan's grid: forecasts hourly temperature (the cooling-demand proxy) for Karachi, Lahore and Islamabad with LightGBM quantile regression (q10/q50/q90), and turns it into practical guidance: when the heat peaks, when to avoid heavy appliances, what a bill will cost under NEPRA slabs, and what NEECA/NEPRA policy says. Success: judges leave convinced the system is real, rigorous and useful.

## Positioning

Every number on screen is real and traceable: model MAE 0.4023 °C vs 1.8795 °C for the declared t-168 persistence baseline (78.6% better), SHA-256-verified raw sources, a 3-layer DuckDB warehouse, an hourly Prefect pipeline with a data-quality agent (FIX / FILL / FLAG), uncertainty bands rather than single guesses, and a policy advisor that cites document and page, or says "I don't know".

## Operating Context

Shown live by the team on a laptop/projector during judging; also opened by judges on their own devices. The free-tier backend sleeps when idle, so first requests can take ~50 s and loading states are visible in the demo.

## Capabilities and Constraints

Five sections, all must remain: Grid health (plan tomorrow, compare cities, data-quality log, model vs baseline, source files and integrity checks), Data warehouse explorer, 7-day forecast with q10/q50/q90 band plus a three.js 3D heat map, NEPRA bill estimator with appliance picker, Policy & conservation advisor (RAG chat, Markdown answers with citations and a fallback state). City selector, light and dark themes, °C/°F, command palette, all wired to the live API endpoints (`/api/forecast`, `/api/weather`, `/api/status`, `/api/manifest`, `/api/tariffs`, `/api/chat`).

## Brand Commitments

Name "Bijli-Insight Pakistan" with the Urdu wordmark بجلی بصیرت پاکستان in Nastaliq. Light and dark themes both supported.

## Evidence on Hand

Real metrics from `data/pipeline_status.json`, raw-file manifest with SHA-256 checksums (`data/raw/manifest.csv`), NEPRA 2024 tariff slabs, NEECA policy PDFs in `knowledge_base/`. No testimonials, users, partners or adoption figures exist; none may be invented.

## Product Principles

1. Prove, don't claim: show the evidence (baseline comparison, uncertainty, provenance) next to every headline number.
2. Rigour readable by anyone: expert depth is available, but each screen leads with a plain-language takeaway.
3. Honest uncertainty: ranges and "I don't know" are features, never hidden.
4. Credible over flashy: no playful or hype devices; nothing that would undermine trust with judges or regulators.

# Dataset Declaration — Bijli-Insight Pakistan

**Team:** Bijli-Insight  
**Track:** Energy Demand  
**Committed:** Mon 19 Oct 2026 (documented Wed 21 Oct 2026)

---

## Source 1: Open-Meteo Historical Weather Archive

| Field | Value |
|---|---|
| **Publisher** | Open-Meteo (open-meteo.com) |
| **URL** | https://archive-api.open-meteo.com/v1/archive |
| **Licence** | CC-BY 4.0 / Open Access |
| **Update frequency** | Hourly archive; near-realtime forecast API |
| **Coverage** | 2024-01-01 to 2025-12-31, hourly — Karachi (24.86N 67.00E), Lahore (31.52N 74.39E), Islamabad (33.68N 73.04E) |
| **Volume** | ~52,560 rows (3 cities × ~17,520 hourly readings) |
| **Variables** | temperature_2m (°C), relative_humidity_2m (%), wind_speed_10m (km/h) |
| **On track list** | Yes — Open-Meteo listed under Energy demand sources |

---

## Source 2: NEPRA Residential Tariff Schedule 2024

| Field | Value |
|---|---|
| **Publisher** | National Electric Power Regulatory Authority (NEPRA), Government of Pakistan |
| **URL** | https://nepra.org.pk/tariff/schedule |
| **Licence** | Public domain — Government of Pakistan official regulatory document |
| **Update frequency** | Annually (per fiscal year tariff notification) |
| **Coverage** | Residential consumer tariff slabs effective 2024 |
| **Volume** | 5 slab records (`data/raw/nepra_tariff_slabs_2024.csv`) |
| **Variables** | slab_label, min_units, max_units, rate_per_unit_pkr, effective_date |
| **On track list** | **No** — NEPRA is a Pakistani government regulator, not listed as a track source |

---

## Volume Summary

| Check | Status |
|---|---|
| Total rows ≥ 50,000 | ✅ 52,565 rows |
| ≥ 2 years of hourly data | ✅ 2024–2025 continuous |
| ≥ 2 independent sources | ✅ Open-Meteo (weather) + NEPRA (tariff) |
| 1 source not on track list | ✅ NEPRA tariff (Source 2) |
| SHA-256 manifest | ✅ `data/raw/manifest.csv` |

---

## Declared Baseline

**t-168 Persistence Baseline:**
$$\hat{y}(t) = y(t - 168)$$

Predicts this hour's temperature equals the same hour one week ago. Declared before any model training. Improvement measured as:

$$\text{improvement} = \frac{\text{baseline\_MAE} - \text{model\_MAE}}{\text{baseline\_MAE}} \times 100\%$$

---

## Why This Data Fits the Problem

Pakistan's grid load is dominated by residential AC demand during summer peak hours (18:00–22:00 PKT). Hourly dry-bulb temperature across the three major DISCO zones (K-Electric/Karachi, LESCO/Lahore, IESCO/Islamabad) directly drives compressor load. NEPRA tariff data enables consumer-facing bill calculations and incentivises demand-side management aligned with NEECA ECBC-2023 guidelines (26°C thermostat mandate).

---

## Limits and Risks

- No direct electricity load (MW) data available for Pakistan without special NEPRA/NTDC access. Temperature is used as a proxy for cooling demand.
- Open-Meteo data covers weather stations at specific coordinates; microclimate variation within cities is not captured.
- NEPRA tariff rates are effective 2024; future rate changes not reflected.
- Model trained on 2024–2025 historical data; performance may degrade during unprecedented weather extremes.

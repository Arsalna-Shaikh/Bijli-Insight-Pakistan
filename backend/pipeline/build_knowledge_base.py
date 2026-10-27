"""
Location: backend/pipeline/build_knowledge_base.py
Purpose: Index every policy source the RAG advisor may cite into ChromaDB.

Sources:
    - NEECA National Energy Policy 2022 (PDF with a text layer)
    - NEECA building-sector page (scanned PDF; indexed from its committed
      OCR transcription, knowledge_base/neeca_residential_ecbc.ocr.txt)
    - NEPRA 2024 residential tariff slabs (data/raw CSV rendered as text)

ensure_knowledge_base() only ingests sources that are not indexed yet, so the
API can call it on every start (Render's disk is ephemeral) at little cost.

Run directly to (re)build the store:  python -m backend.pipeline.build_knowledge_base
"""

import csv
import re
from pathlib import Path
from typing import List

import chromadb
from pypdf import PdfReader

from backend.agents.insight_rag_agent import InsightAgentRAG

KB_DIR = Path("knowledge_base")
TARIFF_CSV = Path("data/raw/nepra_tariff_slabs_2024.csv")
CHROMA_DIR = Path("data/chroma_db")
NEECA = "National Energy Efficiency & Conservation Authority"
NEPRA = "National Electric Power Regulatory Authority (NEPRA)"

POLICY_TITLE = "NEECA National Energy Policy 2022"
BUILDING_TITLE = "NEECA Energy-Efficient Practices in Building Sector"
TARIFF_TITLE = "NEPRA Residential Tariff Schedule 2024"


def _ocr_pages(path: Path) -> List[str]:
    """Split the OCR transcription on its '=== PAGE n ===' markers."""
    text = path.read_text(encoding="utf-8")
    return re.split(r"^=== PAGE \d+ ===$", text, flags=re.M)[1:]


def _tariff_pages(path: Path) -> List[str]:
    """Render the NEPRA slab CSV as one plain-language page."""
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    lines = [
        "NEPRA residential electricity tariff schedule 2024: slab rates "
        "in Pakistani rupees (Rs) per unit (kWh) of monthly consumption."
    ]
    for r in rows:
        upper = (
            f"above {int(r['min_units']) - 1} units"
            if int(r["max_units"]) >= 99999
            else f"{r['min_units']} to {r['max_units']} units"
        )
        lines.append(
            f"{r['slab_label']}: consumption {upper} per month is charged "
            f"Rs {r['rate_per_unit_pkr']} per unit, effective {r['effective_date']}."
        )
    lines.append(
        "Protected slabs cover residential consumers using up to 200 units a "
        "month; unprotected slabs cover consumption above 200 units."
    )
    lines.append(f"Source: {rows[0]['source']}")
    return ["\n".join(lines)]


def _pdf_pages(path: Path) -> List[str]:
    return [page.extract_text() or "" for page in PdfReader(str(path)).pages]


# (title, publisher, chunk-id prefix, page loader)
SOURCES = [
    (POLICY_TITLE, NEECA, "neeca_policy_2022",
     lambda: _pdf_pages(KB_DIR / "neeca_policy_2022.pdf")),
    (BUILDING_TITLE, NEECA, "neeca_residential_ecbc",
     lambda: _ocr_pages(KB_DIR / "neeca_residential_ecbc.ocr.txt")),
    (TARIFF_TITLE, NEPRA, "nepra_tariff_slabs_2024",
     lambda: _tariff_pages(TARIFF_CSV)),
]


def ensure_knowledge_base(agent: InsightAgentRAG) -> int:
    """Index any source missing from the collection. Returns chunks added."""
    if agent.collection is None:
        return 0
    added = 0
    for title, publisher, doc_id, load_pages in SOURCES:
        if agent.collection.get(where={"title": title}, limit=1)["ids"]:
            continue
        added += agent.ingest_pages(load_pages(), doc_id, title, publisher)
    return added


if __name__ == "__main__":
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection("pakistan_energy_policies")
    except Exception:
        pass
    agent = InsightAgentRAG(chroma_client=client)
    print(f"Indexed {ensure_knowledge_base(agent)} chunks")
    print(f"Collection now holds {agent.collection.count()} chunks")

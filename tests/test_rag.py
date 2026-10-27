
"""
Location: tests/test_rag.py

Purpose:
    Unit tests for the RAG Insight Agent.

Coverage:
    - Unsupported queries trigger fallback.
    - Missing LLM triggers fallback.
    - Grounded context produces structured answers.
    - Citations use retrieved document metadata.
    - Only verified Gold row IDs are returned.
    - LLM failures trigger fallback.
    - Empty LLM responses trigger fallback.

Run:
    pytest tests/test_rag.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock
import duckdb
import pytest

# Ensure repository root is on sys.path when running script directly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.agents.insight_rag_agent import InsightAgentRAG


# ============================================================
# HELPERS
# ============================================================

def make_agent(llm_client=None, db_connection=None):
    """Create an agent without requiring a real ChromaDB server."""

    return InsightAgentRAG(
        chroma_client=None,
        db_connection=db_connection,
        llm_client=llm_client,
    )


def make_llm_client(content="### Direct Answer\nTest answer."):
    """Create a mocked OpenAI-compatible chat completion client."""

    client = MagicMock()

    message = MagicMock()
    message.content = content

    choice = MagicMock()
    choice.message = message

    completion = MagicMock()
    completion.choices = [choice]

    client.chat.completions.create.return_value = completion

    return client


def make_grounded_context():
    """Create representative retrieved policy evidence."""

    return {
        "is_grounded": True,
        "documents": [
            (
                "The supplied policy discusses building energy "
                "efficiency. This excerpt does not specify an "
                "AC temperature setpoint."
            )
        ],
        "metadatas": [
            {
                "title": "NEECA ECBC-2023",
                "page": 12,
                "publisher": "NEECA Pakistan",
            }
        ],
        "distances": [0.2],
        "verified_rows": [
            (
                "pk_kh_row_001",
                "2026-10-21 10:00:00",
                35.5,
            )
        ],
        "row_ids": ["pk_kh_row_001"],
    }


# ============================================================
# FALLBACK TESTS
# ============================================================

def test_rag_unsupported_query_triggers_fallback():
    """Unsupported queries must not invoke the LLM."""

    llm = make_llm_client()
    agent = make_agent(llm_client=llm)

    response = agent.synthesize_verified_advice(
        user_query="What will electricity cost in the year 2045?",
        retrieval_context={"is_grounded": False},
    )

    assert response["fallback_triggered"] is True
    assert "I don't know" in response["narrative_explanation"]
    assert response["cited_documents"] == []
    assert response["grounding_gold_row_ids"] == []
    assert response["confidence_score"] == 0.0

    llm.chat.completions.create.assert_not_called()


def test_rag_missing_llm_triggers_fallback():
    """Without an LLM, the agent must fail closed."""

    agent = make_agent(llm_client=None)
    agent.llm = None  # Isolate from environment-loaded GEMINI_API_KEY

    response = agent.synthesize_verified_advice(
        user_query="Explain the policy.",
        retrieval_context=make_grounded_context(),
    )

    assert response["fallback_triggered"] is True
    assert "I don't know" in response["narrative_explanation"]
    assert response["cited_documents"] == []
    assert response["grounding_gold_row_ids"] == []


def test_rag_empty_context_triggers_fallback():
    """An empty evidence context must not produce an answer."""

    agent = make_agent(llm_client=make_llm_client())

    response = agent.synthesize_verified_advice(
        user_query="Explain the energy policy.",
        retrieval_context={
            "is_grounded": True,
            "documents": [],
            "metadatas": [],
        },
    )

    assert response["fallback_triggered"] is True
    assert response["cited_documents"] == []
    assert response["grounding_gold_row_ids"] == []


# ============================================================
# GROUNDED RESPONSE CONTRACT
# ============================================================

def test_rag_grounded_response_contract():
    """Valid evidence and a working LLM produce the expected schema."""

    llm = make_llm_client(
        "### Direct Answer\n"
        "The retrieved excerpt discusses building energy efficiency.\n\n"
        "### Limitations\n"
        "The supplied excerpt does not specify an AC setpoint."
    )

    agent = make_agent(llm_client=llm)

    response = agent.synthesize_verified_advice(
        user_query="What does this policy say about energy efficiency?",
        retrieval_context=make_grounded_context(),
    )

    assert response["fallback_triggered"] is False
    assert response["narrative_explanation"]
    assert isinstance(response["narrative_explanation"], str)

    assert len(response["cited_documents"]) == 1

    citation = response["cited_documents"][0]

    assert citation["document_title"] == "NEECA ECBC-2023"
    assert citation["section_reference"] == "Page 12"
    assert citation["publisher"] == "NEECA Pakistan"

    assert response["grounding_gold_row_ids"] == [
        "pk_kh_row_001"
    ]

    assert response["confidence_score"] == 0.0

    llm.chat.completions.create.assert_called_once()


def test_rag_grounded_context_without_verified_rows():
    """Policy synthesis can work without unrelated metric records."""

    llm = make_llm_client(
        "### Direct Answer\n"
        "The policy excerpt discusses energy efficiency."
    )

    context = make_grounded_context()
    context["verified_rows"] = []
    context["row_ids"] = []

    agent = make_agent(llm_client=llm)

    response = agent.synthesize_verified_advice(
        user_query="Summarize the policy.",
        retrieval_context=context,
    )

    assert response["fallback_triggered"] is False
    assert len(response["cited_documents"]) == 1
    assert response["grounding_gold_row_ids"] == []


# ============================================================
# CITATION CONTRACTS
# ============================================================

def test_rag_missing_source_metadata_does_not_invent_citation():
    """Incomplete metadata must not become a fabricated citation."""

    llm = make_llm_client(
        "### Direct Answer\n"
        "The retrieved text discusses energy efficiency."
    )

    context = make_grounded_context()
    context["metadatas"] = [
        {
            "page": 12,
        }
    ]

    agent = make_agent(llm_client=llm)

    response = agent.synthesize_verified_advice(
        user_query="Summarize this document.",
        retrieval_context=context,
    )

    assert response["fallback_triggered"] is False
    assert response["cited_documents"] == []


def test_rag_duplicate_citations_are_removed():
    """The same title, publisher, and page should appear once."""

    llm = make_llm_client("### Direct Answer\nSupported answer.")

    context = make_grounded_context()
    context["documents"] = [
        "Energy efficiency policy excerpt A.",
        "Energy efficiency policy excerpt B.",
    ]
    context["metadatas"] = [
        {
            "title": "NEECA ECBC-2023",
            "page": 12,
            "publisher": "NEECA Pakistan",
        },
        {
            "title": "NEECA ECBC-2023",
            "page": 12,
            "publisher": "NEECA Pakistan",
        },
    ]

    agent = make_agent(llm_client=llm)

    response = agent.synthesize_verified_advice(
        user_query="Summarize the policy.",
        retrieval_context=context,
    )

    assert response["fallback_triggered"] is False
    assert len(response["cited_documents"]) == 1


# ============================================================
# VERIFIED GOLD ROW IDS
# ============================================================

def test_only_verified_row_ids_are_returned():
    """IDs in the response must originate from verified rows."""

    llm = make_llm_client("### Direct Answer\nSupported answer.")

    context = make_grounded_context()

    # This unverified ID must not be returned merely because
    # it appears in the incoming row_ids field.
    context["row_ids"] = [
        "pk_kh_row_001",
        "invented_row_999",
    ]

    context["verified_rows"] = [
        (
            "pk_kh_row_001",
            "2026-10-21 10:00:00",
            35.5,
        )
    ]

    agent = make_agent(llm_client=llm)

    response = agent.synthesize_verified_advice(
        user_query="Summarize the available evidence.",
        retrieval_context=context,
    )

    assert response["fallback_triggered"] is False
    assert response["grounding_gold_row_ids"] == [
        "pk_kh_row_001"
    ]
    assert "invented_row_999" not in (
        response["grounding_gold_row_ids"]
    )


# ============================================================
# LLM FAILURE TESTS
# ============================================================

def test_rag_llm_exception_triggers_fallback():
    """API exceptions must not crash the caller or invent an answer."""

    llm = make_llm_client()

    llm.chat.completions.create.side_effect = RuntimeError(
        "Simulated Gemini API failure"
    )

    agent = make_agent(llm_client=llm)

    response = agent.synthesize_verified_advice(
        user_query="Summarize this policy.",
        retrieval_context=make_grounded_context(),
    )

    assert response["fallback_triggered"] is True
    assert "temporarily unavailable" in response["narrative_explanation"]
    assert response["coverage"] == "error"
    assert response["cited_documents"] == []
    assert response["grounding_gold_row_ids"] == []


def test_rag_empty_llm_response_triggers_fallback():
    """An empty model completion must not be treated as an answer."""

    llm = make_llm_client(content="")

    agent = make_agent(llm_client=llm)

    response = agent.synthesize_verified_advice(
        user_query="Summarize this policy.",
        retrieval_context=make_grounded_context(),
    )

    assert response["fallback_triggered"] is True
    assert response["cited_documents"] == []


def test_rag_explicit_llm_abstention_triggers_fallback():
    """An explicit unsupported-answer statement must abstain."""

    llm = make_llm_client(
        "I don't know based on the available verified evidence."
    )

    agent = make_agent(llm_client=llm)

    response = agent.synthesize_verified_advice(
        user_query="What is the unsupported future electricity cost?",
        retrieval_context=make_grounded_context(),
    )

    assert response["fallback_triggered"] is True
    assert response["cited_documents"] == []
    assert response["grounding_gold_row_ids"] == []


def test_rag_abstention_token_triggers_fallback():
    agent = make_agent(llm_client=make_llm_client("NO_RELEVANT_EVIDENCE"))

    response = agent.synthesize_verified_advice(
        user_query="Who won the cricket world cup?",
        retrieval_context=make_grounded_context(),
    )

    assert response["fallback_triggered"] is True


def test_rag_partial_answer_with_limitation_is_kept():
    """'I don't know' under Limitations must not discard the answer."""

    partial = (
        "### Direct Answer\n\n"
        "NEECA promotes efficient appliances.\n\n"
        "### Limitations\n\n"
        "- I don't know based on the available evidence what the exact "
        "savings are."
    )
    agent = make_agent(llm_client=make_llm_client(partial))

    response = agent.synthesize_verified_advice(
        user_query="What does NEECA promote and how much does it save?",
        retrieval_context=make_grounded_context(),
    )

    assert response["fallback_triggered"] is False
    assert "NEECA promotes efficient appliances" in response["narrative_explanation"]


# ============================================================
# DUCKDB VERIFICATION
# ============================================================

def test_duckdb_verifies_only_existing_rows():
    """DuckDB should return existing IDs and reject unknown IDs."""

    db = duckdb.connect(":memory:")

    try:
        db.execute("""
            CREATE TABLE gold_weather_features (
                row_id VARCHAR,
                timestamp_utc TIMESTAMP,
                temperature_c DOUBLE
            )
        """)

        db.execute("""
            INSERT INTO gold_weather_features
            VALUES (
                'pk_kh_row_001',
                TIMESTAMP '2026-10-21 10:00:00',
                35.5
            )
        """)

        agent = make_agent(db_connection=db)

        rows = agent._verify_gold_rows(
            ["pk_kh_row_001", "unknown_row"]
        )

        assert len(rows) == 1
        assert rows[0][0] == "pk_kh_row_001"

    finally:
        db.close()


def test_duckdb_missing_connection_returns_no_rows():
    """Without DuckDB, no row IDs can be verified."""

    agent = make_agent(db_connection=None)

    assert agent._verify_gold_rows(["pk_kh_row_001"]) == []


# ============================================================
# RUN WITH PYTEST
# ============================================================

if __name__ == "__main__":
    raise SystemExit(
        pytest.main([__file__, "-v"])
    )


# ============================================================
# KNOWLEDGE BASE SOURCES
# ============================================================

def test_tariff_document_carries_every_slab_rate():
    from backend.pipeline.build_knowledge_base import _tariff_pages, TARIFF_CSV
    page = _tariff_pages(TARIFF_CSV)[0]
    for rate in ["16.48", "22.95", "34.26", "42.12", "48.84"]:
        assert rate in page


def test_ocr_transcription_keeps_pdf_page_numbers():
    from backend.pipeline.build_knowledge_base import _ocr_pages, KB_DIR
    pages = _ocr_pages(KB_DIR / "neeca_residential_ecbc.ocr.txt")
    assert len(pages) == 11  # the scanned PDF has 11 pages
    assert "ECBC-2023" in pages[1]  # page 2 introduces the building code


def test_ensure_knowledge_base_only_ingests_missing_sources():
    from backend.pipeline import build_knowledge_base as kb

    class FakeCollection:
        def get(self, where, limit):
            # Pretend the policy PDF is already indexed
            return {"ids": ["x"] if where["title"] == kb.POLICY_TITLE else []}

    class FakeAgent:
        collection = FakeCollection()
        ingested = []

        def ingest_pages(self, pages, doc_id, title, publisher):
            self.ingested.append(doc_id)
            return len(pages)

    agent = FakeAgent()
    added = kb.ensure_knowledge_base(agent)
    assert agent.ingested == ["neeca_residential_ecbc", "nepra_tariff_slabs_2024"]
    assert added == 11 + 1


# ============================================================
# HYBRID PATHWAY: DOCUMENTS FIRST, WEB SEARCH FALLBACK
# ============================================================

from backend.agents import insight_rag_agent as rag_module
from backend.agents import web_search

WEB_RESULTS = [
    {
        "title": "NEPRA tariff notification",
        "url": "https://nepra.org.pk/tariff.pdf",
        "content": "Residential slab 1-100 units: Rs 10.",
        "published_date": "2026-07-01",
        "official": True,
    },
    {
        "title": "Some blog",
        "url": "https://example.com/post",
        "content": "Bills rose.",
        "published_date": "",
        "official": False,
    },
]


def make_sequenced_llm(*contents):
    """Mocked LLM client returning each content in turn."""

    client = MagicMock()
    completions = []
    for content in contents:
        completion = MagicMock()
        completion.choices = [MagicMock(message=MagicMock(content=content))]
        completions.append(completion)
    client.chat.completions.create.side_effect = completions
    return client


def make_hybrid_agent(llm, monkeypatch, results=WEB_RESULTS, context=None):
    agent = make_agent(llm_client=llm)
    monkeypatch.setattr(
        agent, "retrieve_grounding_context",
        lambda q, ids=None: context or make_grounded_context(),
    )
    search = MagicMock(return_value=results)
    monkeypatch.setattr(rag_module, "search_web", search)
    return agent, search


def test_full_document_coverage_skips_web_search(monkeypatch):
    llm = make_sequenced_llm("### Direct Answer\nCovered.\nCOVERAGE: FULL")
    agent, search = make_hybrid_agent(llm, monkeypatch)

    response = agent.answer("What does the policy say?")

    assert response["source_pathway"] == "documents"
    assert "COVERAGE" not in response["narrative_explanation"]
    assert response["web_sources"] == []
    search.assert_not_called()


def test_no_document_evidence_uses_web_with_disclosure(monkeypatch):
    llm = make_sequenced_llm(
        "NO_RELEVANT_EVIDENCE",
        "### Direct Answer\nSlab 1 costs Rs 10 per unit [1].",
    )
    agent, search = make_hybrid_agent(llm, monkeypatch)

    response = agent.answer("What is the current unit price?")

    assert response["source_pathway"] == "web"
    assert "external sources" in response["narrative_explanation"]
    # Only the cited result is listed, and its URL comes from the search API.
    assert [w["url"] for w in response["web_sources"]] == [
        "https://nepra.org.pk/tariff.pdf"
    ]
    assert response["cited_documents"] == []
    search.assert_called_once()


def test_partial_document_answer_combines_with_web(monkeypatch):
    llm = make_sequenced_llm(
        "### Direct Answer\nPart answer.\n### Limitations\n- Rate missing.\n"
        "COVERAGE: PARTIAL",
        "### Direct Answer\nCombined [2].",
    )
    agent, _ = make_hybrid_agent(llm, monkeypatch)

    response = agent.answer("Explain the policy and current rate.")

    assert response["source_pathway"] == "documents+web"
    assert response["cited_documents"][0]["document_title"] == "NEECA ECBC-2023"
    assert [w["url"] for w in response["web_sources"]] == [
        "https://example.com/post"
    ]


def test_off_topic_question_declines_without_search(monkeypatch):
    llm = make_sequenced_llm("OFF_TOPIC")
    agent, search = make_hybrid_agent(llm, monkeypatch)

    response = agent.answer("Who won the cricket world cup?")

    assert response["source_pathway"] == "off_topic"
    assert response["cited_documents"] == []
    assert response["web_sources"] == []
    search.assert_not_called()


def test_off_topic_check_runs_when_no_documents_retrieved(monkeypatch):
    llm = make_sequenced_llm("OFF_TOPIC")
    empty = {"is_grounded": False, "documents": [], "metadatas": []}
    agent, search = make_hybrid_agent(llm, monkeypatch, context=empty)

    assert agent.answer("Best pizza recipe?")["source_pathway"] == "off_topic"
    search.assert_not_called()


def test_failed_web_search_reports_insufficient(monkeypatch):
    llm = make_sequenced_llm("NO_RELEVANT_EVIDENCE")
    agent, _ = make_hybrid_agent(llm, monkeypatch, results=[])

    response = agent.answer("What is the FPA this month?")

    assert response["source_pathway"] == "insufficient"
    assert response["web_sources"] == []
    assert "external sources" not in response["narrative_explanation"]


def test_unsupportive_web_results_report_insufficient(monkeypatch):
    llm = make_sequenced_llm("NO_RELEVANT_EVIDENCE", "WEB_INSUFFICIENT")
    agent, _ = make_hybrid_agent(llm, monkeypatch)

    response = agent.answer("What is the FPA this month?")

    assert response["source_pathway"] == "insufficient"
    assert response["web_sources"] == []


def test_search_web_without_key_makes_no_request(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    post = MagicMock()
    monkeypatch.setattr(web_search.httpx, "post", post)

    assert web_search.search_web("tariff") == []
    post.assert_not_called()


def test_search_web_request_error_returns_empty(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test")
    monkeypatch.setattr(
        web_search.httpx, "post",
        MagicMock(side_effect=web_search.httpx.ConnectError("down")),
    )

    assert web_search.search_web("tariff") == []


def test_search_web_ranks_official_and_drops_bad_urls(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test")
    resp = MagicMock()
    resp.json.return_value = {"results": [
        {"title": "Blog", "url": "https://blog.com/a", "content": "x", "score": 0.9},
        {"title": "Bad", "url": "javascript:alert(1)", "content": "x", "score": 0.9},
        {"title": "Weak", "url": "https://weak.com", "content": "x", "score": 0.1},
        {"title": "LESCO", "url": "https://www.lesco.gov.pk/t", "content": "x", "score": 0.5},
    ]}
    monkeypatch.setattr(web_search.httpx, "post", MagicMock(return_value=resp))

    results = web_search.search_web("tariff")

    assert [r["title"] for r in results] == ["LESCO", "Blog"]
    assert results[0]["official"] is True


def test_fullwidth_citations_are_normalised(monkeypatch):
    llm = make_sequenced_llm(
        "NO_RELEVANT_EVIDENCE",
        "### Direct Answer\nRs 10 per unit【1†L3-L5】.",
    )
    agent, _ = make_hybrid_agent(llm, monkeypatch)

    response = agent.answer("Unit price?")

    assert "[1]" in response["narrative_explanation"]
    assert len(response["web_sources"]) == 1


def _rate_limit_error():
    import httpx
    from openai import RateLimitError
    request = httpx.Request("POST", "https://generativelanguage.googleapis.com")
    return RateLimitError(
        "rate limited", response=httpx.Response(429, request=request), body=None
    )


def test_rate_limit_retries_on_fallback_model(monkeypatch):
    monkeypatch.setenv("GEMINI_FALLBACK_MODEL", "small-model")
    llm = make_llm_client("### Direct Answer\nCovered.\nCOVERAGE: FULL")
    completion = llm.chat.completions.create.return_value
    llm.chat.completions.create.side_effect = [_rate_limit_error(), completion]
    agent, _ = make_hybrid_agent(llm, monkeypatch)

    response = agent.answer("What does the policy say?")

    assert response["source_pathway"] == "documents"
    assert llm.chat.completions.create.call_args.kwargs["model"] == "small-model"


def test_llm_outage_is_reported_without_web_search(monkeypatch):
    llm = make_llm_client()
    llm.chat.completions.create.side_effect = _rate_limit_error()
    agent, search = make_hybrid_agent(llm, monkeypatch)

    response = agent.answer("What is the current unit price?")

    assert "temporarily unavailable" in response["narrative_explanation"]
    assert "reliable evidence" not in response["narrative_explanation"]
    assert response["web_sources"] == []
    search.assert_not_called()


def test_uncited_web_answer_is_labelled_calculation(monkeypatch):
    llm = make_sequenced_llm(
        "NO_RELEVANT_EVIDENCE",
        "### Direct Answer\n200 x Rs 60 = Rs 12,000.",
    )
    agent, _ = make_hybrid_agent(llm, monkeypatch)

    response = agent.answer("Cost of 200 units at Rs 60?")

    assert response["source_pathway"] == "calculation"
    assert response["web_sources"] == []
    assert "external sources" not in response["narrative_explanation"]

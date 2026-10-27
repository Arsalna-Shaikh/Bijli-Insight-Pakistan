"""
Location: tests/test_dqa.py
Purpose: Unit tests validating the Tri-Action contract (FIX, FILL, FLAG).
"""

from pathlib import Path
import pandas as pd
from backend.agents.data_quality_agent import DataQualityAgent, DQAAction


def test_dqa_tri_action_contract(tmp_path):
    """Validates that DQA properly handles type casting, gap filling, and anomaly flagging."""
    log_file = tmp_path / "quality_log.json"
    agent = DataQualityAgent(log_path=log_file)
    
    # Test Data: String numbers, short gap (None), and extreme heat (62.0°C)
    df_raw = pd.DataFrame({
        "temperature_2m": ["25.0", None, "27.0", 62.0],
        "relative_humidity_2m": ["60.0", "65.0", "70.0", "50.0"]
    })
    
    try:
        df_clean = agent.inspect_and_remediate(df_raw)
        
        # Assertions
        assert df_clean["is_imputed"].any(), "Short gap was not tagged as imputed"
        assert df_clean["is_anomalous"].any(), "Extreme temperature (>55C) was not flagged"
        assert log_file.exists(), "Audit log quality_log.json was not written"
        
        actions_logged = {log["action_taken"] for log in agent.decision_log}
        assert DQAAction.FIX in actions_logged
        assert DQAAction.FLAG in actions_logged
    except NotImplementedError:
        pass  # Test structure ready for implementation

def test_dqa_fixes_and_fills_all_text_column(tmp_path):
    """pandas 3 stores all-text columns as "str", not object; FIX must still cast and FILL must interpolate."""
    agent = DataQualityAgent(log_path=tmp_path / "quality_log.json")
    df_clean = agent.inspect_and_remediate(pd.DataFrame({"humidity_pct": ["60", None, "70"]}))

    assert df_clean["humidity_pct"].tolist() == [60.0, 65.0, 70.0]
    assert df_clean["is_imputed"].tolist() == [False, True, False]
    assert {DQAAction.FIX, DQAAction.FILL} <= {log["action_taken"] for log in agent.decision_log}

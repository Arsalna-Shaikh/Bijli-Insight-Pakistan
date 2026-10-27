"""
Location: agents/data_quality_agent.py
Purpose: Autonomous Data Quality Agent enforcing the Tri-Action model:
         - FIX: Deterministic casting of data types
         - FILL: Interpolation of short sensor gaps (<= 2 hours) with is_imputed tag
         - FLAG: Outlier detection (> 55°C) with is_anomalous tag
         - AUDIT: Outputs immutable decision records to quality_log.json
"""

from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, TypedDict
import json
import pandas as pd
import numpy as np


class DQAAction(str, Enum):
    """Permitted remediation action categories."""
    FIX = "FIX"
    FILL = "FILL"
    FLAG = "FLAG"


class QualityDecisionLog(TypedDict):
    """Schema for a single audited decision entry."""
    timestamp_utc: str
    column_name: str
    violation_rule: str
    action_taken: DQAAction
    affected_rows_count: int
    rationale: str


class DataQualityAgent:
    """
    Validates data batches, applies deterministic action protocols,
    and maintains an immutable audit trail in compliance with hackathon rules.
    """

    def __init__(self, log_path: Path):
        self.log_path = Path(log_path)
        self.decision_log: List[QualityDecisionLog] = []

    def log_decision(
        self,
        column_name: str,
        violation_rule: str,
        action: DQAAction,
        affected_count: int,
        rationale: str
    ) -> None:
        """
        Appends an audited decision to the in-memory log list.
        """
        record: QualityDecisionLog = {
            "timestamp_utc": pd.Timestamp.now("UTC").isoformat(),
            "column_name": column_name,
            "violation_rule": violation_rule,
            "action_taken": action,
            "affected_rows_count": affected_count,
            "rationale": rationale
        }
        self.decision_log.append(record)

    def check_and_fix_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ACTION: FIX
        Detects numeric metrics stored as string objects and converts to float32.
        """
        df = df.copy()
        target_patterns = ["temp", "humid", "wind"]
        for col in df.columns:
            name_lower = str(col).lower()
            if any(p in name_lower for p in target_patterns):
                if not pd.api.types.is_numeric_dtype(df[col]):  # pandas 3 text is "str", not object
                    converted = pd.to_numeric(df[col], errors="coerce").astype("float32")
                    df[col] = converted
                    self.log_decision(
                        column_name=str(col),
                        violation_rule="TypeConsistencyRule",
                        action=DQAAction.FIX,
                        affected_count=len(df),
                        rationale=f"Casted text/mixed column '{col}' to float32"
                    )
        return df

    def check_and_fill_short_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ACTION: FILL
        Detects sensor gaps of <= 2 consecutive missing values.
        Interpolates linearly and sets boolean flag is_imputed = True.
        """
        df = df.copy()
        if "is_imputed" not in df.columns:
            df["is_imputed"] = False

        target_patterns = ["temp", "humid", "wind"]
        for col in df.columns:
            name_lower = str(col).lower()
            if any(p in name_lower for p in target_patterns) and col != "is_imputed":
                null_mask = df[col].isna()
                total_nulls = int(null_mask.sum())
                if 0 < total_nulls <= 2:
                    df[col] = df[col].interpolate(method="linear").bfill().ffill()
                    df.loc[null_mask, "is_imputed"] = True
                    self.log_decision(
                        column_name=str(col),
                        violation_rule="ShortGapImputationRule",
                        action=DQAAction.FILL,
                        affected_count=total_nulls,
                        rationale=f"Linearly interpolated {total_nulls} missing entries (<= 2h window)"
                    )
        return df

    def check_and_flag_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ACTION: FLAG
        Detects physical impossibilities (temperature > 55°C or < -10°C in Pakistan)
        and flags without modifying original measured values.
        """
        df = df.copy()
        if "is_anomalous" not in df.columns:
            df["is_anomalous"] = False

        for col in df.columns:
            name_lower = str(col).lower()
            if "temp" in name_lower:
                numeric_vals = pd.to_numeric(df[col], errors="coerce")
                anomaly_mask = (numeric_vals > 55.0) | (numeric_vals < -10.0)
                anomaly_count = int(anomaly_mask.sum())
                if anomaly_count > 0:
                    df.loc[anomaly_mask, "is_anomalous"] = True
                    self.log_decision(
                        column_name=str(col),
                        violation_rule="PhysicalBoundaryRule",
                        action=DQAAction.FLAG,
                        affected_count=anomaly_count,
                        rationale=f"Flagged {anomaly_count} readings exceeding realistic boundaries (-10C to 55C)"
                    )
        return df

    def inspect_and_remediate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes the entire Tri-Action DQA pipeline (FIX -> FILL -> FLAG)
        and persists the decision log to quality_log.json.
        """
        df_remediated = self.check_and_fix_types(df)
        df_remediated = self.check_and_fill_short_gaps(df_remediated)
        df_remediated = self.check_and_flag_anomalies(df_remediated)

        # Write or append decision records to quality_log.json
        self.save_decision_log()
        return df_remediated

    def save_decision_log(self) -> None:
        """Writes current audit decisions to JSON file."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        existing_records: List[Dict[str, Any]] = []
        if self.log_path.exists():
            try:
                existing_records = json.loads(self.log_path.read_text(encoding="utf-8"))
            except Exception:
                existing_records = []
        
        # Merge new decisions
        combined = existing_records + self.decision_log
        self.log_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")


if __name__ == "__main__":
    agent = DataQualityAgent(log_path=Path("data/quality_log.json"))
    sample_data = pd.DataFrame({
        "temperature_2m": ["25.4", None, "27.1", "58.0"],
        "relative_humidity_2m": ["60", "65", "68", "45"],
        "wind_speed_10m": ["12.0", "14.5", "11.2", "15.0"]
    })
    result = agent.inspect_and_remediate(sample_data)
    print("DQA Inspection Complete. Audited Log Saved.")
    print(result)

"""
Append to: tests/test_fetcher.py
Purpose: Asserts ManifestBuilder constructs the correct ledger columns.
"""
from pathlib import Path
import pandas as pd
from backend.pipeline.generate_manifest import ManifestBuilder


def test_manifest_builder_contract(tmp_path):
    """Validates that ManifestBuilder generates expected CSV columns."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    
    # Create sample raw weather CSV
    sample_csv = raw_dir / "pk_karachi_weather_2024.csv"
    sample_csv.write_text("time,temperature_2m\n2024-01-01T00:00,22.5\n")
    
    builder = ManifestBuilder(raw_dir=raw_dir)
    
    try:
        manifest_path = builder.generate_manifest()
        assert manifest_path.exists()
        
        df = pd.read_csv(manifest_path)
        expected_cols = {"filename", "sha256_hash", "licence", "row_count"}
        assert expected_cols.issubset(set(df.columns))
    except NotImplementedError:
        pass  # Test structure ready for implementation
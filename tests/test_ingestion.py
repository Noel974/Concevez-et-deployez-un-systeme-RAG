import os
import pandas as pd

def test_ingestion_file_exists():
    assert os.path.exists("data/processed/events.csv")

def test_ingestion_has_rows():
    df = pd.read_csv("data/processed/events.csv")
    assert len(df) > 0

def test_ingestion_uid_present():
    df = pd.read_csv("data/processed/events.csv")
    assert "uid" in df.columns

def test_ingestion_text_for_embedding():
    df = pd.read_csv("data/processed/events.csv")
    assert "text_for_embedding" in df.columns
    assert df["text_for_embedding"].notna().all()

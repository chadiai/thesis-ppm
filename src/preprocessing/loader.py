import pandas as pd
from src import config

def load_data():
    """
    Loads raw data.
    """
    filepath = config.DATA_DIR / config.RAW_FILENAME
    print(f"- Loading: {filepath}")
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    df = pd.read_csv(filepath, encoding='utf-8')
    print(f"- Loaded {len(df):,} rows.")
    return df
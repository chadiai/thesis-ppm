import pandas as pd
from src import config


def load_data(filepath=None):
    """
    Loads raw data and dynamically maps column schemas to ensure compatibility
    across different datasets (e.g., TJSP vs. US Federal Court data).
    """
    print(config.DATA_DIR,config.RAW_FILENAME)
    if filepath is None:
        filepath = config.DATA_DIR / config.RAW_FILENAME

    print(f"- Loading: {filepath}")
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Use low_memory=False to handle the large number of attributes in the new CSV
    df = pd.read_csv(filepath, encoding='utf-8', low_memory=False)

    # Schema Adaptation: Maps the new CSV's columns to the repository's expected columns
    schema_mapping = {
        'ucid': 'lawsuit_id',
        'date_filed': 'date',
        'Activity': 'movement',
        'event_judge': 'judge',
        'case_status': 'status',
        'nature_suit': 'subject_matter',
        'case_type': 'class'
    }

    # Only rename columns that actually exist in the loaded dataframe
    df = df.rename(columns={k: v for k, v in schema_mapping.items() if k in df.columns})

    print(f"- Loaded {len(df):,} rows.")
    return df
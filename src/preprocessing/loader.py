import pandas as pd
from src import config


def load_data(filepath=None):
    """
    Loads raw data and dynamically maps column schemas to ensure compatibility
    across different datasets (e.g., TJSP vs. US Federal Court data).
    """
    if filepath is None:
        filepath = config.DATASET_PATH

    print(f"- Loading: {filepath}")
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    df = pd.read_csv(filepath, encoding='utf-8', engine='pyarrow')

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
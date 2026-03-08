import pandas as pd
from src import config

def clean_data(df):
    """
    Generic data cleaning: Duplicates, Currency, Boolean, Dates, Artifacts.
    """
    df = df.copy()
    initial_rows = len(df)
    print("- Cleaning data...")

    # 1. Remove exact duplicates
    df = df.drop_duplicates()
    if len(df) < initial_rows:
        print(f"  - Dropped {initial_rows - len(df)} duplicate rows.")

    # 2. Clean Currency (Generic)
    if 'claim_amount' in df.columns:
        df['claim_amount'] = (
            df['claim_amount']
            .astype(str)
            .str.replace('.', '', regex=False)
            .str.replace(',', '.', regex=False)
        )
        df['claim_amount'] = pd.to_numeric(df['claim_amount'], errors='coerce').fillna(0)

    # 3. Clean Boolean (Generic)
    if 'digital' in df.columns:
        df['digital'] = (df['digital'].astype(str).str.upper().str.strip() == 'VERDADEIRO').astype(int)

    # 4. Clean Dates
    for col in config.DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')

    # 5. Filter Date Artifacts (Generic Year Range 1990-2030)
    if config.COL_DATE in df.columns:
        mask_valid = (df[config.COL_DATE].dt.year >= 1990) & (df[config.COL_DATE].dt.year <= 2030)
        invalid_count = (~mask_valid).sum()
        if invalid_count > 0:
            print(f"  - Dropped {invalid_count} events with invalid dates.")
            df = df[mask_valid]

    # 6. Sorting
    sort_cols = [c for c in [config.COL_CASE_ID, config.COL_DATE, 'order'] if c in df.columns]
    if len(sort_cols) >= 2:
        df = df.sort_values(by=sort_cols)

    return df

# Removed 'remove_outliers' to avoid data leakage (calculating global percentile)
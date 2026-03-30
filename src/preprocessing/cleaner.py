import pandas as pd
from src import config
import ast


def clean_data(df):
    df = df.copy()

    # 1. Clean Currency dynamically (Any column with 'amount' or 'value' in the name)
    for col in df.columns:
        if 'amount' in col.lower() or 'value' in col.lower():
            df[col] = df[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 2. Clean Boolean dynamically (Map 'VERDADEIRO' to 1, rest to 0)
    for col in df.select_dtypes(include=['object']):
        if df[col].astype(str).str.strip().str.upper().isin(['VERDADEIRO', 'FALSO']).any():
            df[col] = (df[col].astype(str).str.upper().str.strip() == 'VERDADEIRO').astype(int)

    # 3. Auto-detect and parse dates (Columns containing 'date' or 'time')
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')

    # 4. Parse Judge Tuples AND Extract Judge Type
    if config.COL_RESOURCE in df.columns:
        def parse_judge_info(val):
            judge_ids = 'Unknown'
            judge_types = 'Unknown'

            if pd.isna(val) or str(val).strip() == '()':
                return judge_ids, judge_types

            val_str = str(val).strip()
            # If it's the English dataset format: "(('SJ000210', 'District Judge'), ...)"
            if val_str.startswith('(('):
                try:
                    parsed = ast.literal_eval(val_str)

                    # Extract IDs (Index 0 of inner tuple)
                    ids = [item[0] for item in parsed if isinstance(item, tuple) and len(item) > 0]
                    # Extract Types (Index 1 of inner tuple)
                    types = [item[1] for item in parsed if isinstance(item, tuple) and len(item) > 1]

                    if ids:
                        judge_ids = "_".join(ids)
                    if types:
                        # Use sorted/set to avoid "District Judge_District Judge", just keep "District Judge"
                        # Or if mixed: "District Judge_Magistrate Judge"
                        unique_types = sorted(list(set(types)))
                        judge_types = "_".join(unique_types)

                    return judge_ids, judge_types
                except Exception:
                    pass

            # Fallback for the original Brazilian dataset
            return val_str, 'Unknown'

        # Apply parsing and split the results into two columns
        parsed_info = df[config.COL_RESOURCE].apply(parse_judge_info)
        df[config.COL_RESOURCE] = parsed_info.apply(lambda x: x[0])
        df['judge_type'] = parsed_info.apply(lambda x: x[1])

    return df


def cluster_activities_hybrid(df, top_n=20):
    """Dynamically keeps the top N activities, bundles the rest into 'Other'"""
    print(f"- Applying Dynamic Activity Clustering (Keeping top {top_n} exact)...")
    df = df.copy()

    top_activities = df[config.COL_ACTIVITY].value_counts().nlargest(top_n).index.tolist()

    df[config.COL_ACTIVITY] = df[config.COL_ACTIVITY].apply(
        lambda x: x if x in top_activities else "Other/Rare Activity"
    )
    return df
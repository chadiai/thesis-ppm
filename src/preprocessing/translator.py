import pandas as pd
import json
import re
from deep_translator import GoogleTranslator
from src import config


def _clean_string(text):
    """
    Normalizes a string by replacing all hidden spaces, tabs, non-breaking
    spaces (\xa0), and newlines with a single standard space.
    """
    if pd.isna(text):
        return ""
    return re.sub(r'\s+', ' ', str(text)).strip()


def _load_cache():
    if config.TRANSLATION_CACHE_FILE.exists():
        with open(config.TRANSLATION_CACHE_FILE, 'r', encoding='utf-8') as f:
            raw_cache = json.load(f)

            flat_cache = {}
            for key, value in raw_cache.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        flat_cache[_clean_string(sub_key)] = sub_value
                else:
                    flat_cache[_clean_string(key)] = value

            print(f"  - Loaded {len(flat_cache)} existing translations from cache.")
            return flat_cache

    print("  - No existing translation cache found. Starting fresh.")
    return {}


def _save_cache(cache):
    with open(config.TRANSLATION_CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=4)


def translate_data(df):
    """
    Orchestrates the translation of specific categorical columns using a persistent cache.
    Skips translation if the dataset appears to already be in English.
    """
    df = df.copy()

    # If we see the English word 'closed' in status, it's the new dataset.
    if 'status' in df.columns and df['status'].astype(str).str.contains('closed', case=False, regex=True).any():
        print("- Dataset appears to already be in English. Skipping translation API calls.")
        return df

    print("- Translating specific categorical columns...")
    cache = _load_cache()
    updated = False

    columns_to_translate = ['movement', 'status', 'class', 'subject_matter', 'court_department']

    for col in columns_to_translate:
        if col in df.columns:
            unique_vals = df[col].dropna().unique()

            for val in unique_vals:
                # Clean the string before checking the cache
                val_str = _clean_string(val)

                if not val_str:
                    continue

                if val_str not in cache:
                    try:
                        translated = GoogleTranslator(source='pt', target='en').translate(val_str)
                        cache[val_str] = translated
                        updated = True
                    except Exception as e:
                        print(f"    Translation failed for '{val_str}': {e}")
                        # Fallback to original so it doesn't get stuck in an endless loop next run
                        cache[val_str] = val_str

                        # Apply the mapping using the cleaned strings
            df[col] = df[col].map(lambda x: cache.get(_clean_string(x), x) if pd.notna(x) else x)

    if updated:
        _save_cache(cache)
        print("  - Translation cache updated.")

    return df
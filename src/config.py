import pathlib

# --- PATHS ---
PROJECT_DIR = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Initialize safely to prevent import crashes
RAW_FILENAME = None
DATASET_PATH = None
NAME = None
TRANSLATED_FILENAME = None
FEATURED_FILENAME = None
MODEL_RESULTS_FILE = None

TRANSLATION_CACHE_FILE = DATA_DIR / "translation_cache.json"

# --- CORE PROCESS MINING COLUMNS ---
# The loader adapter ensures every dataset uses these core names
COL_CASE_ID = 'lawsuit_id'
COL_DATE = 'date'
COL_ACTIVITY = 'movement'
COL_RESOURCE = 'judge'
COL_STATUS = 'status'
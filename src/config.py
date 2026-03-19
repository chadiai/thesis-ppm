import pathlib

# --- PATHS ---
PROJECT_DIR = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


RAW_FILENAME = "TJSP-BL-event-log.csv"
TRANSLATED_FILENAME = "tjsp_translated.csv"
FEATURED_FILENAME = "tjsp_features.csv"
MODEL_RESULTS_FILE = REPORTS_DIR / "model_results.csv"
TRANSLATION_CACHE_FILE = DATA_DIR / "translation_cache.json"

# --- DATASET COLUMNS ---
# Standardize your dataset column names here
CATEGORICAL_COLS = ['movement', 'status', 'class', 'subject_matter', 'court_department']
COL_CASE_ID = 'lawsuit_id'
COL_DATE = 'date'
COL_ACTIVITY = 'movement'
COL_RESOURCE = 'judge'
COL_STATUS = 'status'

# Columns to parse as dates
DATE_COLS = ['date', 'distribution_date']
DATETIME_COLS = []

# --- DOMAIN CONFIGURATION ---

# Case Attributes:
# Static columns to use as baseline features.
# These will be automatically detected and processed (Target Encoded or Scaled).
CASE_ATTRIBUTES = [
    'class',
    'subject_matter',
    'court_department',
    'judge',
    'claim_amount',
    'digital'
]

ACTIVITY_CLUSTERS = {
    "Initial Filings": ["Digitized Initial Petition", "Initial Petition received", "Distributed", "Joined Contestation", "Amendment to the Initial"],
    "Decisions & Judgments": ["Decision", "Action judged", "Action dismissed", "Sentence", "Judgment", "Default sentence"],
    "Hearings": ["Hearing", "Conciliation", "Instruction and Judgment"],
    "Appeals": ["Appeal", "Interlocutory Appeal", "2nd Instance Decision"],
    "Notices & Subpoenas": ["Notary Certificate", "Letter Issued", "Writ", "Subpoena", "AR Negative", "Negative AR", "AR Positive", "Positive AR"],
    "Execution & Financial": ["Execution", "Bacen Jud", "Attachment", "Permit issued", "Deposit"]
}
# Judicial Workload PPM (Predictive Process Monitoring)

A dynamic, dataset-agnostic machine learning pipeline designed to predict lawsuit duration using court event logs. 

Originally built for Brazilian State Court (TJSP) data, this pipeline now features an automatic schema adapter, making it fully compatible with US Federal Court datasets and generic process mining event logs.

## Features

* **Dataset Agnostic:** Automatically infers column schemas, maps attributes, and normalizes judge/resource data (including handling multi-judge panels).
* **Automated Feature Engineering:**
  * *Control Flow:* Sequence tracking, handover detection, and prefix dummy extraction.
  * *Temporal:* Case duration, time since last event, and cyclic calendar features.
  * *Workload Context:* Calculates exact judge queues and subject-matter workloads at any given point in time.
* **Smart Preprocessing:** Built-in Regex cleaners and cached Portuguese-to-English translation via `deep-translator`.
* **Ablation Study Orchestration:** Automatically groups dynamically generated features vs. baseline case attributes to evaluate their predictive power.
* **Multi-Model Support:** Trains and compares **Random Forest**, **XGBoost** (with early stopping), and **LSTMs** (with categorical embeddings).
* **Rich Visualizations:** Automatically generates SHAP summaries, learning curves, workload-error distribution plots, and final showdown graphs.

## Quick Start

### 1. Install Requirements
Ensure you have Python 3.9+ installed, then run:
```
pip install -r requirements.txt
```

### 2. Run the Pipeline
The pipeline is entirely CLI-driven. You **must** provide the path to your raw event log CSV file using the `--filepath` argument. 

**Example (Relative Path):**
```
python src/main.py --filepath "data/new_dataset.csv"
```

**Example (Absolute Path):**
```
python src/main.py --filepath "C:/Users/Name/Downloads/TJSP-BL-event-log.csv"
```

## Outputs & Results

The pipeline dynamically names its outputs based on the dataset you provide, preventing accidental overwrites. Once finished, check the `reports/` directory:

* **`reports/[dataset_name]_results.csv`**: The final MAE (Mean Absolute Error) scores for all 8 feature scenarios across RF, XGB, and LSTM.
* **`reports/figures/`**: Contains all generated plots, including:
  * Exploratory Data Analysis (EDA) distributions.
  * `learning_curve_[model]_[scenario].png` (Train vs. Validation loss over time).
  * `shap_summary.png` (Global feature importance).
  * `error_by_workload_segment.png` (How workload impacts prediction accuracy).
  * `thesis_final_showdown.png` (Final performance comparison).

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from src import config

def _save_plot(filename):
    target_dir = config.FIGURES_DIR / config.NAME
    target_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(target_dir / filename, bbox_inches='tight')
    plt.close()
    print(f"\tSaved: {target_dir / filename}")

def _setup_style():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    if not config.FIGURES_DIR.exists():
        config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def plot_duration_distribution(df):
    _setup_style()
    print("- Plotting Duration Distribution...")
    if 'elapsed_time_days' not in df.columns: return

    # Calculate the total duration for each individual case
    case_durations = df.groupby('lawsuit_id')['elapsed_time_days'].max()

    plt.figure(figsize=(10, 6))
    sns.histplot(case_durations, bins=50, kde=True, color='#3498db')
    plt.title('Distribution of Lawsuit Duration (Days)')
    plt.xlabel('Days')
    plt.ylabel('Number of Cases')
    _save_plot("duration_distribution.png")

def plot_workload_vs_duration(df):
    _setup_style()
    print("- Plotting Workload vs Duration (Scatter)...")

    workload_col = 'judge_workload' if 'judge_workload' in df.columns else 'judge_queue_length'
    if workload_col not in df.columns or 'elapsed_time_days' not in df.columns:
        return

    # Aggregation: Average judge workload per case vs. Total case duration
    case_stats = df.groupby('lawsuit_id').agg({
        workload_col: 'mean',
        'elapsed_time_days': 'max'
    })

    # Filter outliers (Top 5%) for a cleaner scatter plot
    q95 = case_stats['elapsed_time_days'].quantile(0.95)
    df_plot = case_stats[case_stats['elapsed_time_days'] < q95]

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df_plot,
        x=workload_col, y='elapsed_time_days',
        alpha=0.4, color='#2c3e50', edgecolor=None
    )
    plt.title('Hypothesis Check: Judge Workload vs. Lawsuit Duration')
    plt.xlabel('Average Judge Queue Size (Active Cases)')
    plt.ylabel('Case Duration (Days)')
    _save_plot("workload_vs_duration.png")

def plot_cases_per_judge(df):
    _setup_style()
    print("- Plotting Judge Caseload...")
    if 'judge' not in df.columns: return

    judge_counts = df.groupby('judge')['lawsuit_id'].nunique().sort_values(ascending=False).head(20)

    plt.figure(figsize=(12, 8))
    sns.barplot(x=judge_counts.values, y=judge_counts.index, hue=judge_counts.index, legend=False, palette="viridis")
    plt.title('Top 20 Judges by Case Volume')
    plt.xlabel('Unique Lawsuits')
    plt.ylabel('Judge Identifier')
    _save_plot("cases_per_judge.png")

def plot_feature_importance(model, X_test):
    _setup_style()
    print("- Plotting Feature Importance...")

    if not hasattr(model, 'feature_importances_'):
        print("\tModel does not support feature importances.")
        return

    # Get the exact features the model was actually trained on
    if hasattr(model, 'feature_names_in_'):
        features = list(model.feature_names_in_)
    else:
        # Fallback if attribute is missing
        features = [c for c in X_test.columns if c != 'predicted_remaining'][:len(model.feature_importances_)]

    importances = model.feature_importances_

    # Safety check
    if len(features) != len(importances):
        print(f"\t[!] Error: Feature names ({len(features)}) and importances ({len(importances)}) length mismatch.")
        return

    feat_imp = pd.DataFrame({'feature': features, 'importance': importances})
    feat_imp = feat_imp.sort_values(by='importance', ascending=False).head(20)

    max_label_length = 40
    feat_imp['feature'] = feat_imp['feature'].apply(
        lambda x: (x[:max_label_length] + '...') if len(x) > max_label_length else x
    )

    plt.figure(figsize=(12, 8))
    sns.barplot(data=feat_imp, x='importance', y='feature', hue='feature', legend=False, palette='mako')
    plt.title('Top 20 Features Influencing Prediction (XAI)')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    _save_plot("feature_importance.png")

def plot_prefix_length_distribution(df):
    _setup_style()
    print("- Plotting Prefix Length Distribution...")
    if 'prefix_length' not in df.columns: return

    plt.figure(figsize=(10, 6))

    # Calculate the total length of each case (max prefix length per lawsuit)
    case_lengths = df.groupby('lawsuit_id')['prefix_length'].max()
    max_len = case_lengths.quantile(0.99)
    case_lengths = case_lengths[case_lengths <= max_len]

    sns.histplot(case_lengths, bins=30, kde=False, color='#9b59b6')
    plt.title('Distribution of Case Lengths (Events per Case)')
    plt.xlabel('Number of Events')
    plt.ylabel('Frequency (Number of Cases)')
    _save_plot("prefix_length_distribution.png")


def plot_remaining_time_by_prefix(df):
    _setup_style()
    print("- Plotting Remaining Time vs Progress...")
    if 'prefix_length' not in df.columns or 'remaining_time_days' not in df.columns: return

    # Filter out the extreme tail of very long cases for a smoother line
    max_len = df['prefix_length'].quantile(0.95)
    df_plot = df[df['prefix_length'] <= max_len]
    mean_rem = df_plot.groupby('prefix_length')['remaining_time_days'].mean()

    plt.figure(figsize=(10, 6))
    sns.lineplot(x=mean_rem.index, y=mean_rem.values, color='#e74c3c', linewidth=2.5)
    plt.title('Average Remaining Time by Case Progress')
    plt.xlabel('Event Number (Prefix Length)')
    plt.ylabel('Avg. Remaining Time (Days)')
    plt.grid(True, linestyle='--', linewidth=0.5)
    _save_plot("remaining_time_by_prefix.png")

def plot_shap_summary(model, X_test):
    _setup_style()
    print("- Calculating SHAP values...")

    # Ensure we only pass the exact features the model was trained on to SHAP
    if hasattr(model, 'feature_names_in_'):
        expected_features = list(model.feature_names_in_)
        X_model_data = X_test[expected_features]
    else:
        X_model_data = X_test.drop(columns=['predicted_remaining'], errors='ignore')

    # Subsample for speed to prevent memory/computation hangs
    X_sample = X_model_data.sample(n=min(1000, len(X_model_data)), random_state=42).copy()

    # Truncate long feature names so they don't squeeze the plot
    max_label_length = 40
    truncated_columns = [
        (col[:max_label_length] + '...') if len(col) > max_label_length else col
        for col in X_sample.columns
    ]
    X_sample.columns = truncated_columns

    try:
        explainer = shap.Explainer(model)
        shap_values = explainer(X_sample)

        # Force a wider figure explicitly in SHAP
        shap.summary_plot(shap_values, X_sample, show=False, max_display=15, plot_size=(12, 8))

        plt.title('SHAP Summary: Feature Impact on Duration')
        plt.tight_layout()
        _save_plot("shap_summary.png")
    except Exception as e:
        print(f"\t[!] SHAP failed: {e}")


def plot_error_by_prefix_length(X_test, y_test):
    _setup_style()
    print("- Plotting Error by Prefix Length...")
    if 'predicted_remaining' not in X_test.columns or 'prefix_length_raw' not in X_test.columns: return

    df_eval = X_test.copy()
    df_eval['actual'] = y_test
    df_eval['error'] = abs(df_eval['predicted_remaining'] - df_eval['actual'])

    # Smart truncation: Ignore the top 5% longest prefix lengths where error spikes due to lack of sample cases
    max_len = int(df_eval['prefix_length_raw'].quantile(0.95))
    df_plot = df_eval[df_eval['prefix_length_raw'] <= max_len]

    mae_by_len = df_plot.groupby('prefix_length_raw')['error'].mean()

    plt.figure(figsize=(10, 6))
    sns.lineplot(x=mae_by_len.index, y=mae_by_len.values, marker='o', color='#c0392b', linewidth=2)
    plt.title('Model Error (MAE) by Case Progress')
    plt.xlabel('Event Number (Prefix Length)')
    plt.ylabel('Mean Absolute Error (Days)')
    plt.grid(True, linestyle='--')
    _save_plot("error_by_prefix_length.png")

def run_eda_plots(df):
    print("\n[Generating EDA Visualizations]")
    plot_duration_distribution(df)
    plot_workload_vs_duration(df)
    plot_cases_per_judge(df)
    plot_prefix_length_distribution(df)
    plot_remaining_time_by_prefix(df)

def plot_error_by_workload_severity(X_test, y_test):
    _setup_style()
    print("- Plotting Error Segmented by Workload...")
    if 'predicted_remaining' not in X_test.columns or 'judge_workload' not in X_test.columns: return

    df_eval = X_test.copy()
    df_eval['actual'] = y_test
    df_eval['error'] = abs(df_eval['predicted_remaining'] - df_eval['actual'])

    # Segment cases into High vs Low Workload at the time of prediction
    median_workload = df_eval['judge_workload'].median()
    df_eval['Workload Segment'] = df_eval['judge_workload'].apply(
        lambda x: 'High Workload' if x >= median_workload else 'Low Workload'
    )

    max_len = int(df_eval['prefix_length_raw'].quantile(0.95))
    df_plot = df_eval[df_eval['prefix_length_raw'] <= max_len]

    mae_segmented = df_plot.groupby(['prefix_length_raw', 'Workload Segment'])['error'].mean().reset_index()

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=mae_segmented, x='prefix_length_raw', y='error', hue='Workload Segment', linewidth=2)
    plt.title('Model Error (MAE) by Case Progress & Judge Workload')
    plt.xlabel('Event Number (Prefix Length)')
    plt.ylabel('Mean Absolute Error (Days)')
    plt.grid(True, linestyle='--')
    _save_plot("error_by_workload_severity.png")


def plot_thesis_feature_progression():
    _setup_style()
    print("- Plotting Feature Progression (RF vs LSTM)...")

    # Load the results CSV that was just generated
    if not config.MODEL_RESULTS_FILE.exists():
        print(f"\t[!] Cannot find {config.MODEL_RESULTS_FILE}. Run modeling first.")
        return

    df = pd.read_csv(config.MODEL_RESULTS_FILE)

    # Filter for only RF and LSTM models
    df_plot = df[df['Model'].isin(['RF', 'LSTM'])].copy()

    # Map RF's "Full Control Flow" to "Control Flow (Sequence)" so the bars group together properly
    df_plot['Scenario'] = df_plot['Scenario'].replace({"Full Control Flow": "Control Flow (Sequence)"})

    # Define the 5 main scenarios to compare in the desired order
    target_scenarios = [
        "Case Attributes (Baseline)",
        "Temporal Features",
        "Workload Features",
        "All Features",
        "Control Flow (Sequence)"
    ]

    # Filter out the extra RF ablation scenarios (like 'Control Flow: Last Two')
    df_plot = df_plot[df_plot['Scenario'].isin(target_scenarios)]

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        data=df_plot,
        x="Scenario",
        y="MAE",
        hue="Model",
        order=target_scenarios,
        palette="muted"
    )

    plt.title('Error Progression by Feature Scenario (RF vs LSTM)', fontweight='bold', pad=15)
    plt.xlabel('Feature Set', labelpad=10)
    plt.ylabel('Mean Absolute Error (Days)', labelpad=10)
    plt.xticks(rotation=25, ha='right')
    plt.legend(title='Architecture', loc='upper right')
    _save_plot("thesis_feature_progression.png")


def plot_thesis_final_showdown(df_results):
    """
    Plots the absolute best performing scenario for each model type (RF, XGB, LSTM)
    and labels the bar with the specific feature set that achieved it.
    """
    _setup_style()
    print("- Plotting Final Model Showdown (Best Configuration per Model)...")

    if df_results is None or df_results.empty:
        print("  - No results to plot.")
        return

    # 1. Find the index of the lowest MAE for each Model type
    best_idx = df_results.groupby('Model')['MAE'].idxmin()
    best_df = df_results.loc[best_idx].copy()

    # Sort them descending so the lowest error (best model) is on the right
    best_df = best_df.sort_values(by='MAE', ascending=False).reset_index(drop=True)

    plt.figure(figsize=(10, 6))

    # 2. Define colors for each specific model
    color_map = {'RF': '#95a5a6', 'XGB': '#3498db', 'LSTM': '#e74c3c'}
    bar_colors = [color_map.get(str(m).upper(), '#bdc3c7') for m in best_df['Model']]

    # 3. Create the Bar Chart
    bars = plt.bar(best_df['Model'], best_df['MAE'], color=bar_colors, width=0.5)

    # 4. Add the Labels (MAE and Scenario Name) directly above the bars
    for bar, mae, scenario in zip(bars, best_df['MAE'], best_df['Scenario']):
        yval = bar.get_height()

        # Format the text to show the score, then the winning feature set in brackets below it
        label_text = f"{mae:.2f} days\n[{scenario}]"

        plt.text(bar.get_x() + bar.get_width() / 2, yval + (best_df['MAE'].max() * 0.02),
                 label_text, ha='center', va='bottom', fontsize=10, fontweight='bold',
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))

    plt.title(f'Final Showdown: Best Configuration per Algorithm ({config.NAME})', fontsize=14, pad=20)
    plt.ylabel('Mean Absolute Error (Days)', fontsize=12)
    plt.xlabel('Algorithm', fontsize=12)

    # Extend the y-axis slightly so the text boxes don't get cut off at the top
    plt.ylim(0, best_df['MAE'].max() * 1.25)

    plt.grid(axis='y', linestyle='--', alpha=0.7)

    _save_plot("_thesis_final_showdown.png")

def plot_learning_curve(history, model_name, scenario_name):
    """
    Plots the Training vs Validation Loss over epochs/boosting rounds.
    """
    _setup_style()
    print(f"- Plotting Learning Curve for {model_name}...")

    plt.figure(figsize=(10, 6))
    epochs = range(1, len(history['train']) + 1)

    sns.lineplot(x=epochs, y=history['train'], label='Train Loss (MAE)', color='#3498db', linewidth=2)
    if 'val' in history and history['val']:
        sns.lineplot(x=epochs, y=history['val'], label='Validation Loss (MAE)', color='#e74c3c', linewidth=2)

    plt.title(f'Learning Curve: {model_name} ({scenario_name})')
    plt.xlabel('Epochs / Boosting Rounds')
    plt.ylabel('Mean Absolute Error (Days)')
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5)

    # Sanitize scenario name for the file path
    safe_scenario = scenario_name.replace(" ", "_").replace(":", "").replace("(", "").replace(")", "").lower()
    _save_plot(f"learning_curve_{model_name.lower()}_{safe_scenario}.png")

def run_model_plots(model, X_test, y_test):
    print("\n[Generating Model Evaluation Visualizations]")
    plot_feature_importance(model, X_test)
    plot_shap_summary(model, X_test)
    plot_error_by_prefix_length(X_test, y_test)
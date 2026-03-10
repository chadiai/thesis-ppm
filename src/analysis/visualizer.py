import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from src import config


def _save_plot(filename):
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / filename, bbox_inches='tight')
    plt.close()
    print(f"\tSaved: {config.FIGURES_DIR / filename}")


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

    # FIX: Truncate long feature names so they don't squeeze the plot
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
        _save_plot("shap_summary.png")
    except Exception as e:
        print(f"\t[!] SHAP failed: {e}")


def plot_error_by_prefix_length(X_test, y_test):
    _setup_style()
    print("- Plotting Error by Prefix Length...")
    if 'predicted_remaining' not in X_test.columns or 'prefix_length' not in X_test.columns: return

    df_eval = X_test.copy()
    df_eval['actual'] = y_test
    df_eval['error'] = abs(df_eval['predicted_remaining'] - df_eval['actual'])

    # Smart truncation: Ignore the top 5% longest prefix lengths where error spikes due to lack of sample cases
    max_len = int(df_eval['prefix_length'].quantile(0.95))
    df_plot = df_eval[df_eval['prefix_length'] <= max_len]

    mae_by_len = df_plot.groupby('prefix_length')['error'].mean()

    plt.figure(figsize=(10, 6))
    sns.lineplot(x=mae_by_len.index, y=mae_by_len.values, marker='o', color='#c0392b', linewidth=2)
    plt.title('Model Error (MAE) by Case Progress')
    plt.xlabel('Event Number (Prefix Length)')
    plt.ylabel('Mean Absolute Error (Days)')
    plt.grid(True, linestyle='--')
    _save_plot("error_by_prefix.png")


def run_eda_plots(df):
    print("\n[Generating EDA Visualizations]")
    plot_duration_distribution(df)
    plot_workload_vs_duration(df)
    plot_cases_per_judge(df)
    plot_prefix_length_distribution(df)
    plot_remaining_time_by_prefix(df)


def run_model_plots(model, X_test, y_test):
    print("\n[Generating Model Evaluation Visualizations]")
    plot_feature_importance(model, X_test)
    plot_shap_summary(model, X_test)
    plot_error_by_prefix_length(X_test, y_test)
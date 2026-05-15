import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from src import config
from src.analysis import visualizer
from src.modeling import dl_prep

class LawsuitLSTM(nn.Module):
    def __init__(self, embedding_sizes, num_continuous, hidden_size=64, num_layers=2):
        super().__init__()

        # Create a list of Embedding layers (one for each categorical feature)
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_categories, emb_dim, padding_idx=0)
            for num_categories, emb_dim in embedding_sizes
        ])

        # Calculate the total input dimension that the LSTM will receive
        total_emb_dim = sum(emb_dim for _, emb_dim in embedding_sizes)
        lstm_input_size = total_emb_dim + num_continuous

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )

        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x_cat, x_cont):
        emb_outputs = []
        # Pass each categorical column through its respective embedding layer
        for i, emb_layer in enumerate(self.embeddings):
            emb_out = emb_layer(x_cat[:, :, i])
            emb_outputs.append(emb_out)

        # Concatenate the dense embeddings together with the continuous features
        x_emb = torch.cat(emb_outputs, dim=-1)
        x_combined = torch.cat([x_emb, x_cont], dim=-1)

        lstm_out, _ = self.lstm(x_combined)

        out = self.fc1(lstm_out)
        out = self.relu(out)
        out = self.fc2(out)

        return out.squeeze(-1)


def train_and_evaluate_lstm(train_loader, test_loader, embedding_sizes, num_continuous, epochs=25, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"- Training LSTM on {device}...")

    model = LawsuitLSTM(embedding_sizes, num_continuous).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss(reduction='none')

    # Dict to hold history for plotting
    history = {'train': [], 'val': []}

    for epoch in range(epochs):
        # --- TRAINING PHASE ---
        model.train()
        total_train_loss = 0
        total_valid_events = 0

        for x_cat, x_cont, y, mask, _, _ in train_loader:
            x_cat, x_cont, y, mask = x_cat.to(device), x_cont.to(device), y.to(device), mask.to(device)

            optimizer.zero_grad()
            preds = model(x_cat, x_cont)

            loss = criterion(preds, y)
            loss = (loss * mask).sum()
            valid_events = mask.sum()

            (loss / valid_events).backward()
            optimizer.step()

            total_train_loss += loss.item()
            total_valid_events += valid_events.item()

        train_mae = total_train_loss / total_valid_events
        history['train'].append(train_mae)

        # --- VALIDATION PHASE (End of Epoch) ---
        model.eval()
        total_val_loss = 0
        total_val_events = 0

        with torch.no_grad():
            for x_cat, x_cont, y, mask, _, _ in test_loader:
                x_cat, x_cont, y, mask = x_cat.to(device), x_cont.to(device), y.to(device), mask.to(device)
                preds = model(x_cat, x_cont)

                loss = (criterion(preds, y) * mask).sum()
                total_val_loss += loss.item()
                total_val_events += mask.sum().item()

        val_mae = total_val_loss / total_val_events
        history['val'].append(val_mae)

        print(f"  Epoch {epoch + 1:02d}/{epochs} | Train MAE: {train_mae:.2f} | Val MAE: {val_mae:.2f}")

    # Recalculate full errors on the final test set to get standard deviation
    model.eval()
    all_errors = []
    with torch.no_grad():
        for x_cat, x_cont, y, mask, _, _ in test_loader:
            x_cat, x_cont, y, mask = x_cat.to(device), x_cont.to(device), y.to(device), mask.to(device)
            preds = model(x_cat, x_cont)
            loss_tensor = criterion(preds, y) * mask

            # Flatten and keep only valid events (ignore padding)
            loss_flat = loss_tensor.flatten()
            mask_flat = mask.flatten()
            valid_errors = loss_flat[mask_flat > 0]
            all_errors.extend(valid_errors.cpu().numpy())

    all_errors = np.array(all_errors)
    final_mae = np.mean(all_errors)
    final_std = np.std(all_errors)
    print(f"\n--- LSTM Final Test MAE: {final_mae:.2f} ± {final_std:.2f} days ---\n")

    # Return the final metric AND the history
    return final_mae, final_std, history


def run_experiment(df_feat, data_dict):
    print("\n Predictive Modeling (Deep Learning - LSTM with Embeddings)")

    # Reconstruct raw train/test split using the dynamic config ID
    train_ids = data_dict['train_df'][config.COL_CASE_ID].unique()
    test_ids = data_dict['test_df'][config.COL_CASE_ID].unique()

    raw_train_df = df_feat[df_feat[config.COL_CASE_ID].isin(train_ids)].dropna(subset=['remaining_time_days']).copy()
    raw_test_df = df_feat[df_feat[config.COL_CASE_ID].isin(test_ids)].dropna(subset=['remaining_time_days']).copy()

    # Dynamically separate features generated by the pipeline from base case attributes
    pipeline_generated = ['movement', 'Weekday', 'Month', 'elapsed_time_days',
                          'time_since_last_event', 'judge_workload', 'workload_by_subject',
                          'prefix_length', 'judge_changed', 'case_start', 'case_end', 'remaining_time_days']
    pipeline_generated += [c for c in df_feat.columns if c.startswith('Count_')]

    # Everything else belongs to Base Case Attributes
    base_features = [c for c in df_feat.columns if
                     c not in pipeline_generated and c != config.COL_CASE_ID and c != config.COL_DATE]

    base_cat = df_feat[base_features].select_dtypes(exclude=['number']).columns.tolist()
    base_cont = df_feat[base_features].select_dtypes(include=['number']).columns.tolist()

    # Define the 5 LSTM Scenarios using the dynamic lists
    lstm_scenarios = {
        "Case Attributes (Baseline)": {"cat": base_cat, "cont": base_cont},
        "Control Flow (Sequence)": {"cat": base_cat + ['movement'], "cont": base_cont},
        "Temporal Features": {"cat": base_cat + ['Weekday', 'Month'],
                              "cont": base_cont + ['elapsed_time_days', 'time_since_last_event']},
        "Workload Features": {"cat": base_cat, "cont": base_cont + ['judge_workload', 'workload_by_subject']},
        "All Features": {"cat": base_cat + ['movement', 'Weekday', 'Month'],
                         "cont": base_cont + ['elapsed_time_days', 'time_since_last_event', 'judge_workload',
                                              'workload_by_subject']}
    }

    print("\n" + "=" * 60)
    print(" RUNNING LSTM SCENARIOS")
    print("=" * 60)

    results_df = pd.DataFrame(columns=["Model", "Scenario", "MAE", "Num_Features"])

    for scenario_name, cols in lstm_scenarios.items():
        cat_cols = [c for c in cols["cat"] if c in df_feat.columns]
        cont_cols = [c for c in cols["cont"] if c in df_feat.columns]

        print(f"\n--- Model: LSTM | Scenario: {scenario_name} ---")

        train_loader, test_loader, embedding_sizes = dl_prep.prepare_dl_data(
            raw_train_df,
            raw_test_df,
            cat_cols=cat_cols,
            cont_cols=cont_cols,
            batch_size=64
        )

        lstm_mae, lstm_std, history = train_and_evaluate_lstm(
            train_loader,
            test_loader,
            embedding_sizes=embedding_sizes,
            num_continuous=len(cont_cols),
            epochs=20
        )

        lstm_row = pd.DataFrame([{
            "Model": "LSTM",
            "Scenario": scenario_name,
            "MAE": lstm_mae,
            "STD": lstm_std,
            "Num_Features": len(cat_cols) + len(cont_cols)
        }])

        # Avoid the pandas deprecation warning by dropping empty/NA columns if any exist
        results_df = pd.concat([results_df, lstm_row], ignore_index=True)
        visualizer.plot_learning_curve(history, "LSTM", scenario_name)
        print(f"  {scenario_name:40s} | MAE: {lstm_mae:.2f} ± {lstm_std:.2f} days")

    return results_df
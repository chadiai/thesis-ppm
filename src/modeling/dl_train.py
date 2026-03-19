import torch
import torch.nn as nn


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


def train_and_evaluate_lstm(train_loader, test_loader, embedding_sizes, num_continuous, epochs=15, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"- Training LSTM on {device}...")

    model = LawsuitLSTM(embedding_sizes, num_continuous).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss(reduction='none')

    for epoch in range(epochs):
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
        print(f"  Epoch {epoch + 1:02d}/{epochs} | Train MAE: {train_mae:.2f} days")

    # --- EVALUATION ---
    model.eval()
    total_test_loss = 0
    total_test_events = 0

    with torch.no_grad():
        for x_cat, x_cont, y, mask, _, _ in test_loader:
            x_cat, x_cont, y, mask = x_cat.to(device), x_cont.to(device), y.to(device), mask.to(device)
            preds = model(x_cat, x_cont)

            loss = (criterion(preds, y) * mask).sum()
            total_test_loss += loss.item()
            total_test_events += mask.sum().item()

    test_mae = total_test_loss / total_test_events
    print(f"\n--- LSTM (With Embeddings) Final Test MAE: {test_mae:.2f} days ---\n")

    return test_mae
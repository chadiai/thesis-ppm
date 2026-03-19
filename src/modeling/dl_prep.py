import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class LawsuitDataset(Dataset):
    def __init__(self, grouped_data, cat_cols, cont_cols, target_col='remaining_time_days'):
        self.cases_cat = []
        self.cases_cont = []
        self.targets = []
        self.case_ids = []

        for case_id, group in grouped_data:
            # Extract categorical and continuous features separately
            x_cat = group[cat_cols].values.astype(np.int64)
            x_cont = group[cont_cols].values.astype(np.float32)
            y = group[target_col].values.astype(np.float32)

            self.cases_cat.append(torch.tensor(x_cat))
            self.cases_cont.append(torch.tensor(x_cont))
            self.targets.append(torch.tensor(y))
            self.case_ids.append(case_id)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.cases_cat[idx], self.cases_cont[idx], self.targets[idx], self.case_ids[idx]


def pad_collate_fn(batch):
    xs_cat, xs_cont, ys, case_ids = zip(*batch)
    lengths = torch.tensor([len(x) for x in xs_cat])

    # Pad sequences with 0. 0 will act as our "ignore" or "unknown" index for embeddings
    xs_cat_padded = torch.nn.utils.rnn.pad_sequence(xs_cat, batch_first=True, padding_value=0)
    xs_cont_padded = torch.nn.utils.rnn.pad_sequence(xs_cont, batch_first=True, padding_value=0.0)
    ys_padded = torch.nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=0.0)

    mask = torch.arange(xs_cat_padded.size(1))[None, :] < lengths[:, None]

    return xs_cat_padded, xs_cont_padded, ys_padded, mask, lengths, case_ids


def prepare_dl_data(train_df, test_df, cat_cols, cont_cols, batch_size=64):
    print("  - Encoding categoricals for Embedding layers...")

    embedding_sizes = []
    train_encoded = train_df.copy()
    test_encoded = test_df.copy()

    for col in cat_cols:
        unique_vals = train_df[col].dropna().unique()
        # Map values to an index (starting at 1, because 0 is reserved for padding/unknowns)
        val2idx = {val: idx + 1 for idx, val in enumerate(unique_vals)}

        train_encoded[col] = train_df[col].map(val2idx).fillna(0).astype(int)
        test_encoded[col] = test_df[col].map(val2idx).fillna(0).astype(int)

        # Determine embedding dimension based on a standard heuristic
        num_categories = len(val2idx) + 1
        emb_dim = min(50, (num_categories + 1) // 2)
        embedding_sizes.append((num_categories, emb_dim))

    print("  - Standardizing continuous variables...")
    for col in cont_cols:
        mean = train_encoded[col].mean()
        std = train_encoded[col].std() + 1e-8
        train_encoded[col] = (train_encoded[col] - mean) / std
        test_encoded[col] = (test_encoded[col] - mean) / std

    print(f"  - Grouping {len(train_encoded)} train events into sequences...")
    train_grouped = train_encoded.sort_values(['lawsuit_id', 'date']).groupby('lawsuit_id')
    test_grouped = test_encoded.sort_values(['lawsuit_id', 'date']).groupby('lawsuit_id')

    train_dataset = LawsuitDataset(train_grouped, cat_cols, cont_cols)
    test_dataset = LawsuitDataset(test_grouped, cat_cols, cont_cols)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn)

    return train_loader, test_loader, embedding_sizes
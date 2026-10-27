import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import yaml

def load_config(config_path='configs/default.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_stratified_splits(df, seed=42, test_ratio=0.2, val_ratio=0.1):
    config = load_config()
    processed_ids = [os.path.basename(p).replace(".npy", "").strip() for p in glob_and_list_npy(config['paths']['processed_dir'])]
    
    # Normalize Excel IDs to strip any trailing/leading whitespaces
    df_copy = df.copy()
    df_copy['ID'] = df_copy['ID'].astype(str).str.strip()
    
    print("Sample processed IDs (Preprocessed):", processed_ids[:5])
    print("Sample IDs in Metadata Excel:", df_copy['ID'].head(5).tolist())

    df_valid = df_copy[df_copy['ID'].isin(processed_ids) & df_copy['CDR'].notna()].copy()
    print(f"Total metadata subjects matching processed files: {len(df_valid)}")

    def map_cdr(cdr):
        if cdr == 0: return 0
        elif cdr == 0.5: return 1
        elif cdr >= 1.0: return 2
        return -1

    df_valid['label'] = df_valid['CDR'].apply(map_cdr)
    df_valid = df_valid[df_valid['label'] != -1].reset_index(drop=True)

    np.random.seed(seed)
    unique_subjects = df_valid['ID'].unique()
    np.random.shuffle(unique_subjects)

    labels_dict = df_valid.set_index('ID')['label'].to_dict()
    classes = {0: [], 1: [], 2: []}
    for subj in unique_subjects:
        classes[labels_dict[subj]].append(subj)

    train_ids, val_ids, test_ids = [], [], []
    for c, subjs in classes.items():
        n_total = len(subjs)
        n_test = int(np.ceil(n_total * test_ratio))
        n_val = int(np.ceil(n_total * val_ratio))
        test_ids.extend(subjs[:n_test])
        val_ids.extend(subjs[n_test:n_test + n_val])
        train_ids.extend(subjs[n_test + n_val:])

    splits = {'train': train_ids, 'val': val_ids, 'test': test_ids}
    os.makedirs(os.path.dirname(config['paths']['split_json']), exist_ok=True)
    with open(config['paths']['split_json'], 'w') as f:
        json.dump(splits, f, indent=4)

    print(f"Created splits: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")
    return splits, df_valid

def glob_and_list_npy(dir_path):
    import glob
    return glob.glob(os.path.join(dir_path, "*.npy"))

class NeuroLensDataset(Dataset):
    def __init__(self, df, subject_ids, processed_dir, transform=None):
        self.df = df[df['ID'].isin(subject_ids)].copy().reset_index(drop=True)
        self.processed_dir = processed_dir
        self.transform = transform
        def map_cdr(cdr):
            if cdr == 0: return 0
            if cdr == 0.5: return 1
            return 2
        self.df['label'] = self.df['CDR'].apply(map_cdr)
        self.df['gender_bin'] = self.df['M/F'].map({'M': 0, 'F': 1}).fillna(0)
        self.df['hand_bin'] = self.df['Hand'].map({'R': 0, 'L': 1}).fillna(0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        subj_id = row['ID']
        label = int(row['label'])
        mri_path = os.path.join(self.processed_dir, f"{subj_id}.npy")
        mri = np.load(mri_path).astype(np.float32)
        mri = np.expand_dims(mri, axis=0)
        educ_val = row['Educ']
        educ_missing = 1.0 if pd.isna(educ_val) else 0.0
        ses_val = row['SES']
        ses_missing = 1.0 if pd.isna(ses_val) else 0.0
        clinical_vector = np.array([
            float(row['Age']),
            float(row['gender_bin']),
            float(row['hand_bin']),
            float(0.0 if educ_missing else educ_val),
            float(0.0 if ses_missing else ses_val),
            float(row['eTIV']),
            float(row['nWBV']),
            float(row['ASF']),
            educ_missing,
            ses_missing
        ], dtype=np.float32)
        if self.transform:
            mri = self.transform(mri)
        return torch.tensor(mri, dtype=torch.float32), torch.tensor(clinical_vector, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

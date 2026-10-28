"""Evaluation for NeuroLens."""
import torch, numpy as np
from src.utils import compute_metrics, print_metrics

def evaluate_on_test_set(model, test_loader, device, num_classes=3, use_amp=False):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for mri, clinical, labels in test_loader:
            mri, clinical = mri.to(device), clinical.to(device)
            logits = model(mri, clinical)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels)
    y_pred = np.argmax(all_logits, axis=1)
    y_pred_proba = np.exp(all_logits) / np.exp(all_logits).sum(axis=1, keepdims=True)
    return compute_metrics(all_labels, y_pred, y_pred_proba, num_classes)

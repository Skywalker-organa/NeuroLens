"""Utility functions for NeuroLens training and evaluation."""
import os, json, yaml, numpy as np, torch
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support, 
    confusion_matrix, roc_auc_score
)

def load_config(config_path="configs/default.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_class_weights(labels, num_classes=3):
    counts = np.bincount(labels, minlength=num_classes)
    total = counts.sum()
    weights = total / (counts + 1e-6)
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)

def compute_metrics(y_true, y_pred, y_pred_proba=None, num_classes=3):
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['weighted_f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    class_names = ['CN', 'Very Mild', 'AD']
    for i, name in enumerate(class_names[:num_classes]):
        metrics[f'{name}_precision'] = prec[i]
        metrics[f'{name}_recall'] = rec[i]
        metrics[f'{name}_f1'] = f1[i]
        metrics[f'{name}_support'] = support[i]
    
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    
    if y_pred_proba is not None and y_pred_proba.shape[1] == num_classes:
        try:
            auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro', zero_division=0)
            metrics['auc_macro'] = auc
        except:
            metrics['auc_macro'] = None
    
    return metrics

def print_metrics(metrics, title="Metrics"):
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Macro F1:    {metrics['macro_f1']:.4f}")
    if metrics.get('auc_macro'): print(f"AUC:         {metrics['auc_macro']:.4f}")
    print(f"\n{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    for cls in ['CN', 'Very Mild', 'AD']:
        p = metrics.get(f'{cls}_precision', 0)
        r = metrics.get(f'{cls}_recall', 0)
        f = metrics.get(f'{cls}_f1', 0)
        print(f"{cls:<15} {p:<12.4f} {r:<12.4f} {f:<12.4f}")
    print(f"{'='*60}\n")

def save_checkpoint(model, optimizer, scheduler, epoch, metrics, checkpoint_dir, name="best"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }
    if scheduler: ckpt['scheduler_state_dict'] = scheduler.state_dict()
    path = os.path.join(checkpoint_dir, f"{name}.pt")
    torch.save(ckpt, path)
    return path

def load_checkpoint(model, optimizer, checkpoint_path, scheduler=None, device="cpu"):
    if not os.path.exists(checkpoint_path):
        return 0, {}
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    return ckpt.get('epoch', 0), ckpt.get('metrics', {})

def save_metrics_log(log_path, metrics_dict):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2, default=str)

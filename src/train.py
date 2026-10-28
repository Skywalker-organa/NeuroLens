"""Training loop for NeuroLens."""
import torch, torch.nn as nn, numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from src.utils import compute_class_weights, compute_metrics, print_metrics, save_checkpoint

def train_epoch(model, dataloader, criterion, optimizer, scaler, device, use_amp=True):
    model.train()
    total_loss = 0.0
    for batch_idx, (mri, clinical, labels) in enumerate(dataloader):
        mri, clinical, labels = mri.to(device), clinical.to(device), labels.to(device)
        optimizer.zero_grad()
        if use_amp:
            with autocast():
                logits = model(mri, clinical)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(mri, clinical)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
        if (batch_idx + 1) % 5 == 0: print(f"  Batch {batch_idx + 1}, Loss: {loss.item():.4f}")
    return total_loss / max(len(dataloader), 1)

def validate_epoch(model, dataloader, criterion, device, use_amp=True, num_classes=3):
    model.eval()
    total_loss = 0.0
    all_logits, all_labels = [], []
    with torch.no_grad():
        for mri, clinical, labels in dataloader:
            mri, clinical, labels = mri.to(device), clinical.to(device), labels.to(device)
            if use_amp:
                with autocast():
                    logits = model(mri, clinical)
                    loss = criterion(logits, labels)
            else:
                logits = model(mri, clinical)
                loss = criterion(logits, labels)
            total_loss += loss.item()
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels)
    y_pred = np.argmax(all_logits, axis=1)
    y_pred_proba = np.exp(all_logits) / np.exp(all_logits).sum(axis=1, keepdims=True)
    metrics = compute_metrics(all_labels, y_pred, y_pred_proba, num_classes)
    return total_loss / max(len(dataloader), 1), metrics

def train(model, train_loader, val_loader, config, device, checkpoint_dir="checkpoints", resume_from=None):
    num_epochs = config['training'].get('num_epochs', 20)
    learning_rate = config['training'].get('learning_rate', 1e-3)
    use_amp = config['training'].get('use_amp', True)
    num_classes = config['model'].get('num_classes', 3)
    patience = config['training'].get('early_stop_patience', 5)
    
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    train_labels = np.array([y for _, _, y in train_loader.dataset])
    class_weights = compute_class_weights(train_labels, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = GradScaler() if use_amp else None
    
    best_f1 = -1.0
    patience_counter = 0
    all_metrics = {'train': [], 'val': []}
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp)
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device, use_amp, num_classes)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print_metrics(val_metrics, f"Epoch {epoch + 1} Val Metrics")
        
        all_metrics['train'].append({'loss': train_loss})
        all_metrics['val'].append({'loss': val_loss, 'f1': val_metrics['macro_f1']})
        
        if val_metrics['macro_f1'] > best_f1:
            best_f1 = val_metrics['macro_f1']
            patience_counter = 0
            save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, checkpoint_dir, "best")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        
        scheduler.step()
    
    return all_metrics

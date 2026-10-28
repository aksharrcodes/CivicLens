"""Train two binary transfer-learning models for CivicLens.

Expected data layout:
  data/pothole/train/{pothole,not_pothole}/...
  data/pothole/val/{pothole,not_pothole}/...
  data/garbage/train/{garbage,not_garbage}/...
  data/garbage/val/{garbage,not_garbage}/...

Use images you have permission to use. The QR4Change dataset (CC BY 4.0)
contains pothole/non-pothole and garbage/non-garbage image sets; see README.
"""
from __future__ import annotations
import argparse, json, random, time
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

SEED = 42
random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224

def build_model(num_classes=2, pretrained=True):
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model

def train_one(task, data_root: Path, out_dir: Path, epochs=6, batch_size=32, lr=1e-3):
    task_dir = data_root / task
    positive = "pothole" if task == "pothole" else "garbage"
    negative = "not_pothole" if task == "pothole" else "not_garbage"
    train_dir, val_dir = task_dir / "train", task_dir / "val"
    for p in [train_dir / positive, train_dir / negative, val_dir / positive, val_dir / negative]:
        if not p.exists():
            raise FileNotFoundError(f"Missing class folder: {p}")
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(8),
        transforms.ColorJitter(brightness=.15, contrast=.15, saturation=.1),
        transforms.ToTensor(),
        transforms.Normalize([.485,.456,.406],[.229,.224,.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
        transforms.Normalize([.485,.456,.406],[.229,.224,.225]),
    ])
    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(val_dir, transform=val_tf)
    if len(train_ds.classes) != 2 or len(val_ds.classes) != 2:
        raise ValueError(f"{task}: expected exactly two classes in train and val folders")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=(DEVICE=='cuda'))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=(DEVICE=='cuda'))
    model = build_model().to(DEVICE)
    # Freeze the pretrained feature extractor first; train the classification head.
    for p in model.features.parameters(): p.requires_grad = False
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    best_f1, best_state, history = -1, None, []
    started = time.time()
    for epoch in range(epochs):
        model.train(); total_loss=0; seen=0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images); loss = criterion(logits, labels)
            loss.backward(); optimizer.step()
            total_loss += loss.item()*images.size(0); seen += images.size(0)
        model.eval(); y_true=[]; y_pred=[]
        with torch.no_grad():
            for images, labels in val_loader:
                logits = model(images.to(DEVICE))
                y_pred.extend(logits.argmax(1).cpu().tolist()); y_true.extend(labels.tolist())
        acc = accuracy_score(y_true, y_pred); f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        history.append({'epoch': epoch+1, 'train_loss': total_loss/max(seen,1), 'val_accuracy': acc, 'val_macro_f1': f1})
        print(f"{task} epoch {epoch+1}/{epochs} loss={history[-1]['train_loss']:.4f} val_acc={acc:.4f} macro_f1={f1:.4f}")
        if f1 > best_f1:
            best_f1=f1; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    if best_state is None: raise RuntimeError('Training produced no checkpoint')
    model.load_state_dict(best_state); model.eval()
    y_true=[]; y_pred=[]; y_prob=[]
    with torch.no_grad():
        for images, labels in val_loader:
            probs = model(images.to(DEVICE)).softmax(1).cpu()
            y_prob.extend(probs[:, train_ds.class_to_idx[positive]].tolist())
            y_pred.extend(probs.argmax(1).tolist()); y_true.extend(labels.tolist())
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        'task': task, 'state_dict': model.state_dict(), 'class_to_idx': train_ds.class_to_idx,
        'positive_class': positive, 'negative_class': negative, 'image_size': IMG_SIZE,
        'normalization_mean': [.485,.456,.406], 'normalization_std': [.229,.224,.225],
        'validation_accuracy': accuracy_score(y_true,y_pred),
        'validation_macro_f1': f1_score(y_true,y_pred,average='macro',zero_division=0),
        'confusion_matrix': confusion_matrix(y_true,y_pred).tolist(),
        'classification_report': classification_report(y_true,y_pred,target_names=train_ds.classes,output_dict=True,zero_division=0),
        'history': history, 'training_seconds': round(time.time()-started,2),
    }
    torch.save(artifact, out_dir / f'{task}_mobilenet_v3_small.pth')
    metrics = {k:v for k,v in artifact.items() if k not in ('state_dict',)}
    (out_dir / f'{task}_metrics.json').write_text(json.dumps(metrics, indent=2))
    print(f"Saved {task} model. Validation accuracy={artifact['validation_accuracy']:.3f}; macro-F1={artifact['validation_macro_f1']:.3f}")
    return metrics

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',default='data'); p.add_argument('--out',default='models'); p.add_argument('--epochs',type=int,default=6); p.add_argument('--batch-size',type=int,default=32); p.add_argument('--lr',type=float,default=1e-3)
    a=p.parse_args(); root=Path(a.data); out=Path(a.out)
    results={}
    for task in ['pothole','garbage']:
        results[task]=train_one(task,root,out,a.epochs,a.batch_size,a.lr)
    (out/'training_summary.json').write_text(json.dumps({k:{'validation_accuracy':v['validation_accuracy'],'validation_macro_f1':v['validation_macro_f1'],'confusion_matrix':v['confusion_matrix']} for k,v in results.items()},indent=2))
    print('Training finished. Review models/*_metrics.json before making performance claims.')
if __name__=='__main__': main()

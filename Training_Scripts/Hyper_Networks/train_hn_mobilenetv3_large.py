import os, json, time, random, math, warnings
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v3_large

from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report)

from tqdm import tqdm

# --- TensorBoard ---
from torch.utils.tensorboard import SummaryWriter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================
# YOUR PATHS
# =============================
DATA_ROOT    = r"D:\\cologne\\HyperNetwork\\Data"                   # has train/, validation/, test/ and *_metadata.json
BASE_WEIGHTS = r"D:\\cologne\\Models\\Models\\mobilenet_best.pth"    # your fine-tuned MobileNetV3
OUT_DIR      = r"D:\\cologne\\HyperNetwork\\Nueral_Network\\mobilenetv3_large_hn"         # outputs go here
TB_LOG_DIR   = r"runs/hn_mnv3"                                   # tensorboard logs

# =============================
# Small utils
# =============================
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def ensure_dir(p): os.makedirs(p, exist_ok=True)

# =============================
# Load split-level global metadata
# =============================
def load_split_global_meta(data_root: str, split: str) -> Dict[str, Dict[str, Any]]:
    root = Path(data_root)
    candidates = list(root.glob(f"{split}_metadata.json")) + list(root.glob(f"{split}_metadata.*"))
    if not candidates:
        for p in root.iterdir():
            n = p.name.lower()
            if n.startswith(split.lower()) and "metadata" in n and p.suffix.lower() in [".json"]:
                candidates.append(p)
    meta_map: Dict[str, Dict[str, Any]] = {}
    if not candidates:
        return meta_map
    with open(candidates[0], "r", encoding="utf-8") as f:
        raw = json.load(f)
    for k, v in raw.items():
        fname = Path(k).name
        meta_map[fname.lower()] = v
    return meta_map

# =============================
# Metadata featurizer
# =============================
class MetaFeaturizer:
    def __init__(self):
        self.keys: List[str] = []
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, meta_dicts: List[Dict[str, Any]]):
        keyset = set()
        for md in meta_dicts:
            keyset.update(md.keys())
        self.keys = sorted(list(keyset)) if keyset else ["__dummy__"]
        X = []
        for md in meta_dicts:
            row = []
            for k in self.keys:
                v = md.get(k, 0.0)
                try: row.append(float(v))
                except: row.append(0.0)
            X.append(row)
        X = np.array(X, np.float32) if len(X) else np.zeros((1, len(self.keys)), np.float32)
        self.mean = X.mean(axis=0); self.std = X.std(axis=0); self.std[self.std==0] = 1.0

    def transform(self, md: Dict[str, Any]) -> np.ndarray:
        row = []
        for k in self.keys:
            v = md.get(k, 0.0)
            try: row.append(float(v))
            except: row.append(0.0)
        x = np.array(row, np.float32)
        return (x - self.mean) / self.std if self.mean is not None else x

    @property
    def dim(self): return max(1, len(self.keys))

# =============================
# Dataset
# =============================
class ImageWithMetadataDataset(Dataset):
    def __init__(self, data_root: str, split_name: str, featurizer: MetaFeaturizer, img_size=224, augment=False):
        split_dir = Path(data_root) / split_name
        self.split_name = split_name
        self.split_dir = split_dir
        self.classes = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {c:i for i,c in enumerate(self.classes)}
        self.samples: List[Tuple[Path, int]] = []
        for cls in self.classes:
            for p in (split_dir / cls).rglob("*"):
                if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"} and p.is_file():
                    self.samples.append((p, self.class_to_idx[cls]))
        self.split_meta = load_split_global_meta(data_root, split_name)
        self.featurizer = featurizer

        if augment:
            self.tf = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(0.2,0.2,0.2,0.05),
                transforms.ToTensor(),  # must be before RandomErasing
                transforms.RandomErasing(p=0.2, scale=(0.02,0.1), ratio=(0.3,3.3)),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ])

    def __len__(self): return len(self.samples)

    def _get_meta(self, img_path: Path) -> Dict[str, Any]:
        return self.split_meta.get(img_path.name.lower(), {})

    def __getitem__(self, idx):
        p, y = self.samples[idx]
        x = Image.open(p).convert("RGB")
        x = self.tf(x)
        md = self._get_meta(p)
        mv = self.featurizer.transform(md)
        return x, torch.from_numpy(mv), y, str(p)

# =============================
# MixUp helpers
# =============================
def mixup_data(x, meta, y, alpha=0.2):
    if alpha <= 0: return x, meta, y, 1.0, torch.arange(x.size(0), device=x.device)
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    return lam*x + (1-lam)*x[idx], lam*meta + (1-lam)*meta[idx], y, lam, idx

def mixup_criterion(criterion, pred, y, lam, idx):
    return lam*criterion(pred, y) + (1-lam)*criterion(pred, y[idx])

# =============================
# HyperNet (FiLM) + wrapper
# =============================
class HyperFiLM(nn.Module):
    def __init__(self, meta_dim, channels, hidden=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(meta_dim, hidden), nn.ReLU(True), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(True), nn.Dropout(dropout),
            nn.Linear(hidden, channels*2)
        )
        with torch.no_grad():
            self.net[-1].weight.zero_(); self.net[-1].bias.zero_()

    def forward(self, meta):
        out = self.net(meta)
        gamma, beta = torch.chunk(out, 2, dim=1)
        gamma = 1.0 + 0.1*torch.tanh(gamma)
        beta  = 0.1*torch.tanh(beta)
        return gamma, beta

class HNMobileNetV3Large(nn.Module):
    def __init__(self, base, meta_dim, film_channels, freeze_early=True):
        super().__init__()
        self.base = base
        self.hyper = HyperFiLM(meta_dim, film_channels)
        if freeze_early:
            for p in self.base.features.parameters(): p.requires_grad = False
            for idx in [-1,-2]:
                for p in self.base.features[idx].parameters(): p.requires_grad = True
            for p in self.base.classifier.parameters(): p.requires_grad = True

    def unfreeze_all(self):
        for p in self.base.parameters(): p.requires_grad = True

    def forward(self, x, meta):
        feat = self.base.features(x)  # [B,C,H,W] where C == film_channels
        gamma, beta = self.hyper(meta)
        feat = feat * gamma.unsqueeze(-1).unsqueeze(-1) + beta.unsqueeze(-1).unsqueeze(-1)
        out = self.base.avgpool(feat); out = torch.flatten(out, 1)
        return self.base.classifier(out)

# =============================
# Plots & eval
# =============================
def save_confusion_matrix(cm, class_names, out_path):
    fig = plt.figure(figsize=(5,4), dpi=140)
    plt.imshow(cm, interpolation="nearest"); plt.title("Confusion Matrix"); plt.colorbar()
    ax = plt.gca(); ax.set_xticks(range(len(class_names))); ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45); ax.set_yticklabels(class_names)
    thresh = cm.max()/2. if cm.size else 0.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f"{cm[i,j]:d}", ha="center", va="center",
                     color="white" if cm[i,j] > thresh else "black")
    plt.ylabel("True label"); plt.xlabel("Predicted label")
    plt.tight_layout(); ensure_dir(os.path.dirname(out_path)); plt.savefig(out_path, bbox_inches="tight"); plt.close(fig)

def figure_confusion_matrix(cm, class_names, title="Confusion Matrix"):
    fig = plt.figure(figsize=(5,4), dpi=140)
    plt.imshow(cm, interpolation="nearest"); plt.title(title); plt.colorbar()
    ax = plt.gca(); ax.set_xticks(range(len(class_names))); ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45); ax.set_yticklabels(class_names)
    thresh = cm.max()/2. if cm.size else 0.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f"{cm[i,j]:d}", ha="center", va="center",
                     color="white" if cm[i,j] > thresh else "black")
    plt.ylabel("True label"); plt.xlabel("Predicted label")
    plt.tight_layout()
    return fig

@torch.no_grad()
def evaluate(model, loader, device, split_name, out_dir, class_names, tb_writer: Optional[SummaryWriter]=None, epoch:int=0):
    model.eval()
    ys, ps = [], []
    for x, m, y, _ in loader:
        x, m, y = x.to(device, non_blocking=True), m.to(device).float(), y.to(device)
        logits = model(x, m); pred = logits.argmax(1)
        ys += y.tolist(); ps += pred.tolist()

    acc = accuracy_score(ys, ps)
    precision_bin, recall_bin, f1_bin, _ = precision_recall_fscore_support(ys, ps, average="binary", zero_division=0)
    precision_c, recall_c, f1_c, _ = precision_recall_fscore_support(ys, ps, average=None, zero_division=0)
    cm = confusion_matrix(ys, ps)

    # Save to disk
    save_confusion_matrix(cm, class_names, os.path.join(out_dir, f"{split_name.lower()}_confusion_matrix.png"))
    with open(os.path.join(out_dir, f"{split_name.lower()}_classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(classification_report(ys, ps, target_names=class_names, zero_division=0))

    # Log to TensorBoard
    if tb_writer is not None:
        tb_writer.add_scalar(f"{split_name}/accuracy", acc, epoch)
        tb_writer.add_scalar(f"{split_name}/precision", precision_bin, epoch)
        tb_writer.add_scalar(f"{split_name}/recall", recall_bin, epoch)
        tb_writer.add_scalar(f"{split_name}/f1", f1_bin, epoch)
        # per-class
        for i, name in enumerate(class_names):
            tb_writer.add_scalar(f"{split_name}/per_class_f1/{name}", float(f1_c[i]), epoch)
            tb_writer.add_scalar(f"{split_name}/per_class_precision/{name}", float(precision_c[i]), epoch)
            tb_writer.add_scalar(f"{split_name}/per_class_recall/{name}", float(recall_c[i]), epoch)
        # confusion matrix image
        fig = figure_confusion_matrix(cm, class_names, title=f"{split_name} Confusion")
        tb_writer.add_figure(f"{split_name}/confusion_matrix", fig, epoch)
        plt.close(fig)

    return dict(
        accuracy=float(acc),
        precision=float(precision_bin),
        recall=float(recall_bin),
        f1=float(f1_bin),
        cm=cm.tolist(),
        per_class=dict(
            precision=[float(x) for x in precision_c],
            recall=[float(x) for x in recall_c],
            f1=[float(x) for x in f1_c],
            names=class_names
        )
    )

class EarlyStopper:
    def __init__(self, patience=7, min_delta=1e-4):
        self.patience=patience; self.min_delta=min_delta; self.best=-1e9; self.count=0
    def step(self, metric):
        if metric > self.best + self.min_delta:
            self.best=metric; self.count=0; return False
        self.count+=1; return self.count>self.patience

def compute_class_weights(train_set):
    counts = np.zeros(len(train_set.classes), np.float32)
    for _,_,y,_ in train_set: counts[y]+=1
    probs = counts / max(1, counts.sum())
    maxp, minp = probs.max(), probs.min()
    imbalance = (maxp - minp)/maxp if maxp>0 else 0.0
    weights = (1.0/(counts+1e-6)) * (counts.sum()/len(counts))
    return torch.tensor(weights, dtype=torch.float32), float(imbalance), counts

def cosine_lr(optimizer, base_lr, warmup_steps, total_steps):
    def f(step):
        if step < warmup_steps: return (step+1)/max(1,warmup_steps)
        t = (step - warmup_steps)/max(1,total_steps-warmup_steps)
        return 0.5*(1.0+math.cos(math.pi*t))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)

# =============================
# Robust MobileNetV3 loader (silent)
# =============================
def build_mnv3_large_from_ckpt(state):
    stem_key = "features.0.0.weight"
    stem_out = state[stem_key].shape[0] if stem_key in state and hasattr(state[stem_key], "shape") else None
    width_mult = 2.0 if stem_out == 32 else 1.0
    print(f"[INFO] Guessed width_mult={width_mult} from checkpoint (stem_out={stem_out})")
    model = mobilenet_v3_large(weights=None, width_mult=width_mult)
    if model.classifier[-1].out_features != 2:
        in_feats = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_feats, 2)
    return model

def safe_load(model, state):
    model_sd = model.state_dict()
    filtered = {}
    for k, v in state.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            filtered[k] = v
    model.load_state_dict(filtered, strict=False)
    loaded = len(filtered)
    skipped = len(state) - loaded
    missing = len(model_sd) - loaded
    print(f"[INFO] Loaded {loaded} matching keys | Skipped {skipped} mismatches | Missing {missing}")

# =============================
# Diagnostics helper
# =============================
def diagnose(train_loss, train_acc, val_acc, val_f1, per_class, epoch, log_f, tb_writer: Optional[SummaryWriter]=None):
    gap = train_acc - val_acc
    f1s = per_class["f1"]
    names = per_class["names"]
    f1_gap = max(f1s) - min(f1s) if len(f1s) > 1 else 0.0

    msg = [f"[MONITOR] Epoch {epoch:02d}: gen_gap={gap:.3f}, train_loss={train_loss:.4f}, train_acc={train_acc:.3f}, val_acc={val_acc:.3f}, val_f1={val_f1:.3f}"]

    if gap > 0.12 and val_f1 < 0.9:
        msg.append("  • Possible OVERFITTING: high train vs val gap. Consider stronger aug, more dropout, or earlier stop.")
    if train_acc < 0.80 and val_acc < 0.80:
        msg.append("  • Possible UNDERFITTING: both accuracies low. Consider longer training, unfreezing more layers, or higher lr.")
    if f1_gap > 0.20:
        worst = names[int(np.argmin(f1s))]
        best = names[int(np.argmax(f1s))]
        msg.append(f"  • Possible CLASS BIAS: F1 gap {f1_gap:.2f} (best={best}:{max(f1s):.2f}, worst={worst}:{min(f1s):.2f}). Consider class-balanced sampling or focal loss.")

    text = "\n".join(msg)
    print(text)
    with open(log_f, "a", encoding="utf-8") as f:
        f.write(text + "\n")

    # Log diagnostics to TensorBoard
    if tb_writer is not None:
        tb_writer.add_scalar("diagnostics/generalization_gap", gap, epoch)
        tb_writer.add_scalar("diagnostics/class_f1_gap", f1_gap, epoch)

# =============================
# MAIN
# =============================
def main():
    warnings.filterwarnings("ignore", category=FutureWarning)

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_dir(OUT_DIR)
    print(f"[INFO] Device: {device}")

    # TensorBoard writer
    tb = SummaryWriter(log_dir=TB_LOG_DIR)

    # ---- Fit featurizer on TRAIN metadata ----
    train_meta_map = load_split_global_meta(DATA_ROOT, "train")
    print(f"[INFO] Train metadata entries: {len(train_meta_map)}")
    featurizer = MetaFeaturizer()
    featurizer.fit(list(train_meta_map.values()) or [{}])
    print(f"[INFO] Meta keys ({featurizer.dim}): {featurizer.keys}")
    # Log meta info
    tb.add_text("meta/keys", ", ".join(featurizer.keys), 0)

    # ---- Datasets / Loaders ----
    dtrain = ImageWithMetadataDataset(DATA_ROOT, "train", featurizer, img_size=224, augment=True)
    dval   = ImageWithMetadataDataset(DATA_ROOT, "validation", featurizer, img_size=224, augment=False)
    dtest  = ImageWithMetadataDataset(DATA_ROOT, "test", featurizer, img_size=224, augment=False)

    class_names = dtrain.classes
    print(f"[INFO] Classes: {class_names}")
    tb.add_text("meta/classes", ", ".join(class_names), 0)

    train_loader = DataLoader(dtrain, batch_size=16, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(dval,   batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(dtest,  batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    # ---- Build base from checkpoint + safe load ----
    print(f"[INFO] Loading base weights: {BASE_WEIGHTS}")
    state = torch.load(BASE_WEIGHTS, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state: state = state["state_dict"]

    base = build_mnv3_large_from_ckpt(state)
    safe_load(base, state)

    # Determine FiLM channels from model
    film_channels = base.classifier[0].in_features
    print(f"[INFO] FiLM channels inferred from model: {film_channels}")

    # ---- Wrap with HyperNet FiLM ----
    model = HNMobileNetV3Large(base, meta_dim=featurizer.dim, film_channels=film_channels, freeze_early=True).to(device)

    # ---- Loss, optimizer, schedule, regularization ----
    class_weights, imbalance, counts = compute_class_weights(dtrain)
    use_weights = imbalance > 0.10
    if use_weights:
        print(f"[INFO] Class imbalance detected ({imbalance:.1%}), enabling class-weighted loss. Counts={counts.tolist()}")
    tb.add_scalar("data/class_imbalance", imbalance, 0)
    tb.add_histogram("data/class_counts", counts, 0)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if use_weights else None,
                                    label_smoothing=0.05)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=1e-4)

    total_steps = max(1, len(train_loader)) * 40
    scheduler = cosine_lr(optimizer, base_lr=1e-4,
                          warmup_steps=max(10, len(train_loader)*1),
                          total_steps=total_steps)

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type=="cuda"))

    early = EarlyStopper(patience=7, min_delta=1e-4)

    best_f1 = -1.0
    best_path = os.path.join(OUT_DIR, "hn_mobilenetv3_large_best.pth")
    last_path = os.path.join(OUT_DIR, "hn_mobilenetv3_large_last.pth")
    log_path  = os.path.join(OUT_DIR, "training_log.csv")
    mon_log   = os.path.join(OUT_DIR, "monitoring_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,train_acc,val_acc,val_precision,val_recall,val_f1\n")
    with open(mon_log, "w", encoding="utf-8") as f:
        f.write("Diagnostics per epoch\n")

    mixup_alpha = 0.1
    max_epochs = 40
    print("\n[INFO] Starting training...\n")
    global_step = 0
    for epoch in range(1, max_epochs+1):
        t0 = time.time()
        model.train()
        losses=[]; ys=[]; ps=[]

        for batch_idx, (x, m, y, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}", ncols=100)):
            x, m, y = x.to(device, non_blocking=True), m.to(device).float(), y.to(device)

            x_in, m_in, y_in, lam, idx = mixup_data(x, m, y, alpha=mixup_alpha)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16 if device.type=="cuda" else torch.bfloat16):
                logits = model(x_in, m_in)
                loss = mixup_criterion(criterion, logits, y_in, lam, idx) if mixup_alpha>0 else criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            scheduler.step()

            losses.append(loss.item())

            # batch accuracy (approx with unmixed labels)
            with torch.no_grad():
                pred = logits.argmax(1)
                ys += y.tolist(); ps += pred.tolist()

            # --- TensorBoard batch logging (lightweight) ---
            tb.add_scalar("train/batch_loss", loss.item(), global_step)
            if batch_idx % 10 == 0:
                batch_acc = accuracy_score(y.tolist(), pred.tolist())
                tb.add_scalar("train/batch_acc", batch_acc, global_step)

            global_step += 1

        # Gradual unfreeze
        if epoch == 6:
            print("[INFO] Unfreezing entire backbone for fine-tuning.")
            model.unfreeze_all()

        train_acc = accuracy_score(ys, ps)
        val_metrics = evaluate(model, val_loader, device, "Val", OUT_DIR, class_names, tb_writer=tb, epoch=epoch)

        dt = time.time()-t0
        epoch_loss = float(np.mean(losses))
        print(f"Epoch {epoch:02d} | {dt:.1f}s | TrainLoss {epoch_loss:.4f} Acc {train_acc:.4f} | "
              f"Val Acc {val_metrics['accuracy']:.4f} P {val_metrics['precision']:.4f} "
              f"R {val_metrics['recall']:.4f} F1 {val_metrics['f1']:.4f}")

        # --- TensorBoard epoch logging ---
        tb.add_scalar("train/epoch_loss", epoch_loss, epoch)
        tb.add_scalar("train/epoch_acc",  train_acc,  epoch)
        tb.add_scalar("opt/lr", optimizer.param_groups[0]["lr"], epoch)

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{epoch_loss:.6f},{train_acc:.6f},{val_metrics['accuracy']:.6f},"
                    f"{val_metrics['precision']:.6f},{val_metrics['recall']:.6f},{val_metrics['f1']:.6f}\n")

        # Epoch-level diagnostics (overfit/underfit/bias)
        diagnose(epoch_loss, train_acc, val_metrics['accuracy'], val_metrics['f1'],
                 val_metrics['per_class'], epoch, mon_log, tb_writer=tb)

        torch.save(model.state_dict(), last_path)
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), best_path)
            print(f"[INFO] New best (F1={best_f1:.4f}) saved to {best_path}")

        if early.step(val_metrics["f1"]):
            print("[INFO] Early stopping triggered.")
            break

    # ---- Test with best ----
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        print(f"[INFO] Loaded best checkpoint for Test: {best_path}")
    test_metrics = evaluate(model, test_loader, device, "Test", OUT_DIR, class_names, tb_writer=tb, epoch=epoch+1)

    print("\n=== FINAL TEST METRICS ===")
    print(f"Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall:    {test_metrics['recall']:.4f}")
    print(f"F1-score:  {test_metrics['f1']:.4f}")
    print(f"Confusion Matrix:\n{np.array(test_metrics['cm'])}")

    with open(os.path.join(OUT_DIR, "final_test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"[INFO] Outputs saved to: {OUT_DIR}")

    tb.close()

if __name__ == "__main__":
    main()

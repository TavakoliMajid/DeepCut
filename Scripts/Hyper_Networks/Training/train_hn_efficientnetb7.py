# train_hn_efficientnet_b7.py
import os, json, warnings
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
from PIL import Image
from torchvision import models, transforms
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
# <-- MODIFIED: Added accuracy_score
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, log_loss, accuracy_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================
# FIXED PATHS (edit ONLY if needed)
# ============================
DATA_ROOT = Path(r"D:\\cologne\\RES_NET\\New_Data\\Tissue_Brain")
INIT_BACKBONE_PATH = Path(r"D:\\cologne\\Models\\Models\\EfficientNet_b7_March17.pth")  # <-- set your fine-tuned B7 here
OUTDIR = Path(r"D:\\cologne\\HyperNetwork\\Nueral_Network\\efficientnet_b7_hn")

# ============================
# Training/output defaults
# ============================
IMG_SIZE = 600          # B7 likes 600px (you can try 528 for speed)
EPOCHS = 30
BATCH_SIZE = 6          # B7 is heavy; keep this modest
NUM_WORKERS = 4
LR = 1e-4
WEIGHT_DECAY = 1e-4     # <-- MODIFIED: Increased to 1e-4 for regularization
PATIENCE = 15           
HEAD_DROPOUT = 0.15     # <-- MODIFIED: Reduced to 0.15 to reduce overfitting
USE_AMP = True
SEED = 42

# ---- Eval / calibration ----
MIN_PRECISION = 0.80    # <-- MODIFIED: Reduced to 0.80 to allow more recall
USE_TTA = True          

# ---- Uncertainty band (off) ----
USE_UNCERTAIN_BAND = False
UNCERTAIN_BAND = 0.05

# ---- Label smoothing ----
LABEL_SMOOTH = 0.02

# ---- EMA ----
USE_EMA = True
EMA_DECAY = 0.999

# ====== Global meta normalization ======
META_MEAN = None
META_STD  = None

# ============================
# Helpers
# ============================
def set_seed(seed:int=42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def count_pos_weight(labels: np.ndarray) -> float:
    pos = (labels==1).sum(); neg = (labels==0).sum()
    return 1.2 if pos==0 else max(1.2, float(neg)/float(pos))  # <-- MODIFIED: Increased pos_weight to 1.2

def class_weights_per_sample(labels: np.ndarray) -> np.ndarray:
    counts = np.bincount(labels.astype(int), minlength=2).astype(float)
    counts[counts==0] = 1.0
    w = 1.0 / counts
    return w[labels.astype(int)]

CUT_METHOD_VOCAB: Dict[str, int] = {}
def _cut_method_id(name: str) -> int:
    n = (name or "").strip().lower()
    if n not in CUT_METHOD_VOCAB:
        CUT_METHOD_VOCAB[n] = len(CUT_METHOD_VOCAB)
    return CUT_METHOD_VOCAB[n]

def _parse_magnification(mag: Any) -> float:
    if isinstance(mag, (int, float)): return float(mag)
    if isinstance(mag, str):
        m = "".join(ch for ch in mag if ch.isdigit() or ch == ".")
        return float(m) if m else 0.0
    return 0.0

def build_metadata_list(md: dict) -> list:
    return [
        _parse_magnification(md.get("magnification", "0x")),
        float(md.get("laser_power", 0)),
        float(md.get("laser_speed", 0)),
        float(md.get("laser_aperture", 0)),
        float(md.get("final_pulse_power", 0)),
        float(_cut_method_id(md.get("cut_method", ""))),
        float(md.get("laser_frequency", 0)),
    ]

def load_split_items(json_path: Path, data_root: Path) -> list:
    data = json.load(open(json_path, "r"))
    def _abs(p: str) -> str:
        p = p.replace("\\", os.sep)
        pp = Path(p)
        return str(pp.resolve() if pp.is_absolute() else (data_root / pp).resolve())

    if isinstance(data, list):
        items = []
        for it in data:
            p = it.get("image_path") or it.get("path")
            if not p: continue
            items.append({
                "image_path": _abs(p),
                "label": int(it["label"]),
                "metadata": it["metadata"],
            })
        return items

    if isinstance(data, dict):
        items = []
        for rel_or_abs, md in data.items():
            abs_path = _abs(rel_or_abs)
            lp = str(rel_or_abs).lower()
            label = 1 if "bad_cut" in lp else 0
            items.append({
                "image_path": abs_path,
                "label": label,
                "metadata": build_metadata_list(md if isinstance(md, dict) else {}),
            })
        return items
    raise ValueError(f"Unsupported JSON structure in {json_path}")

# ============================
# Dataset
# ============================
class JsonDataset(Dataset):
    def __init__(self, items: List[Dict[str,Any]], img_size:int=600, augment:bool=True):
        self.items = items
        pil_augs = []
        if augment:
            pil_augs = [
                transforms.RandomHorizontalFlip(0.6),
                transforms.RandomVerticalFlip(0.6),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.25, contrast=0.25),
            ]
        tensor_augs = []
        if augment:
            tensor_augs = [transforms.RandomErasing(p=0.12, scale=(0.02, 0.06), ratio=(0.3, 3.3))]
        self.tf = transforms.Compose(
            [transforms.Resize((img_size, img_size))] + pil_augs +
            [transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])] + tensor_augs
        )
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        it = self.items[i]; img_path = it["image_path"]
        if not os.path.exists(img_path):
            return self.__getitem__(np.random.randint(0, len(self.items)))
        try:
            img = Image.open(img_path).convert("RGB")
            x = self.tf(img)
        except Exception:
            return self.__getitem__(np.random.randint(0, len(self.items)))
        meta_arr = np.array(it["metadata"], dtype=np.float32)
        global META_MEAN, META_STD
        if META_MEAN is not None and META_STD is not None:
            meta_arr = (meta_arr - META_MEAN) / META_STD
        y = torch.tensor(it["label"], dtype=torch.float32)
        meta = torch.tensor(meta_arr, dtype=torch.float32)
        return x, y, meta

# ============================
# EfficientNet-B7 backbone + HyperNet FiLM
# ============================
class FeatureEffB7(nn.Module):
    def __init__(self):
        super().__init__()
        # torchvision >=0.13: models.efficientnet_b7
        self.m = models.efficientnet_b7(weights=None)
        # features live in self.m.features, classifier is self.m.classifier
        self.features = self.m.features
        self.out_channels = 2560  # B7 last feature channels
    def forward(self, x):
        return self.features(x)  # (B, 2560, H/32, W/32)

class FiLMAdapter(nn.Module):
    def forward(self, feat: torch.Tensor, g: torch.Tensor, b: torch.Tensor):
        B,C,H,W = feat.shape
        return feat * g.view(B,C,1,1) + b.view(B,C,1,1)

class HyperNet(nn.Module):
    def __init__(self, meta_dim:int, out_channels:int, hidden:int=160, dropout:float=0.5):
        super().__init__()
        self.out_channels = out_channels
        self.mlp = nn.Sequential(
            nn.Linear(meta_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 2*out_channels)
        )
        with torch.no_grad():
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)
    def forward(self, meta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.mlp(meta); C = self.out_channels
        gamma = h[:,:C] + 1.0; beta = h[:,C:]
        return gamma, beta

class HyperFiLMEffB7(nn.Module):
    def __init__(self, meta_dim:int, freeze_early:bool=True, head_dropout:float=0.15):
        super().__init__()
        self.backbone = FeatureEffB7()
        self.hypernet = HyperNet(meta_dim=meta_dim, out_channels=self.backbone.out_channels, hidden=256, dropout=0.5)
        self.adapter = FiLMAdapter()
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.head = nn.Sequential(nn.Dropout(head_dropout),
                                  nn.Linear(self.backbone.out_channels, 512),
                                  nn.ReLU(inplace=True),
                                  nn.Linear(512, 1))
        if freeze_early:
            for p in self.backbone.features.parameters():
                p.requires_grad = False
    def forward(self, x, meta):
        f = self.backbone(x); g,b = self.hypernet(meta)
        f = self.adapter(f, g, b)
        x = self.gap(f).flatten(1)
        return self.head(x).squeeze(1)

# ============================
# Load your fine-tuned B7 features
# ============================
def load_backbone_features_from_ckpt(model: HyperFiLMEffB7, ckpt_path: Path):
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    new_sd = {}
    for k,v in sd.items():
        kk = k.replace("module.", "")
        # common naming in EfficientNet checkpoints:
        # "features.*", sometimes "backbone.features.*"
        if kk.startswith("features."):
            new_sd[f"backbone.features.{kk.split('features.',1)[1]}"] = v
        if kk.startswith("backbone.features."):
            new_sd[kk] = v
        # sometimes whole model saved under "m.features.*"
        if kk.startswith("m.features."):
            new_sd["backbone."+kk[2:]] = v
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    print(f"[Init] Loaded feature layers from {ckpt_path}")
    if missing:    print(f"[Init] Missing keys (ok): {len(missing)}")
    if unexpected: print(f"[Init] Unexpected keys (ignored): {len(unexpected)}")

# ============================
# Loss / Metrics / Calibration
# ============================
def bce_logits_loss(logits: torch.Tensor, targets: torch.Tensor, pos_weight: float, label_smooth: float = 0.0):
    if label_smooth > 0.0:
        targets = targets*(1.0 - label_smooth) + 0.5*label_smooth
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=logits.device))(logits, targets)

def _tta_variants(xb: torch.Tensor):
    return [xb, torch.flip(xb,(3,)), torch.flip(xb,(2,)), torch.flip(xb,(2,3))]

@torch.no_grad()
def evaluate(model, loader, device, threshold: float = 0.5, temperature: float = 1.0, use_tta: bool=False):
    model.eval()
    all_logits, all_targets = [], []
    for xb,yb,mb in loader:
        xb,yb,mb = xb.to(device), yb.to(device), mb.to(device)
        if use_tta:
            logits = torch.stack([model(xa, mb) for xa in _tta_variants(xb)],0).mean(0)
        else:
            logits = model(xb, mb)
        if temperature != 1.0: logits = logits/temperature
        all_logits.append(logits.detach().cpu().numpy())
        all_targets.append(yb.detach().cpu().numpy())
    logits = np.concatenate(all_logits); targets = np.concatenate(all_targets)
    probs = 1/(1+np.exp(-logits))
    preds = (probs >= threshold).astype(np.int32)
    p, r, f1, _ = precision_recall_fscore_support(targets, preds, average="binary", zero_division=0)
    acc = accuracy_score(targets, preds)  # <-- ADDED
    cm = confusion_matrix(targets, preds)
    nll = log_loss(targets, probs, labels=[0,1])
    # <-- MODIFIED: Added accuracy to the returned dict
    return {"precision":p, "recall":r, "f1":f1, "accuracy":acc, "cm":cm, "nll":nll, "probs":probs, "targets":targets, "logits":logits}

def search_temperature(logits: np.ndarray, targets: np.ndarray):
    Ts = np.linspace(0.05, 3.0, 60)
    best_T, best_nll = 1.0, 1e9
    for T in Ts:
        probs = 1/(1+np.exp(-(logits/T))); nll = log_loss(targets, probs, labels=[0,1])
        if nll < best_nll: best_nll, best_T = nll, T
    return float(best_T)

def search_threshold_for_precision(probs: np.ndarray, targets: np.ndarray, min_precision=0.80):
    best_t, best_recall = 0.5, -1
    for t in np.linspace(0.2, 0.99, 160):
        preds = (probs >= t).astype(np.int32)
        p, r, _, _ = precision_recall_fscore_support(targets, preds, average="binary", zero_division=0)
        if p >= min_precision and r > best_recall:
            best_recall, best_t = r, float(t)
    if best_recall < 0:  # fallback: max F1
        best_t, best_f1 = 0.5, -1
        for t in np.linspace(0.2, 0.99, 160):
            preds = (probs >= t).astype(np.int32)
            _,_,f1,_ = precision_recall_fscore_support(targets, preds, average="binary", zero_division=0)
            if f1 > best_f1: best_f1, best_t = f1, float(t)
    return best_t, max(best_recall, 0.0)

def decide_with_band(p, thr, band=0.05):
    if p >= thr + band: return 1
    if p <= thr - band: return 0
    return -1

# ============================
# Plotting
# ============================
# <-- MODIFIED: Function signature and content updated to plot accuracy
def plot_metric_curves(epochs, tr_prec, tr_rec, tr_f1, tr_acc, va_prec, va_rec, va_f1, va_acc, save_path: Path):
    plt.figure(figsize=(10,7))
    plt.plot(epochs, tr_prec, marker='o', linestyle="--", alpha=0.7, label='Train Precision')
    plt.plot(epochs, tr_rec, marker='o', linestyle="--", alpha=0.7, label='Train Recall')
    plt.plot(epochs, tr_f1, marker='o', linestyle="--", alpha=0.7, label='Train F1')
    plt.plot(epochs, tr_acc, marker='o', label='Train Accuracy')
    plt.plot(epochs, va_prec, marker='s', alpha=0.7, label='Val Precision')
    plt.plot(epochs, va_rec, marker='s', alpha=0.7, label='Val Recall')
    plt.plot(epochs, va_f1, marker='s', label='Val F1')
    plt.plot(epochs, va_acc, marker='s', label='Val Accuracy')
    plt.xlabel('Epoch'); plt.ylabel('Score'); plt.title('Train vs Val Metrics Over Epochs')
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close()

def plot_confusion_matrix(cm: np.ndarray, title: str, save_path: Path):
    fig, ax = plt.subplots(figsize=(6,5)); im = ax.imshow(cm, interpolation='nearest'); ax.figure.colorbar(im, ax=ax)
    classes = ['Good Cut', 'Bad Cut']; tick = np.arange(len(classes))
    ax.set_xticks(tick); ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(tick); ax.set_yticklabels(classes)
    ax.set_ylabel('True Label'); ax.set_xlabel('Predicted Label'); ax.set_title(title)
    thr = cm.max()/2.0 if cm.max()>0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                    color="white" if cm[i, j] > thr else "black")
    fig.tight_layout(); plt.savefig(save_path, dpi=300); plt.close()

# ============================
# EMA helper
# ============================
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.data.clone() for n,p in model.named_parameters() if p.requires_grad}
    def update(self, model):
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1-self.decay)
    def apply_to(self, model):
        for n, p in model.named_parameters():
            if n in self.shadow:
                p.data.copy_(self.shadow[n])

@torch.no_grad()
def evaluate_with_ema(model, ema: 'EMA|None', loader, device, threshold=0.5, temperature=1.0, use_tta=False):
    if ema is None: return evaluate(model, loader, device, threshold, temperature, use_tta)
    backup = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
    ema.apply_to(model); out = evaluate(model, loader, device, threshold, temperature, use_tta)
    model.load_state_dict(backup, strict=True); return out

# ============================
# Training
# ============================
def main():
    set_seed(SEED); ensure_dir(DATA_ROOT); ensure_dir(OUTDIR)

    # ---- Load splits ----
    train_json = DATA_ROOT / "train.json"; val_json = DATA_ROOT / "validation.json"; test_json = DATA_ROOT / "test.json"
    for p in [train_json, val_json, test_json]:
        if not p.exists(): raise FileNotFoundError(f"Missing split file: {p}")

    train_items = load_split_items(train_json, DATA_ROOT)
    val_items   = load_split_items(val_json, DATA_ROOT)
    test_items  = load_split_items(test_json, DATA_ROOT)
    if len(train_items)==0 or len(val_items)==0 or len(test_items)==0:
        raise RuntimeError("One of the splits is empty after parsing. Check JSON structure/paths.")

    # ---- Meta normalization (fit on train) ----
    global META_MEAN, META_STD
    metas_train = np.array([it["metadata"] for it in train_items], dtype=np.float32)
    META_MEAN = metas_train.mean(0); META_STD = np.clip(metas_train.std(0), 1e-6, None)

    # ---- meta_dim ----
    meta_dim = len(train_items[0]["metadata"]); print(f"[Info] meta_dim = {meta_dim}")

    # ---- Class stats ----
    y_train = np.array([it["label"] for it in train_items], dtype=np.int64)
    cls_counts = np.bincount(y_train, minlength=2)
    print(f"[Info] train class counts -> good={int(cls_counts[0])}  bad={int(cls_counts[1])}")

    # ---- Sampler: class-balanced ----
    sample_weights = class_weights_per_sample(y_train)
    sampler = WeightedRandomSampler(weights=torch.as_tensor(sample_weights, dtype=torch.double),
                                      num_samples=len(sample_weights), replacement=True)

    # ---- DataLoaders ----
    train_dl = DataLoader(JsonDataset(train_items, img_size=IMG_SIZE, augment=True),
                          batch_size=BATCH_SIZE, sampler=sampler, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)
    train_eval_dl = DataLoader(JsonDataset(train_items, img_size=IMG_SIZE, augment=False),
                               batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    val_dl   = DataLoader(JsonDataset(val_items,   img_size=IMG_SIZE, augment=False),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_dl  = DataLoader(JsonDataset(test_items,  img_size=IMG_SIZE, augment=False),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # ---- Model ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HyperFiLMEffB7(meta_dim=meta_dim, freeze_early=True, head_dropout=HEAD_DROPOUT).to(device)
    load_backbone_features_from_ckpt(model, INIT_BACKBONE_PATH)

    # ---- AMP ----
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)
        autocast_ctx = torch.autocast; autocast_kw = dict(device_type='cuda', dtype=torch.float16, enabled=USE_AMP)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
        autocast_ctx = torch.cuda.amp.autocast; autocast_kw = dict(enabled=USE_AMP)

    # ---- Loss/optim/sched ----
    pos_w = count_pos_weight(y_train); print(f"[Info] pos_weight (train): {pos_w:.3f}")
    params = list(model.hypernet.parameters()) + list(model.head.parameters())
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    ema = EMA(model, EMA_DECAY) if USE_EMA else None

    # ---- History ----
    # <-- MODIFIED: Added train_acc and val_acc to history
    hist = {"epoch": [], "train_p": [], "train_r": [], "train_f1": [], "train_acc": [],
            "val_p": [], "val_r": [], "val_f1": [], "val_acc": []}
    best_val_f1 = -1.0; patience = PATIENCE

    # ---- Train ----
    for epoch in range(1, EPOCHS+1):
        model.train()
        for xb,yb,mb in train_dl:
            xb,yb,mb = xb.to(device), yb.to(device), mb.to(device)
            opt.zero_grad(set_to_none=True)
            with autocast_ctx(**autocast_kw):
                logits = model(xb, mb)
                loss = bce_logits_loss(logits, yb, pos_weight=pos_w, label_smooth=LABEL_SMOOTH)
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update()
            if ema: ema.update(model)
        sched.step()

        train_metrics = evaluate_with_ema(model, ema, train_eval_dl, device, threshold=0.5, temperature=1.0, use_tta=USE_TTA)
        val_metrics   = evaluate_with_ema(model, ema, val_dl,        device, threshold=0.5, temperature=1.0, use_tta=USE_TTA)

        hist["epoch"].append(epoch)
        # <-- MODIFIED: Loop to append metrics including accuracy
        for k,src in [("train_p","precision"),("train_r","recall"),("train_f1","f1"),("train_acc", "accuracy")]:
            hist[k].append(float(train_metrics[src]))
        for k,src in [("val_p","precision"),("val_r","recall"),("val_f1","f1"),("val_acc", "accuracy")]:
            hist[k].append(float(val_metrics[src]))

        # <-- MODIFIED: Print statement now includes accuracy
        print(f"Epoch {epoch:03d} | lr={sched.get_last_lr()[0]:.2e} | "
              f"Train Acc={train_metrics['accuracy']:.3f} F1={train_metrics['f1']:.3f} P={train_metrics['precision']:.3f} R={train_metrics['recall']:.3f} | "
              f"Val Acc={val_metrics['accuracy']:.3f} F1={val_metrics['f1']:.3f} P={val_metrics['precision']:.3f} R={val_metrics['recall']:.3f} NLL={val_metrics['nll']:.3f}")

        if val_metrics["f1"] > best_val_f1 + 1e-4:
            best_val_f1 = val_metrics["f1"]; patience = PATIENCE
            if ema:
                backup = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
                ema.apply_to(model); torch.save(model.state_dict(), OUTDIR/"model_best.pt")
                model.load_state_dict(backup, strict=True)
            else:
                torch.save(model.state_dict(), OUTDIR/"model_best.pt")
        else:
            patience -= 1
            if patience <= 0:
                print("Early stopping."); break

    # ---- Load best & calibrate ----
    model.load_state_dict(torch.load(OUTDIR/"model_best.pt", map_location=device))
    val_eval = evaluate_with_ema(model, ema if USE_EMA else None, val_dl, device, threshold=0.5, temperature=1.0, use_tta=USE_TTA)
    T = search_temperature(val_eval["logits"], val_eval["targets"])
    val_probs_T = 1/(1+np.exp(-(val_eval["logits"]/T)))
    thr, _ = search_threshold_for_precision(val_probs_T, val_eval["targets"], min_precision=MIN_PRECISION)

    # ---- Test ----
    test_metrics = evaluate_with_ema(model, ema if USE_EMA else None, test_dl, device, threshold=thr, temperature=T, use_tta=USE_TTA)

    # ---- Uncertainty (optional) ----
    abstain_val = abstain_test = 0
    if USE_UNCERTAIN_BAND:
        val_preds_band = np.array([decide_with_band(p, thr, UNCERTAIN_BAND) for p in val_probs_T])
        abstain_val = int((val_preds_band == -1).sum())
        test_eval_raw = evaluate_with_ema(model, ema if USE_EMA else None, test_dl, device, threshold=0.5, temperature=T, use_tta=USE_TTA)
        test_probs_T = 1/(1+np.exp(-(test_eval_raw["logits"])))
        test_preds_band = np.array([decide_with_band(p, thr, UNCERTAIN_BAND) for p in test_probs_T])
        abstain_test = int((test_preds_band == -1).sum())

    # ---- Save artifacts ----
    ensure_dir(OUTDIR)
    np.savetxt(OUTDIR/"val_confusion_matrix.csv", val_eval["cm"], fmt="%d", delimiter=",")
    np.savetxt(OUTDIR/"test_confusion_matrix.csv", test_metrics["cm"], fmt="%d", delimiter=",")
    with open(OUTDIR/"summary.json","w") as f:
        # <-- MODIFIED: summary.json now includes accuracy
        json.dump({
            "temperature": T, "threshold": thr,
            "precision_target": MIN_PRECISION,
            "uncertainty_band": (UNCERTAIN_BAND if USE_UNCERTAIN_BAND else 0.0),
            "use_tta": USE_TTA,
            "val": {k:(float(v) if not hasattr(v,"tolist") else v.tolist()) for k,v in val_eval.items() if k in ["precision","recall","f1","nll","accuracy"]},
            "test": {k:(float(v) if not hasattr(v,"tolist") else v.tolist()) for k,v in test_metrics.items() if k in ["precision","recall","f1","nll","accuracy"]},
            "abstain": {"val": abstain_val, "test": abstain_test}
        }, f, indent=2)

    with open(OUTDIR/"meta_norm.json","w") as f:
        json.dump({"mean": META_MEAN.tolist(), "std": META_STD.tolist()}, f, indent=2)

    # ---- Plots ----
    # <-- MODIFIED: Call to plot_metric_curves now includes accuracy data
    plot_metric_curves(hist["epoch"], hist["train_p"], hist["train_r"], hist["train_f1"], hist["train_acc"],
                       hist["val_p"], hist["val_r"], hist["val_f1"], hist["val_acc"], OUTDIR/"train_val_metrics_over_epochs.png")

    def _plot_cm(cm, title, path): plot_confusion_matrix(cm, title, path)
    _plot_cm(val_eval['cm'],  "Validation Confusion Matrix", OUTDIR/"val_confusion_matrix_plot.png")
    _plot_cm(test_metrics['cm'], "Test Confusion Matrix", OUTDIR/"test_confusion_matrix_plot.png")

    print(f"[Done] Saved to {OUTDIR}")
    print(f"[Calib] T={T:.3f}  Thr={thr:.3f}  (min_precision={MIN_PRECISION}, TTA={USE_TTA})")
    if USE_UNCERTAIN_BAND:
        print(f"[Uncertainty] band=±{UNCERTAIN_BAND:.02f} | abstain: val={abstain_val}, test={abstain_test}")
    # <-- MODIFIED: Final printout now includes test accuracy
    print(f"[Test ] Acc={test_metrics['accuracy']:.3f} F1={test_metrics['f1']:.3f}  P={test_metrics['precision']:.3f}  R={test_metrics['recall']:.3f}")
    print("[Plots] train_val_metrics_over_epochs.png, val_confusion_matrix_plot.png, test_confusion_matrix_plot.png")
    print("[Saved] meta_norm.json, summary.json")

if __name__ == "__main__":
    main()
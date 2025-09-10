from __future__ import annotations
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
import yaml
from PIL import Image
import cv2
import importlib


class ModelAdapter:
    def __init__(self, model: nn.Module, device: torch.device,
                 preprocess: T.Compose, class_names: list[str], use_hn: bool):
        self.model = model.eval().to(device)
        self.device = device
        self.preprocess = preprocess
        self.class_names = class_names
        self.use_hn = use_hn

    @torch.inference_mode()
    def predict(self, bgr_img, metadata: dict | None = None):
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        x = self.preprocess(pil).unsqueeze(0).to(self.device)
        if self.use_hn:
            logits = self.model(x, metadata)
        else:
            logits = self.model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu()
        score, cls = torch.max(probs, dim=0)
        return self.class_names[int(cls)], float(score), probs.numpy()


def build_preprocess(cfg: dict) -> T.Compose:
    pp = cfg.get("preprocess", {})
    resize_shorter = pp.get("resize_shorter", 256)
    center_crop   = pp.get("center_crop", 224)
    mean = pp.get("mean", [0.485, 0.456, 0.406])
    std  = pp.get("std",  [0.229, 0.224, 0.225])
    return T.Compose([
        T.Resize(resize_shorter),
        T.CenterCrop(center_crop),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_pytorch_model(cfg: dict) -> nn.Module:
    loader = cfg.get("loader", "state_dict")
    arch = cfg.get("arch")
    weights_path = cfg.get("weights_path")
    num_classes = int(cfg.get("num_classes", 2))

    if loader == "torchscript":
        return torch.jit.load(weights_path, map_location="cpu")

    if arch == "efficientnet_b7":
        m = models.efficientnet_b7(weights=None)
        in_f = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_f, num_classes)
    elif arch == "resnet50":
        m = models.resnet50(weights=None)
        in_f = m.fc.in_features
        m.fc = nn.Linear(in_f, num_classes)
    elif arch == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(weights=None)
        in_f = m.classifier[3].in_features
        m.classifier[3] = nn.Linear(in_f, num_classes)
    else:
        raise ValueError(f"Unsupported arch: {arch}")

    sd = torch.load(weights_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    m.load_state_dict(sd, strict=False)
    return m


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Apple Silicon (MPS)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_adapter(cfg: dict) -> ModelAdapter:
    device = pick_device()
    preprocess = build_preprocess(cfg)
    class_names = cfg.get("class_names", ["Good_Cut", "Bad_Cut"])
    use_hn = bool(cfg.get("use_hypernetwork", False))

    if use_hn:
        # base backbone (no HN head)
        base = build_pytorch_model({**cfg, "use_hypernetwork": False})
        meta_conf = cfg.get("metadata", {})
        custom_cls = meta_conf.get("custom_model_class")
        if not custom_cls:
            raise RuntimeError("HN model requires metadata.custom_model_class in YAML")
        mod = importlib.import_module("custom_models")
        wrapper_cls = getattr(mod, custom_cls)
        model = wrapper_cls(
            base_model=base,
            meta_dim=len(meta_conf.get("schema", [])),
            num_classes=len(class_names)
        )
        # load the same weights if your wrapper expects them on its classifier;
        # else ensure your checkpoint already matches the wrapper.
        # (Here we assume your YAML weights_path points to wrapper-compatible weights.)
        # If needed, load again:
        sd = torch.load(cfg.get("weights_path"), map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        try:
            model.load_state_dict(sd, strict=False)
        except Exception:
            pass
    else:
        model = build_pytorch_model(cfg)

    return ModelAdapter(model=model, device=device, preprocess=preprocess,
                        class_names=class_names, use_hn=use_hn)


class ModelRegistry:
    def __init__(self, root: str | Path = "configs/models"):
        self.root = Path(root)
        self.models: dict[str, dict] = {}
        self.refresh()

    def refresh(self):
        self.models.clear()
        for yml in list(self.root.glob("*.yaml")) + list(self.root.glob("*.yml")):
            cfg = load_yaml(yml)
            name = cfg.get("name", yml.stem)
            self.models[name] = {"path": str(yml), "cfg": cfg}

    def get_names(self) -> list[str]:
        return sorted(self.models.keys())

    def get_cfg(self, name: str) -> dict:
        return self.models[name]["cfg"]

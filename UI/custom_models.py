from __future__ import annotations
from typing import Dict, Optional

import torch
import torch.nn as nn


def _prepare_metadata_vector(metadata: Dict[str, str] | None) -> torch.Tensor:
    if not metadata:
        return torch.zeros(1, 0)
    vec = []
    for _, v in metadata.items():
        try:
            vec.append(float(v))
        except Exception:
            vec.append((hash(v) % 1000) / 1000.0)
    return torch.tensor(vec, dtype=torch.float32).unsqueeze(0)  # [1, D]


class FiLMBlock(nn.Module):
    def __init__(self, feat_dim: int, meta_dim: int):
        super().__init__()
        self.gamma = nn.Linear(max(1, meta_dim), feat_dim)
        self.beta  = nn.Linear(max(1, meta_dim), feat_dim)

    def forward(self, feats: torch.Tensor, meta_vec: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma(meta_vec)
        beta  = self.beta(meta_vec)
        return feats * (1 + gamma) + beta


class EffB7HyperNetWrapper(nn.Module):
    def __init__(self, base_model: nn.Module, meta_dim: int, num_classes: int):
        super().__init__()
        self.base = base_model
        try:
            in_f = self.base.classifier[1].in_features
        except Exception:
            in_f = getattr(self.base, "num_features", 2560)
        self.film = FiLMBlock(in_f, meta_dim)
        self.classifier = nn.Linear(in_f, num_classes)

    def forward(self, x: torch.Tensor, metadata: Optional[Dict[str, str]] = None) -> torch.Tensor:
        feats = self.base.forward_features(x)  # [N,C,H,W]
        pooled = self.base.avgpool(feats)     # [N,C,1,1]
        pooled = torch.flatten(pooled, 1)     # [N,C]
        meta_vec = _prepare_metadata_vector(metadata)
        if meta_vec.shape[0] != pooled.shape[0]:
            meta_vec = meta_vec.expand(pooled.shape[0], -1)
        adapted = self.film(pooled, meta_vec)
        return self.classifier(adapted)

    @torch.inference_mode()
    def predict(self, x, metadata=None):
        return self.forward(x, metadata)


class ResNet50HyperNetWrapper(nn.Module):
    def __init__(self, base_model: nn.Module, meta_dim: int, num_classes: int):
        super().__init__()
        self.base = base_model
        in_f = self.base.fc.in_features
        self.base.fc = nn.Identity()  # expose pooled features
        self.film = FiLMBlock(in_f, meta_dim)
        self.classifier = nn.Linear(in_f, num_classes)

    def forward(self, x: torch.Tensor, metadata: Optional[Dict[str, str]] = None) -> torch.Tensor:
        feats = self.base(x)  # [N,C]
        meta_vec = _prepare_metadata_vector(metadata)
        if meta_vec.shape[0] != feats.shape[0]:
            meta_vec = meta_vec.expand(feats.shape[0], -1)
        adapted = self.film(feats, meta_vec)
        return self.classifier(adapted)

    @torch.inference_mode()
    def predict(self, x, metadata=None):
        return self.forward(x, metadata)


class MobileNetV3HyperNetWrapper(nn.Module):
    def __init__(self, base_model: nn.Module, meta_dim: int, num_classes: int):
        super().__init__()
        self.base = base_model
        in_f = self.base.classifier[3].in_features
        self.feature_head = nn.Sequential(*list(self.base.classifier.children())[:-1])
        self.base.classifier = nn.Identity()
        self.film = FiLMBlock(in_f, meta_dim)
        self.classifier = nn.Linear(in_f, num_classes)

    def forward(self, x: torch.Tensor, metadata: Optional[Dict[str, str]] = None) -> torch.Tensor:
        feats = self.base.features(x)
        feats = self.base.avgpool(feats)
        feats = torch.flatten(feats, 1)
        feats = self.feature_head(feats)  # [N, in_f]
        meta_vec = _prepare_metadata_vector(metadata)
        if meta_vec.shape[0] != feats.shape[0]:
            meta_vec = meta_vec.expand(feats.shape[0], -1)
        adapted = self.film(feats, meta_vec)
        return self.classifier(adapted)

    @torch.inference_mode()
    def predict(self, x, metadata=None):
        return self.forward(x, metadata)

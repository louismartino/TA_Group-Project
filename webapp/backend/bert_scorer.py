"""
Joint DistilBERT + structured features scorer.
Matches Filipe's model architecture from joint_fine_tuning.ipynb.
"""
import os
import pickle
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler
from safetensors.torch import load_file


class DistilBERTWithStructuredFeatures(nn.Module):
    """Exact replication of Filipe's joint model."""
    def __init__(self, model_name: str, structured_dim: int,
                 num_labels: int = 2, dropout: float = 0.2, hidden_dim: int = 256):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        text_hidden_size = self.text_encoder.config.hidden_size  # 768
        self.classifier = nn.Sequential(
            nn.Linear(text_hidden_size + structured_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )
        self.loss_fn = nn.CrossEntropyLoss()

    @staticmethod
    def mean_pooling(last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        masked = last_hidden_state * mask
        summed = masked.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def forward(self, input_ids=None, attention_mask=None,
                structured_features=None, labels=None):
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.mean_pooling(outputs.last_hidden_state, attention_mask)
        pooled = self.dropout(pooled)
        combined = torch.cat([pooled, structured_features], dim=1)
        logits = self.classifier(combined)
        loss = self.loss_fn(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}


class BertScorer:
    """Wraps the joint BERT model for easy inference."""

    def __init__(self, checkpoint_dir: str, train_csv_path: str,
                 features_scale: list, features_no_scale: list,
                 text_model_name: str = "distilbert-base-uncased",
                 max_length: int = 128, device: Optional[str] = None):
        self.checkpoint_dir = checkpoint_dir
        self.features_scale = features_scale
        self.features_no_scale = features_no_scale
        self.max_length = max_length
        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Fit scaler on training data (deterministic — same as Filipe's training run)
        df_train = pd.read_csv(train_csv_path)
        self.scaler = StandardScaler()
        self.scaler.fit(df_train[features_scale].astype(np.float32).values)

        # Build model
        structured_dim = len(features_scale) + len(features_no_scale)
        self.model = DistilBERTWithStructuredFeatures(
            model_name=text_model_name,
            structured_dim=structured_dim,
        )

        # Load weights
        weights_path = os.path.join(checkpoint_dir, "model.safetensors")
        if os.path.isfile(weights_path):
            state = load_file(weights_path)
        else:
            bin_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
            state = torch.load(bin_path, map_location="cpu", weights_only=True)

        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device)
        self.model.eval()

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)

    def _build_structured_vector(self, category: str, country: str,
                                  goal_usd: float, duration: int,
                                  blurb_length: int, sentiment_score: float,
                                  readability_score: float, name_blurb_similarity: float,
                                  cci_index: float = 100.0) -> np.ndarray:
        """Build the 29-D structured feature vector in the exact order training used:
           [21 no-scale] + [8 scale]"""
        feat = {}

        # Scaled features (8)
        feat["duration"] = float(duration)
        feat["CCI_index"] = float(cci_index)
        feat["blurb_length"] = float(blurb_length)
        feat["sentiment_score"] = float(sentiment_score)
        feat["readability_score"] = float(readability_score)
        feat["name_blurb_similarity"] = float(name_blurb_similarity)
        feat["log_goal"] = float(np.log1p(goal_usd))
        feat["CCI_per_goal"] = float(cci_index / max(goal_usd, 1))

        # No-scale: category dummies, country dummies, z-score_log_goal
        for col in self.features_no_scale:
            if col.startswith("cat_"):
                feat[col] = 1.0 if col == f"cat_{category}" else 0.0
            elif col.startswith("country_"):
                feat[col] = 1.0 if col == f"country_{country}" else 0.0
            elif col == "z-score_log_goal":
                feat[col] = 0.0  # unknown at inference — set to 0
            else:
                feat[col] = 0.0

        # Build in correct order: no-scale first, then scaled (matches Filipe's training)
        no_scale_arr = np.array([feat[c] for c in self.features_no_scale], dtype=np.float32)
        scale_raw = np.array([feat[c] for c in self.features_scale], dtype=np.float32).reshape(1, -1)
        scale_scaled = self.scaler.transform(scale_raw).flatten()

        combined = np.concatenate([no_scale_arr, scale_scaled]).astype(np.float32)
        combined = np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)
        return combined  # shape: (29,)

    def score(self, name: str, blurb: str, category: str, country: str,
              goal_usd: float, duration: int, blurb_length: int,
              sentiment_score: float, readability_score: float,
              name_blurb_similarity: float, cci_index: float = 100.0) -> float:
        """Returns predicted success probability (0-1)."""
        text = f"{name} {blurb}"
        text_inputs = self.tokenizer(
            text, truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt",
        )
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}

        struct = self._build_structured_vector(
            category, country, goal_usd, duration,
            blurb_length, sentiment_score, readability_score,
            name_blurb_similarity, cci_index,
        )
        struct_tensor = torch.from_numpy(struct).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model(
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"],
                structured_features=struct_tensor,
            )
            probs = torch.softmax(out["logits"], dim=-1)
            return float(probs[0, 1].item())

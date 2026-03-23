"""
ensemble.py
===========
Stacking ensemble implementation (§3 — Model Ensemble Pipeline, Figure 1).

Stage 1 base models:
  - LSTM
  - GRU
  - LightGBM
  - Linear Regression (Ridge)
  - DNN

Stage 2 meta-learner:
  - Ridge regression trained on out-of-fold base model predictions

Matches the LSTM + GRU + LR + LightGBM configuration from Table 1.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from typing import Optional
from sklearn.linear_model import Ridge

from models import (
    build_lstm_model, build_gru_model, build_dnn_model,
    train_lightgbm, train_linear_regression,
    evaluate_predictions, combined_loss, sharpe_metric,
)

# ─── Callbacks ────────────────────────────────────────────────────────────────

def get_callbacks(model_name: str, patience: int = 15):
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience,
            restore_best_weights=True, verbose=0
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=7,
            min_lr=1e-6, verbose=0
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f"./checkpoints/{model_name}_best.keras",
            monitor="val_loss", save_best_only=True, verbose=0
        ),
    ]


# ─── Training helpers ─────────────────────────────────────────────────────────

def train_keras_model(model, X_train, y_train, X_val, y_val,
                      epochs=100, batch_size=256, model_name="model"):
    os.makedirs("./checkpoints", exist_ok=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=get_callbacks(model_name),
        verbose=0,
    )
    print(f"  {model_name}: trained {len(history.history['loss'])} epochs, "
          f"best val_loss={min(history.history['val_loss']):.4f}")
    return model, history


# ─── Ensemble ─────────────────────────────────────────────────────────────────

class StockEnsemble:
    """
    Full stacking ensemble pipeline from Sui et al. (2024).

    Usage
    -----
    ensemble = StockEnsemble()
    ensemble.fit(data)
    results  = ensemble.evaluate(data)
    preds    = ensemble.predict(X_seq, X_flat)
    """

    def __init__(self, seq_len: int = 20, epochs: int = 80, batch_size: int = 256):
        self.seq_len    = seq_len
        self.epochs     = epochs
        self.batch_size = batch_size
        self.models_    = {}
        self.meta_      = None
        self.history_   = {}

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(self, data: dict):
        """
        Train all base models, then fit meta-learner on validation predictions.

        Parameters
        ----------
        data : dict returned by data_pipeline.build_dataset()
        """
        seq_len    = self.seq_len
        n_features = data["X_train_seq"].shape[2]
        n_flat     = data["X_train_flat"].shape[1]

        print("\n" + "═"*60)
        print("  STAGE 1 — Training base models")
        print("═"*60)

        # ── LSTM ──────────────────────────────────────────────────────────────
        print("\n[1/5] LSTM …")
        lstm = build_lstm_model(seq_len, n_features)
        lstm, h = train_keras_model(
            lstm,
            data["X_train_seq"], data["y_train_seq"],
            data["X_val_seq"],   data["y_val_seq"],
            epochs=self.epochs, batch_size=self.batch_size, model_name="LSTM"
        )
        self.models_["lstm"] = lstm
        self.history_["lstm"] = h.history

        # ── GRU ───────────────────────────────────────────────────────────────
        print("[2/5] GRU …")
        gru = build_gru_model(seq_len, n_features)
        gru, h = train_keras_model(
            gru,
            data["X_train_seq"], data["y_train_seq"],
            data["X_val_seq"],   data["y_val_seq"],
            epochs=self.epochs, batch_size=self.batch_size, model_name="GRU"
        )
        self.models_["gru"] = gru
        self.history_["gru"] = h.history

        # ── DNN ───────────────────────────────────────────────────────────────
        print("[3/5] DNN …")
        dnn = build_dnn_model(n_flat)
        dnn, h = train_keras_model(
            dnn,
            data["X_train_flat"], data["y_train"],
            data["X_val_flat"],   data["y_val"],
            epochs=self.epochs, batch_size=self.batch_size, model_name="DNN"
        )
        self.models_["dnn"] = dnn
        self.history_["dnn"] = h.history

        # ── LightGBM ──────────────────────────────────────────────────────────
        print("[4/5] LightGBM …")
        lgbm = train_lightgbm(
            data["X_train_flat"], data["y_train"],
            data["X_val_flat"],   data["y_val"],
        )
        self.models_["lgbm"] = lgbm

        # ── Linear Regression ─────────────────────────────────────────────────
        print("[5/5] Linear Regression …")
        lr = train_linear_regression(data["X_train_flat"], data["y_train"])
        self.models_["lr"] = lr

        print("\n" + "═"*60)
        print("  STAGE 2 — Fitting meta-learner on val predictions")
        print("═"*60)
        val_meta = self._stack_predictions(
            data["X_val_seq"], data["X_val_flat"], n_features
        )
        self.meta_ = Ridge(alpha=0.1)
        self.meta_.fit(val_meta, data["y_val_seq"])
        print(f"  Meta-learner weights: {self.meta_.coef_}")
        return self

    # ── predict ───────────────────────────────────────────────────────────────

    def _stack_predictions(self, X_seq, X_flat, n_features=None):
        """Collect base-model predictions → (N, 5) stacking matrix."""
        # Align seq and flat lengths (seq is shorter by seq_len)
        seq_len = self.seq_len
        n_seq = len(X_seq)

        # Flat arrays need to be trimmed to match seq output length
        X_flat_aligned = X_flat[-n_seq:] if len(X_flat) >= n_seq else X_flat

        p_lstm = self.models_["lstm"].predict(X_seq, verbose=0).flatten()
        p_gru  = self.models_["gru"].predict(X_seq,  verbose=0).flatten()
        p_dnn  = self.models_["dnn"].predict(X_flat_aligned, verbose=0).flatten()
        p_lgbm = self.models_["lgbm"].predict(X_flat_aligned)
        p_lr   = self.models_["lr"].predict(X_flat_aligned)

        # Trim all to same length
        min_len = min(len(p_lstm), len(p_gru), len(p_dnn), len(p_lgbm), len(p_lr))
        return np.column_stack([
            p_lstm[:min_len], p_gru[:min_len], p_dnn[:min_len],
            p_lgbm[:min_len], p_lr[:min_len]
        ])

    def predict(self, X_seq, X_flat) -> np.ndarray:
        """Ensemble prediction via trained meta-learner."""
        meta_input = self._stack_predictions(X_seq, X_flat)
        return self.meta_.predict(meta_input)

    # ── evaluate ──────────────────────────────────────────────────────────────

    def evaluate(self, data: dict) -> pd.DataFrame:
        """Compute metrics for every base model + ensemble on test set."""
        print("\n" + "═"*60)
        print("  TEST SET EVALUATION (replicating Table 1)")
        print("═"*60)
        results = []

        X_seq  = data["X_test_seq"]
        X_flat = data["X_test_flat"]
        y_seq  = data["y_test_seq"]

        n_seq = len(X_seq)
        X_flat_a = X_flat[-n_seq:]
        y_flat_a = data["y_test"][-n_seq:]

        # Individual models
        p_lstm = self.models_["lstm"].predict(X_seq,  verbose=0).flatten()
        p_gru  = self.models_["gru"].predict(X_seq,   verbose=0).flatten()
        p_dnn  = self.models_["dnn"].predict(X_flat_a, verbose=0).flatten()
        p_lgbm = self.models_["lgbm"].predict(X_flat_a)
        p_lr   = self.models_["lr"].predict(X_flat_a)

        for label, pred in [
            ("LightGBM",          p_lgbm),
            ("LSTM",              p_lstm),
            ("GRU",               p_gru),
            ("DNN",               p_dnn),
            ("Linear Regression", p_lr),
        ]:
            results.append(evaluate_predictions(y_flat_a, pred, label))

        # Simple average ensemble
        avg = (p_lstm + p_gru + p_lgbm + p_lr) / 4
        results.append(evaluate_predictions(y_seq, avg[:len(y_seq)],
                                            "Simple Avg (LSTM+GRU+LGBM+LR)"))

        # Meta-learner stacking
        p_ens = self.predict(X_seq, X_flat)
        min_len = min(len(p_ens), len(y_seq))
        results.append(evaluate_predictions(y_seq[:min_len], p_ens[:min_len],
                                            "Stacking Ensemble (all 5)"))

        df = pd.DataFrame(results).set_index("label")
        print("\n" + df.to_string())
        return df

    # ── save / load ───────────────────────────────────────────────────────────

    def save(self, directory: str = "./saved_model"):
        os.makedirs(directory, exist_ok=True)
        self.models_["lstm"].save(os.path.join(directory, "lstm.keras"))
        self.models_["gru"].save(os.path.join(directory,  "gru.keras"))
        self.models_["dnn"].save(os.path.join(directory,  "dnn.keras"))
        with open(os.path.join(directory, "lgbm.pkl"), "wb") as f:
            pickle.dump(self.models_["lgbm"], f)
        with open(os.path.join(directory, "lr.pkl"), "wb") as f:
            pickle.dump(self.models_["lr"], f)
        with open(os.path.join(directory, "meta.pkl"), "wb") as f:
            pickle.dump(self.meta_, f)
        # Save training history
        with open(os.path.join(directory, "history.json"), "w") as f:
            safe = {k: {kk: [float(v) for v in vv]
                        for kk, vv in h.items()}
                    for k, h in self.history_.items()}
            json.dump(safe, f)
        print(f"  Model saved to {directory}/")

    @classmethod
    def load(cls, directory: str = "./saved_model"):
        from tensorflow.keras.models import load_model
        obj = cls.__new__(cls)
        obj.models_ = {}
        obj.models_["lstm"] = load_model(
            os.path.join(directory, "lstm.keras"),
            custom_objects={"combined_loss": combined_loss(0.5),
                            "sharpe_metric": sharpe_metric}
        )
        obj.models_["gru"] = load_model(
            os.path.join(directory, "gru.keras"),
            custom_objects={"combined_loss": combined_loss(0.5),
                            "sharpe_metric": sharpe_metric}
        )
        obj.models_["dnn"] = load_model(
            os.path.join(directory, "dnn.keras"),
            custom_objects={"combined_loss": combined_loss(0.5)}
        )
        with open(os.path.join(directory, "lgbm.pkl"), "rb") as f:
            obj.models_["lgbm"] = pickle.load(f)
        with open(os.path.join(directory, "lr.pkl"), "rb") as f:
            obj.models_["lr"] = pickle.load(f)
        with open(os.path.join(directory, "meta.pkl"), "rb") as f:
            obj.meta_ = pickle.load(f)
        return obj

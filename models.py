"""
models.py
=========
All model components from Sui et al. (2024):
  - LSTM branch  (§3.1)
  - GRU branch   (§3.2)
  - Logistic Regression baseline (§3.3)
  - LightGBM     (§3.4)
  - Ensemble meta-learner (stacking §3.5)

Loss functions implemented:
  - MSE          (§3.5.1)
  - Correlation  (§3.5.2)
  - Sharpe Ratio (§3.5.3) — used as secondary metric
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# ─── Custom Loss Functions (§3.5) ─────────────────────────────────────────────

def correlation_loss(y_true, y_pred):
    """1 - Pearson correlation coefficient (§3.5.2). Minimise → maximise correlation."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    mu_t = tf.reduce_mean(y_true)
    mu_p = tf.reduce_mean(y_pred)
    cov   = tf.reduce_mean((y_true - mu_t) * (y_pred - mu_p))
    std_t = tf.math.reduce_std(y_true) + 1e-8
    std_p = tf.math.reduce_std(y_pred) + 1e-8
    return 1.0 - cov / (std_t * std_p)


def combined_loss(alpha=0.5):
    """Weighted combination of MSE + Correlation loss."""
    def loss(y_true, y_pred):
        mse  = tf.reduce_mean(tf.square(y_true - y_pred))
        corr = correlation_loss(y_true, y_pred)
        return alpha * mse + (1 - alpha) * corr
    loss.__name__ = "combined_loss"
    return loss


def sharpe_metric(y_true, y_pred):
    """Approximate Sharpe ratio as Keras metric (§3.5.3). Higher = better."""
    ret = y_pred
    mean_r = tf.reduce_mean(ret)
    std_r  = tf.math.reduce_std(ret) + 1e-8
    return mean_r / std_r


# ─── LSTM Model (§3.1) ────────────────────────────────────────────────────────

def build_lstm_model(seq_len: int, n_features: int,
                     lstm_units=(128, 64), dropout=0.3,
                     dense_units=(64, 32)) -> keras.Model:
    """
    Architecture (§3.1):
      BatchNorm → Dense → Dropout → Reshape → LSTM × 2 → Dense → Output
    """
    inp = keras.Input(shape=(seq_len, n_features), name="lstm_input")

    x = layers.BatchNormalization()(inp)

    # Dense projection before recurrent layers
    for units in dense_units:
        x = layers.TimeDistributed(layers.Dense(units, activation="relu",
                kernel_regularizer=regularizers.l2(1e-4)))(x)
        x = layers.Dropout(dropout)(x)

    # LSTM stack
    for i, units in enumerate(lstm_units):
        return_seq = (i < len(lstm_units) - 1)
        x = layers.LSTM(units, return_sequences=return_seq,
                        dropout=dropout, recurrent_dropout=0.1)(x)

    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, name="lstm_output")(x)

    model = keras.Model(inp, out, name="LSTM_model")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=combined_loss(alpha=0.5),
        metrics=["mse", sharpe_metric],
    )
    return model


# ─── GRU Model (§3.2) ─────────────────────────────────────────────────────────

def build_gru_model(seq_len: int, n_features: int,
                    gru_units=(128, 64), dropout=0.3,
                    dense_units=(64,)) -> keras.Model:
    """
    Architecture (§3.2):
      Dense → BatchNorm → Reshape → GRU × 2 → Dense → Output
    """
    inp = keras.Input(shape=(seq_len, n_features), name="gru_input")

    x = inp
    for units in dense_units:
        x = layers.TimeDistributed(layers.Dense(units, activation="relu",
                kernel_regularizer=regularizers.l2(1e-4)))(x)

    x = layers.BatchNormalization()(x)

    # GRU stack
    for i, units in enumerate(gru_units):
        return_seq = (i < len(gru_units) - 1)
        x = layers.GRU(units, return_sequences=return_seq,
                       dropout=dropout, recurrent_dropout=0.1)(x)

    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1, name="gru_output")(x)

    model = keras.Model(inp, out, name="GRU_model")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=combined_loss(alpha=0.5),
        metrics=["mse", sharpe_metric],
    )
    return model


# ─── LightGBM Model (§3.4) ────────────────────────────────────────────────────

LGBM_PARAMS = {
    "objective":       "regression",
    "metric":          ["rmse", "mae"],
    "learning_rate":   0.05,
    "num_leaves":      63,
    "max_depth":       -1,
    "min_child_samples": 20,
    "feature_fraction":  0.8,
    "bagging_fraction":  0.8,
    "bagging_freq":      5,
    "reg_alpha":         0.1,
    "reg_lambda":        0.1,
    "n_estimators":    1000,
    "random_state":    42,
    "verbose":         -1,
}


def train_lightgbm(X_train, y_train, X_val, y_val, params=None):
    """Train LightGBM with early stopping."""
    if params is None:
        params = LGBM_PARAMS.copy()

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )
    return model


# ─── Logistic Regression Baseline (§3.3) ──────────────────────────────────────
# Paper uses LR as a directional baseline; here we use Ridge regression for
# continuous return prediction (consistent with regression framing of the ensemble).

def train_linear_regression(X_train, y_train):
    """Ridge regression baseline (§3.3)."""
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    return model


# ─── Keras DNN (mentioned in results Table 1) ─────────────────────────────────

def build_dnn_model(n_features: int, dropout=0.3) -> keras.Model:
    """Standard feed-forward DNN for flat feature input."""
    inp = keras.Input(shape=(n_features,), name="dnn_input")
    x = layers.BatchNormalization()(inp)
    for units in [256, 128, 64, 32]:
        x = layers.Dense(units, activation="relu",
                          kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, name="dnn_output")(x)
    model = keras.Model(inp, out, name="DNN_model")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=combined_loss(0.5),
        metrics=["mse"],
    )
    return model


# ─── Evaluation Metrics (§4.1–4.3) ───────────────────────────────────────────

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, label: str = ""):
    """Compute RMSE, Correlation, MSE as in Table 1."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mse  = mean_squared_error(y_true, y_pred)
    corr = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0.0
    r    = y_pred
    sharpe = (r.mean() / (r.std() + 1e-8)) * np.sqrt(252)  # annualised
    print(f"  [{label:30s}] RMSE={rmse:.4f}  Corr={corr:.4f}  MSE={mse:.6f}  Sharpe={sharpe:.3f}")
    return {"label": label, "rmse": rmse, "corr": corr, "mse": mse, "sharpe": sharpe}

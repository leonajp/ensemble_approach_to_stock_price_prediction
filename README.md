# Ensemble Stock Prediction — Sui et al. (2024)

Implementation of **"An Ensemble Approach to Stock Price Prediction Using Deep Learning and Time Series Models"** (Mujie Sui et al., Preprints 2024).

## Architecture

```
Raw OHLCV Data
      │
      ▼
Data Preprocessing (§3.6)
  ├─ Forward-fill missing values
  ├─ IQR outlier capping
  └─ Feature engineering:
       Return, MA(5/10/20), EMA(5/10/20),
       Volatility, RSI(14), Bollinger Bands

      │
      ├──────────────────────┬────────────────────┐
      ▼                      ▼                    ▼
  Sequences (T×F)        Flat features (F,)   Flat features
  for LSTM / GRU         for DNN / LGBM       for Ridge LR

      │                      │                    │
      ▼                      ▼                    ▼
  LSTM (§3.1)           LightGBM (§3.4)     Linear Reg (§3.3)
  GRU  (§3.2)           DNN
      │                      │                    │
      └──────────────────────┴────────────────────┘
                             │
                      Stacking (meta-learner)
                      Ridge regression on OOF preds
                             │
                      Ensemble prediction
                             │
                      Trading Signals
                      LONG / SHORT / FLAT
```

## Loss Functions (§3.5)

| Loss               | Formula                         |
|--------------------|---------------------------------|
| MSE                | mean((y - ŷ)²)                  |
| Correlation Loss   | 1 − Corr(y, ŷ)                  |
| Combined           | 0.5 × MSE + 0.5 × Corr Loss     |
| Sharpe (metric)    | E[r] / σ[r] × √252              |

## Files

| File                  | Purpose                                      |
|-----------------------|----------------------------------------------|
| `data_pipeline.py`    | Download, preprocess, feature engineering    |
| `models.py`           | LSTM, GRU, DNN, LightGBM, LR definitions    |
| `ensemble.py`         | Stacking orchestration, save/load            |
| `train_and_signal.py` | End-to-end runner + signal generation        |
| `requirements.txt`    | Dependencies                                 |

## Quick Start

```bash
pip install -r requirements.txt

# Quick smoke test (3 tickers, 30 epochs, ~5 min)
python train_and_signal.py --quick

# Full run with paper's model
python train_and_signal.py --tickers AAPL MSFT NVDA JPM GS --epochs 100

# Custom tickers
python train_and_signal.py --tickers 9984.T 7203.T 6758.T --epochs 80
```

## Outputs

| File                      | Contents                                          |
|---------------------------|---------------------------------------------------|
| `evaluation_results.csv`  | RMSE / Correlation / MSE / Sharpe per model       |
| `trading_signals.csv`     | Per-day: predicted return, signal (±1/0), confidence |
| `backtest_results.csv`    | Cumulative PnL, strategy vs buy-and-hold          |
| `saved_model/`            | Serialised model weights                          |
| `data_cache/`             | Parquet cache of raw OHLCV data                   |

## Signal Logic

```python
pred_return > +0.003  →  LONG  (+1)
pred_return < -0.003  →  SHORT (-1)
otherwise             →  FLAT  ( 0)
```

Thresholds are configurable in `generate_signals()`.

## Performance (Paper Table 1 — JPX Dataset)

| Model                        | RMSE  | Correlation |
|------------------------------|-------|-------------|
| LightGBM                     | 0.172 | 0.182       |
| LSTM                         | 0.214 | 0.234       |
| Keras DNN + RNN              | 0.261 | 0.245       |
| Keras DNN + LightGBM         | 0.315 | 0.324       |
| **LSTM + GRU + LR + LightGBM** | **0.351** | **0.362** |

## Extending

To plug in your existing ClickHouse/Bloomberg data pipeline, replace `build_dataset()` 
in `data_pipeline.py` with a function that returns the same dict schema:
```python
{
    "X_train_seq": np.ndarray,  # (N, seq_len, n_features)
    "y_train_seq": np.ndarray,  # (N,)
    "X_val_seq":   ...,
    "y_val_seq":   ...,
    "X_test_seq":  ...,
    "y_test_seq":  ...,
    "X_train_flat": np.ndarray, # (N, n_features)
    "y_train":      np.ndarray,
    "X_val_flat":   ...,
    "y_val":        ...,
    "X_test_flat":  ...,
    "y_test":       ...,
    "feature_names": list[str],
    "scaler":        StandardScaler,
    "raw_df":        pd.DataFrame,
}
```

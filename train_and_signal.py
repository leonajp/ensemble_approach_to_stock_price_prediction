"""
train_and_signal.py
===================
End-to-end runner:
  1. Download & preprocess data
  2. Train the full ensemble
  3. Evaluate (replicates Table 1)
  4. Generate trading signals on a rolling basis
  5. Save signals to CSV for downstream use

Run:
    python train_and_signal.py [--tickers AAPL MSFT NVDA] [--quick]

Flags:
    --quick   : 3 tickers, 30 epochs — fast smoke test (~5 min)
    --tickers : space-separated list of Yahoo Finance ticker symbols
"""

import argparse
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Sui et al. (2024) ensemble runner")
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="List of Yahoo Finance tickers (default: 15-ticker set)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick run: 3 tickers, 30 epochs")
    parser.add_argument("--epochs",   type=int, default=80)
    parser.add_argument("--seq-len",  type=int, default=20,
                        help="Lookback window length for LSTM/GRU")
    parser.add_argument("--no-save",  action="store_true",
                        help="Skip saving model checkpoints")
    return parser.parse_args()


def main():
    args = parse_args()

    # Lazy imports (so argparse --help works without heavy deps)
    from data_pipeline import build_dataset, DEFAULT_TICKERS, SEQUENCE_LEN
    from ensemble import StockEnsemble

    tickers = args.tickers
    epochs  = args.epochs
    seq_len = args.seq_len

    if args.quick:
        tickers = tickers or ["AAPL", "MSFT", "NVDA"]
        epochs  = 30
        print("⚡ Quick mode: 3 tickers, 30 epochs")
    else:
        tickers = tickers or DEFAULT_TICKERS

    print(f"\n{'='*60}")
    print(f"  Sui et al. (2024) — Ensemble Stock Prediction Pipeline")
    print(f"  Tickers : {tickers}")
    print(f"  Epochs  : {epochs}  |  Seq len: {seq_len}")
    print(f"{'='*60}\n")

    # ── Step 1: Data ─────────────────────────────────────────────────────────
    data = build_dataset(tickers=tickers)

    # ── Step 2: Train ────────────────────────────────────────────────────────
    ensemble = StockEnsemble(seq_len=seq_len, epochs=epochs)
    ensemble.fit(data)

    # ── Step 3: Evaluate ─────────────────────────────────────────────────────
    results_df = ensemble.evaluate(data)
    results_df.to_csv("./evaluation_results.csv")
    print("\nEvaluation results saved → evaluation_results.csv")

    # ── Step 4: Save model ───────────────────────────────────────────────────
    if not args.no_save:
        ensemble.save("./saved_model")

    # ── Step 5: Generate trading signals ─────────────────────────────────────
    print("\n" + "═"*60)
    print("  SIGNAL GENERATION")
    print("═"*60)
    signals = generate_signals(ensemble, data)
    signals.to_csv("./trading_signals.csv")
    print(f"\n  Signals saved → trading_signals.csv ({len(signals)} rows)")
    print(signals.tail(10).to_string())

    return ensemble, data, results_df, signals


# ─── Signal Generation ────────────────────────────────────────────────────────

def generate_signals(ensemble, data: dict,
                     long_threshold: float  =  0.003,
                     short_threshold: float = -0.003) -> pd.DataFrame:
    """
    Convert predicted next-day returns into discrete trading signals.

    Signal logic:
        pred_return >  long_threshold  → LONG  (+1)
        pred_return < short_threshold  → SHORT (-1)
        otherwise                      → FLAT  ( 0)

    Returns
    -------
    DataFrame with columns: [predicted_return, signal, confidence]
    """
    preds = ensemble.predict(data["X_test_seq"], data["X_test_flat"])
    y_true = data["y_test_seq"]
    n = min(len(preds), len(y_true))

    signals = np.where(preds[:n] > long_threshold,   1,
              np.where(preds[:n] < short_threshold,  -1, 0))

    # Confidence score: distance from 0 threshold, normalised
    confidence = np.abs(preds[:n]) / (np.abs(preds[:n]).max() + 1e-8)

    # Retrieve dates from the raw dataframe (test window)
    raw_df = data["raw_df"]
    test_dates = raw_df.index[-(n):]

    df = pd.DataFrame({
        "date":             test_dates[:n],
        "predicted_return": preds[:n],
        "actual_return":    y_true[:n],
        "signal":           signals,
        "confidence":       confidence,
        "correct_direction": (np.sign(preds[:n]) == np.sign(y_true[:n])).astype(int),
    })
    df = df.set_index("date")

    # Summary stats
    direction_acc = df["correct_direction"].mean()
    n_long  = (df["signal"] ==  1).sum()
    n_short = (df["signal"] == -1).sum()
    n_flat  = (df["signal"] ==  0).sum()
    print(f"\n  Signal summary:")
    print(f"    LONG={n_long} | SHORT={n_short} | FLAT={n_flat}")
    print(f"    Directional accuracy: {direction_acc:.1%}")

    return df


# ─── Backtesting helper (simple long/short PnL) ───────────────────────────────

def backtest_signals(signals: pd.DataFrame,
                     transaction_cost: float = 0.001) -> pd.DataFrame:
    """
    Simple signal-based backtest.  Returns daily strategy returns.

    position(t) = signal(t-1)   [signal generated EoD, traded next open]
    strategy_return = position × actual_return − |Δposition| × cost
    """
    sig = signals["signal"].shift(1).fillna(0)
    ret = signals["actual_return"]
    tc  = transaction_cost * (sig - sig.shift(1)).abs().fillna(0)

    strat = sig * ret - tc
    bh    = ret  # buy-and-hold benchmark (average across portfolio)

    cum_strat = (1 + strat).cumprod()
    cum_bh    = (1 + bh).cumprod()

    annual_ret  = strat.mean() * 252
    annual_vol  = strat.std()  * np.sqrt(252)
    sharpe      = annual_ret / (annual_vol + 1e-8)
    max_dd      = (cum_strat / cum_strat.cummax() - 1).min()

    print("\n  Backtest results:")
    print(f"    Annualised return : {annual_ret:.2%}")
    print(f"    Annualised vol    : {annual_vol:.2%}")
    print(f"    Sharpe ratio      : {sharpe:.3f}")
    print(f"    Max drawdown      : {max_dd:.2%}")
    print(f"    Final cum. return : {cum_strat.iloc[-1] - 1:.2%}")
    print(f"    B&H cum. return   : {cum_bh.iloc[-1] - 1:.2%}")

    result = signals.copy()
    result["strategy_return"]    = strat
    result["cumulative_strategy"] = cum_strat
    result["cumulative_bh"]       = cum_bh
    return result


if __name__ == "__main__":
    ensemble, data, results, signals = main()

    # Optional: run backtest
    print("\n" + "═"*60)
    print("  SIMPLE BACKTEST")
    print("═"*60)
    backtest = backtest_signals(signals)
    backtest.to_csv("./backtest_results.csv")
    print("  Backtest results saved → backtest_results.csv")

"""Example script for running Kronos predictions with A-share data from AkShare.

This script downloads historical OHLCV data for a single A-share using AkShare,
converts it into the format expected by :class:`KronosPredictor`, and then
generates a forecast window tailored to the requested future dates. The
predicted OHLCVA series is saved as a tab-separated ``.txt`` file alongside the
script.

Requirements
------------
The script depends on ``akshare`` in addition to the core Kronos
requirements.  Install it via ``pip install akshare`` before running the
example.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Iterable

import pandas as pd
import torch
import akshare as ak

sys.path.append(str(Path(__file__).resolve().parent.parent))
from model import Kronos, KronosTokenizer, KronosPredictor


def fetch_a_share_daily(symbol: str) -> pd.DataFrame:
    """Fetch daily OHLCV data for a given A-share symbol via AkShare.

    Parameters
    ----------
    symbol:
        The 6-digit stock code recognised by AkShare, e.g. ``"600519"``.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the ``open/high/low/close/volume/amount`` columns
        and a ``timestamps`` column in ascending chronological order.
    """

    try:
        raw_df = ak.stock_zh_a_hist(symbol=symbol, period="daily", adjust="")
    except Exception as exc:  # pragma: no cover - network/proxy failures
        raise RuntimeError(
            "Failed to download data from AkShare. Check your internet access "
            "or proxy settings and retry."
        ) from exc

    rename_map = {
        "日期": "timestamps",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
    }
    df = raw_df.rename(columns=rename_map)

    # Ensure the selected columns exist and are numeric.
    df = df[list(rename_map.values())].copy()
    df["timestamps"] = pd.to_datetime(df["timestamps"], utc=False)
    numeric_cols = ["open", "high", "low", "close", "volume", "amount"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # Drop rows with missing values and sort chronologically.
    df = df.dropna(subset=numeric_cols).sort_values("timestamps").reset_index(drop=True)

    return df


def main() -> None:
    symbol = "600519"  # Example: Kweichow Moutai
    lookback = 400
    forecast_start = pd.Timestamp("2025-07-01")
    forecast_end = pd.Timestamp("2025-07-31")

    print(f"Downloading daily data for {symbol} via AkShare...")
    try:
        df = fetch_a_share_daily(symbol)
    except RuntimeError as exc:
        print(exc)
        return

    # Limit the historical window to the trading days strictly before the forecast window.
    history_df = df[df["timestamps"] < forecast_start].copy()
    if history_df.shape[0] < lookback:
        raise ValueError(
            "Not enough historical data points fetched before the forecast start. "
            "Please lower `lookback` or ensure the AkShare dataset is complete."
        )

    # Build the forecast timestamp sequence (business days between start/end).
    y_timestamp_index = pd.bdate_range(forecast_start, forecast_end, tz=None)
    y_timestamp: Iterable[pd.Timestamp] = pd.Series(y_timestamp_index)
    pred_len = len(y_timestamp)
    if pred_len == 0:
        raise ValueError("No business days found between forecast_start and forecast_end.")

    df_window = history_df.iloc[-lookback:].reset_index(drop=True)

    x_df = df_window.loc[:, ["open", "high", "low", "close", "volume", "amount"]]
    x_timestamp = df_window.loc[:, "timestamps"].reset_index(drop=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Loading Kronos models on device: {device}")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)

    print("Running prediction...")
    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=pred_len,
        T=1.0,
        top_p=0.9,
        sample_count=1,
        verbose=True,
    )

    print("Forecasted data (head):")
    print(pred_df.head())

    # Persist the results to a human-readable text file.
    output_dir = Path(__file__).resolve().parent
    output_path = output_dir / f"prediction_{symbol}_{forecast_start:%Y%m}_daily.txt"
    pred_df.to_csv(
        output_path,
        sep="\t",
        float_format="%.4f",
        date_format="%Y-%m-%d",
        header=True,
    )
    print(f"Saved prediction results to: {output_path}")

    print("Skipping plot: future ground truth is not yet available for visual comparison.")


if __name__ == "__main__":
    main()


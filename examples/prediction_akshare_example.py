"""Simple AkShare + Kronos batch prediction script.

This script keeps the original CLI usage (``python examples/prediction_akshare_example.py``)
while staying friendly to Jupyter notebooks.  Place one or multiple JSON files inside
the ``examples`` directory (or supply a custom path) and each record containing
``time``, ``title`` and ``code_name`` will be enriched with three days of predicted
prices from the Kronos-small model.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import time as dt_time
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List

import akshare as ak
import pandas as pd
from pandas.tseries.offsets import BusinessDay
import torch

try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    PROJECT_ROOT = Path.cwd().resolve()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from model import Kronos, KronosPredictor, KronosTokenizer

LOOKBACK_DAYS = 300
PREDICTION_DAYS = 3
REQUIRED_FIELDS = {"time", "title", "code_name"}
DEFAULT_OUTPUT = Path("akshare_predictions.json")
DEFAULT_MODEL = "NeoQuasar/Kronos-small"
DEFAULT_TOKENIZER = "NeoQuasar/Kronos-Tokenizer-base"


@dataclass
class PredictionResult:
    record: Dict[str, Any]
    prices: List[Dict[str, Any]]
    error: str | None = None


def load_json_records(directory: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as fp:
                payload = json.load(fp)
        except json.JSONDecodeError as exc:
            print(f"Skipping {path.name}: invalid JSON ({exc}).")
            continue

        items: Iterable[Any]
        if isinstance(payload, list):
            items = payload
        elif isinstance(payload, dict):
            items = payload.get("data", payload)
            if isinstance(items, dict):
                items = [items]
        else:
            print(f"Skipping {path.name}: unsupported JSON structure {type(payload)!r}.")
            continue

        for item in items:
            if isinstance(item, dict):
                records.append(item)
            else:
                print(f"Skipping entry in {path.name}: expected object, got {type(item)!r}.")
    return records


def fetch_a_share_daily(symbol: str) -> pd.DataFrame:
    try:
        raw_df = ak.stock_zh_a_hist(symbol=symbol, period="daily", adjust="")
    except Exception as exc:  # pragma: no cover - network failures
        raise RuntimeError(
            "Failed to download data from AkShare. Check your internet connectivity "
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
    df = df[list(rename_map.values())].copy()
    df["timestamps"] = pd.to_datetime(df["timestamps"], utc=False)

    numeric_cols = ["open", "high", "low", "close", "volume", "amount"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=numeric_cols).sort_values("timestamps")
    return df.reset_index(drop=True)


def normalise_symbol(code_name: str) -> str:
    digits = "".join(ch for ch in str(code_name) if ch.isdigit())
    return digits if len(digits) == 6 else str(code_name)


def parse_event_time(raw_time: Any) -> pd.Timestamp:
    ts = pd.to_datetime(raw_time, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid time value: {raw_time!r}")
    if getattr(ts, "tzinfo", None) is not None:
        ts = ts.tz_convert(None)
    try:
        ts = ts.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    return ts


def determine_forecast_start(event_ts: pd.Timestamp) -> pd.Timestamp:
    cutoff = dt_time(15, 0)
    start_ts = event_ts.normalize()
    if event_ts.time() >= cutoff:
        start_ts += BusinessDay(1)
    return start_ts


def ensure_series(values: Iterable[pd.Timestamp]) -> pd.Series:
    series = pd.to_datetime(list(values), errors="coerce")
    if getattr(series, "tz", None) is not None:
        series = series.tz_convert(None)
    try:
        series = series.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    if pd.isna(series).any():
        raise ValueError("Timestamp sequence contains invalid entries.")
    return pd.Series(series).reset_index(drop=True)


def load_predictor(device: str | None = None) -> KronosPredictor:
    tokenizer = KronosTokenizer.from_pretrained(DEFAULT_TOKENIZER)
    model = Kronos.from_pretrained(DEFAULT_MODEL)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = KronosPredictor(tokenizer=tokenizer, model=model, device=device)
    return predictor


def predict_record(record: Dict[str, Any], predictor: KronosPredictor) -> PredictionResult:
    missing = REQUIRED_FIELDS - record.keys()
    if missing:
        return PredictionResult(record=record, prices=[], error=f"missing fields: {sorted(missing)}")

    try:
        event_ts = parse_event_time(record["time"])
    except ValueError as exc:
        return PredictionResult(record=record, prices=[], error=str(exc))

    symbol = normalise_symbol(record["code_name"])

    try:
        history = fetch_a_share_daily(symbol)
    except RuntimeError as exc:
        return PredictionResult(record=record, prices=[], error=str(exc))

    start_ts = determine_forecast_start(event_ts)
    history = history[history["timestamps"] < start_ts]
    if len(history) < LOOKBACK_DAYS:
        return PredictionResult(
            record=record,
            prices=[],
            error=f"Not enough historical data before {start_ts.date()} (need {LOOKBACK_DAYS} days).",
        )

    lookback = history.tail(LOOKBACK_DAYS).set_index("timestamps")
    lookback = lookback[["open", "high", "low", "close", "volume", "amount"]]

    forecast_dates = pd.bdate_range(start=start_ts, periods=PREDICTION_DAYS)
    x_timestamp = ensure_series(lookback.index)
    y_timestamp = ensure_series(forecast_dates)

    pred_df = predictor.predict(
        history=lookback,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        sample_count=1,
    )

    pred_df = pred_df.reset_index().rename(columns={"index": "timestamp"})
    prices = []
    for _, row in pred_df.iterrows():
        prices.append(
            {
                "timestamp": row["timestamp"].strftime("%Y-%m-%d"),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row.get("volume", 0.0)),
                "amount": float(row.get("amount", 0.0)),
            }
        )

    enriched = dict(record)
    enriched["predicted_prices"] = prices
    return PredictionResult(record=enriched, prices=prices)


def run_prediction(records: List[Dict[str, Any]], predictor: KronosPredictor) -> List[Dict[str, Any]]:
    enriched_records: List[Dict[str, Any]] = []
    for record in records:
        result = predict_record(record, predictor)
        if result.error:
            combined = dict(record)
            combined["error"] = result.error
            enriched_records.append(combined)
            print(f"Failed to predict {record.get('code_name')}: {result.error}")
        else:
            enriched_records.append(result.record)
            print(
                f"Predicted {record.get('code_name')} from {result.prices[0]['timestamp']} "
                f"to {result.prices[-1]['timestamp']}"
            )
    return enriched_records


def save_predictions(records: List[Dict[str, Any]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(records, fp, ensure_ascii=False, indent=2)
    print(f"Saved {len(records)} records to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict A-share prices from AkShare records")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing JSON files (default: examples directory)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to save predictions JSON (default: akshare_predictions.json)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for model inference (auto-detect by default)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_json_records(args.input)
    if not records:
        print("No JSON records found. Place files alongside this script or specify --input.")
        return

    predictor = load_predictor(device=args.device)
    enriched = run_prediction(records, predictor)
    save_predictions(enriched, args.output)


if __name__ == "__main__":
    main()

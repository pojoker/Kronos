"""AkShare-powered batch prediction workflow for Kronos.

This script reads JSON files located alongside it, extracts the
``time``, ``title`` and ``code_name`` fields, downloads the corresponding
A-share OHLCV history via AkShare, and generates three business days of
Kronos forecasts per entry.  The output is written to a consolidated
JSON file containing the original metadata plus the predicted price
series.

Usage
-----
1. Install AkShare alongside the project requirements::

       pip install -r requirements.txt akshare

2. Place one or more JSON files in the same directory as this script.
   Each JSON file should contain a list (or dictionary with a ``data``
   list) of entries that include at least the ``time``, ``title`` and
   ``code_name`` fields.  Additional keys (``content``, ``company_chn_name``
   etc.) will be preserved in the output.

3. Run the script::

       python examples/prediction_akshare_example.py

   The script will create ``akshare_predictions.json`` containing the
   enriched records.  For each entry the forecast window starts on the
   event date unless the event timestamp is later than 15:00:00, in
   which case the window begins on the next business day.
"""

from __future__ import annotations

import json
from datetime import time as dt_time
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, MutableMapping, Sequence

import akshare as ak
import pandas as pd
from pandas.tseries.offsets import BusinessDay
import torch

# Resolve the project root even when ``__file__`` is undefined (e.g. in notebooks).
try:
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:  # pragma: no cover - triggered in interactive sessions
    _PROJECT_ROOT = Path.cwd().resolve()

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))
from model import Kronos, KronosTokenizer, KronosPredictor

LOOKBACK_DAYS = 300
PREDICTION_LENGTH = 3
REQUIRED_FIELDS = {"time", "title", "code_name"}


def fetch_a_share_daily(symbol: str) -> pd.DataFrame:
    """Fetch daily OHLCV data for a given A-share symbol via AkShare."""

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

    df = df[list(rename_map.values())].copy()
    df["timestamps"] = pd.to_datetime(df["timestamps"], utc=False)
    numeric_cols = ["open", "high", "low", "close", "volume", "amount"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=numeric_cols).sort_values("timestamps").reset_index(drop=True)
    return df


def load_json_records(directory: Path) -> List[Dict[str, Any]]:
    """Load and flatten all JSON files within ``directory``."""

    records: List[Dict[str, Any]] = []
    for json_path in sorted(directory.glob("*.json")):
        try:
            with json_path.open("r", encoding="utf-8") as fp:
                payload = json.load(fp)
        except json.JSONDecodeError as exc:
            print(f"Skipping {json_path.name}: invalid JSON ({exc}).")
            continue

        candidates: Iterable[Any]
        if isinstance(payload, list):
            candidates = payload
        elif isinstance(payload, dict):
            if isinstance(payload.get("data"), list):
                candidates = payload["data"]
            else:
                candidates = [payload]
        else:
            print(f"Skipping {json_path.name}: unsupported JSON structure {type(payload)!r}.")
            continue

        for item in candidates:
            if isinstance(item, dict):
                records.append(item)
            else:
                print(f"Skipping entry in {json_path.name}: expected object, got {type(item)!r}.")
    return records


def normalise_symbol(code_name: str) -> str:
    """Extract the 6-digit symbol recognised by AkShare from ``code_name``."""

    digits = "".join(ch for ch in str(code_name) if ch.isdigit())
    if len(digits) == 6:
        return digits
    return str(code_name)


def parse_event_time(raw_time: Any) -> pd.Timestamp:
    """Parse an event timestamp into a timezone-naive ``Timestamp``."""

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


def ensure_series(indexable: Sequence[pd.Timestamp] | Iterable[pd.Timestamp]) -> pd.Series:
    """Convert timestamps into a pandas Series with a working ``.dt`` accessor."""

    if isinstance(indexable, pd.Series):
        converted = pd.to_datetime(indexable, errors="coerce")
    elif isinstance(indexable, pd.Index):
        converted = pd.to_datetime(indexable, errors="coerce")
    else:
        converted = pd.to_datetime(list(indexable), errors="coerce")

    if isinstance(converted, pd.Series):
        values = pd.Index(converted)
    else:
        values = pd.Index(converted)

    if getattr(values, "tz", None) is not None:
        values = values.tz_convert(None)
    try:
        values = values.tz_localize(None)
    except (TypeError, AttributeError):
        pass

    if values.hasnans:
        raise ValueError("Timestamp sequence contains invalid entries.")

    return pd.Series(values).reset_index(drop=True)


def determine_forecast_start(event_ts: pd.Timestamp) -> pd.Timestamp:
    """Compute the forecast anchor date given an event timestamp."""

    anchor = event_ts.normalize()
    if event_ts.time() > dt_time(15, 0):
        anchor = anchor + BusinessDay(1)
    return anchor


def build_prediction_payload(pred_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert the Kronos prediction DataFrame into a JSON-serialisable list."""

    records: List[Dict[str, Any]] = []
    for ts, row in pred_df.iterrows():
        records.append(
            {
                "date": ts.strftime("%Y-%m-%d"),
                "open": round(float(row["open"]), 4),
                "high": round(float(row["high"]), 4),
                "low": round(float(row["low"]), 4),
                "close": round(float(row["close"]), 4),
                "volume": round(float(row["volume"]), 2),
                "amount": round(float(row["amount"]), 2),
            }
        )
    return records


def enrich_record_with_prediction(
    record: MutableMapping[str, Any],
    predictor: KronosPredictor,
    data_cache: Dict[str, pd.DataFrame],
) -> Dict[str, Any]:
    """Return a copy of ``record`` augmented with three-day Kronos predictions."""

    missing = REQUIRED_FIELDS - record.keys()
    if missing:
        raise ValueError(f"Record is missing required fields: {sorted(missing)}")

    event_ts = parse_event_time(record["time"])
    forecast_start = determine_forecast_start(event_ts)
    symbol_key = str(record["code_name"]).strip()
    symbol = normalise_symbol(symbol_key)

    if symbol not in data_cache:
        print(f"Downloading daily data for {symbol} via AkShare...")
        data_cache[symbol] = fetch_a_share_daily(symbol)
    df = data_cache[symbol]

    history_df = df[df["timestamps"] < forecast_start]
    if history_df.shape[0] < LOOKBACK_DAYS:
        raise ValueError(
            f"Not enough historical data before {forecast_start.date()} for symbol {symbol}. "
            f"Expected at least {LOOKBACK_DAYS} trading days, got {history_df.shape[0]}."
        )

    df_window = history_df.iloc[-LOOKBACK_DAYS:].reset_index(drop=True)
    x_df = df_window.loc[:, ["open", "high", "low", "close", "volume", "amount"]]
    x_timestamp = ensure_series(df_window.loc[:, "timestamps"])

    y_index = pd.bdate_range(forecast_start, periods=PREDICTION_LENGTH, tz=None)
    if len(y_index) != PREDICTION_LENGTH:
        raise ValueError(
            f"Unable to build a {PREDICTION_LENGTH}-day business window starting from {forecast_start.date()}."
        )
    y_timestamp = ensure_series(y_index)

    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=PREDICTION_LENGTH,
        T=1.0,
        top_p=0.9,
        sample_count=1,
        verbose=False,
    )

    enriched = dict(record)
    enriched["code_name"] = symbol_key
    enriched["forecast_start_date"] = y_index[0].strftime("%Y-%m-%d")
    enriched["predicted_prices"] = build_prediction_payload(pred_df)
    return enriched


def main(input_dir: str | Path | None = None, output_path: str | Path | None = None) -> None:
    script_dir = Path(__file__).resolve().parent
    input_directory = Path(input_dir) if input_dir is not None else script_dir
    output_file = Path(output_path) if output_path is not None else script_dir / "akshare_predictions.json"

    records = load_json_records(input_directory)
    if not records:
        print(f"No JSON records found in {input_directory}. Nothing to predict.")
        return

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Loading Kronos models on device: {device}")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)

    enriched_records: List[Dict[str, Any]] = []
    data_cache: Dict[str, pd.DataFrame] = {}

    for record in records:
        try:
            enriched = enrich_record_with_prediction(record, predictor, data_cache)
        except Exception as exc:
            print(
                f"Skipping record with code_name={record.get('code_name')}: {exc}")
            errored = dict(record)
            errored["prediction_error"] = str(exc)
            enriched_records.append(errored)
        else:
            enriched_records.append(enriched)

    output_file.write_text(
        json.dumps(enriched_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved {len(enriched_records)} records to {output_file}")


if __name__ == "__main__":
    main()

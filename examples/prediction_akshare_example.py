"""AkShare-powered batch prediction workflow for Kronos.

This module can be executed as a standalone script or imported into a
Jupyter notebook.  When imported, helper functions are exposed so that
you can interactively load JSON event records, prepare the Kronos
predictor, and generate forecasts without touching the filesystem unless
you explicitly choose to.

Notebook quick-start
=====================

1. Install dependencies::

       %pip install -r requirements.txt akshare

2. Import helpers and load a predictor::

       from examples.prediction_akshare_example import (
           load_json_records,
           load_kronos_predictor,
           predict_records,
       )
       predictor = load_kronos_predictor()

3. Supply records and display predictions::

       records = load_json_records(Path("path/to/json_dir"))
       results = predict_records(records, predictor)
       results[0]["predicted_prices"]

CLI usage
=========

Running ``python examples/prediction_akshare_example.py`` keeps the
previous behaviour: JSON files located alongside the script are
processed and the enriched records are written to
``akshare_predictions.json``.
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

__all__ = [
    "LOOKBACK_DAYS",
    "PREDICTION_LENGTH",
    "REQUIRED_FIELDS",
    "fetch_a_share_daily",
    "load_json_records",
    "normalise_symbol",
    "parse_event_time",
    "ensure_series",
    "determine_forecast_start",
    "build_prediction_payload",
    "enrich_record_with_prediction",
    "load_kronos_predictor",
    "predict_records",
    "predict_directory",
    "save_predictions",
]

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


def load_kronos_predictor(
    *, device: str | None = None, max_context: int = 512
) -> KronosPredictor:
    """Instantiate ``KronosPredictor`` for interactive use.

    Parameters
    ----------
    device:
        Optional device string (``"cpu"``, ``"cuda:0"`` 等)。若未提供则
        根据 ``torch.cuda.is_available`` 自动选择。
    max_context:
        历史窗口的最大长度，默认 512，对应 Kronos-small 的限制。

    Returns
    -------
    KronosPredictor
        已加载的预测器，可在 Notebook 中复用。
    """

    resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Loading Kronos models on device: {resolved_device}")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    return KronosPredictor(model, tokenizer, device=resolved_device, max_context=max_context)


def predict_records(
    records: Sequence[MutableMapping[str, Any]],
    predictor: KronosPredictor | None = None,
    *,
    device: str | None = None,
    data_cache: Dict[str, pd.DataFrame] | None = None,
    on_error: str = "include",
) -> List[Dict[str, Any]]:
    """Generate predictions for in-memory records.

    Parameters
    ----------
    records:
        事件列表，每个元素需包含 ``time``、``title``、``code_name`` 字段。
    predictor:
        已初始化的 ``KronosPredictor``。若缺省则会调用
        :func:`load_kronos_predictor`。
    device:
        当 ``predictor`` 缺省时用于加载模型的设备字符串。
    data_cache:
        可选的行情缓存字典，可在多次调用之间复用以减少 AkShare 请求。
    on_error:
        ``"include"``（默认）表示在结果中保留失败记录并附带 ``prediction_error``；
        ``"skip"`` 表示忽略失败记录；``"raise"`` 会在第一条失败时立即抛出异常。

    Returns
    -------
    list of dict
        与输入对应的一组 enriched 记录。
    """

    if predictor is None:
        predictor = load_kronos_predictor(device=device)

    cache = data_cache if data_cache is not None else {}
    enriched_records: List[Dict[str, Any]] = []

    for record in records:
        try:
            enriched_records.append(enrich_record_with_prediction(record, predictor, cache))
        except Exception as exc:  # pragma: no cover - network errors, data issues
            if on_error == "raise":
                raise
            if on_error == "skip":
                continue
            errored = dict(record)
            errored["prediction_error"] = str(exc)
            enriched_records.append(errored)

    return enriched_records


def predict_directory(
    input_dir: str | Path,
    predictor: KronosPredictor | None = None,
    **predict_kwargs: Any,
) -> List[Dict[str, Any]]:
    """Convenience wrapper to load JSON files from a directory and predict."""

    directory = Path(input_dir)
    records = load_json_records(directory)
    if not records:
        print(f"No JSON records found in {directory}. Nothing to predict.")
        return []
    return predict_records(records, predictor, **predict_kwargs)


def save_predictions(records: Sequence[Dict[str, Any]], output_path: str | Path) -> Path:
    """Persist enriched predictions to ``output_path`` and return the path."""

    output_file = Path(output_path)
    output_file.write_text(
        json.dumps(list(records), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved {len(records)} records to {output_file}")
    return output_file


def main(input_dir: str | Path | None = None, output_path: str | Path | None = None) -> None:
    script_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    input_directory = Path(input_dir) if input_dir is not None else script_dir
    output_file = Path(output_path) if output_path is not None else script_dir / "akshare_predictions.json"

    predictor = load_kronos_predictor()
    enriched_records = predict_directory(input_directory, predictor)
    if enriched_records:
        save_predictions(enriched_records, output_file)


if __name__ == "__main__":
    main()

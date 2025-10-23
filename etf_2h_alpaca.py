import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List

import pandas as pd
import requests

API_KEY = os.environ["ALPACA_KEY_ID"]
API_SEC = os.environ["ALPACA_SECRET_KEY"]
FEED = os.getenv("ALPACA_FEED", "sip")

BASE = "https://data.alpaca.markets/v2/stocks/bars"


def load_symbols(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip() and not line.startswith("#")]


def chunks(items: List[str], size: int) -> Iterable[List[str]]:
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


def fetch_bars(
    symbols: List[str],
    start: datetime,
    end: datetime,
    timeframe: str,
    limit: int,
) -> Dict[str, List[dict]]:
    headers = {"APCA-API-KEY-ID": API_KEY, "APCA-API-SECRET-KEY": API_SEC}
    params_common = {
        "timeframe": timeframe,
        "adjustment": "raw",
        "start": start.isoformat(),
        "end": end.isoformat(),
        "feed": FEED,
        "limit": limit,
    }

    collected: Dict[str, List[dict]] = {}
    for batch in chunks(symbols, 50):
        params = {**params_common, "symbols": ",".join(batch)}
        response = requests.get(BASE, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        data = response.json().get("bars", {})
        for sym, bars in data.items():
            if not bars:
                continue
            collected.setdefault(sym, []).extend(bars)

    return collected


def build_two_hour_summary(symbol_bars: Dict[str, List[dict]], end: datetime) -> pd.DataFrame:
    rows = []
    cutoff = end - timedelta(hours=2)
    for sym, bars in symbol_bars.items():
        frame = pd.DataFrame(bars)
        frame["t"] = pd.to_datetime(frame["t"], utc=True)
        frame = frame.sort_values("t")
        frame = frame[frame["t"] >= cutoff]
        if frame.empty:
            continue
        first_close = float(frame["c"].iloc[0])
        last_close = float(frame["c"].iloc[-1])
        pct = (last_close / first_close - 1.0) * 100.0
        rows.append(
            {
                "Ticker": sym,
                "FirstClose": first_close,
                "LastClose": last_close,
                "Bars": int(frame.shape[0]),
                "PctChange_2h": pct,
            }
        )

    return pd.DataFrame(rows).sort_values("PctChange_2h", ascending=False)


def build_hourly_dataset(symbol_bars: Dict[str, List[dict]], end: datetime) -> pd.DataFrame:
    cutoff = end - timedelta(hours=48)
    rows = []
    for sym, bars in symbol_bars.items():
        frame = pd.DataFrame(bars)
        frame["t"] = pd.to_datetime(frame["t"], utc=True)
        frame = frame.sort_values("t")
        frame = frame[frame["t"] >= cutoff]
        if frame.empty:
            continue
        for row in frame.itertuples():
            rows.append(
                {
                    "Ticker": sym,
                    "Timestamp": row.t.isoformat(),
                    "Open": float(row.o),
                    "High": float(row.h),
                    "Low": float(row.l),
                    "Close": float(row.c),
                    "Volume": int(row.v),
                    "Trades": int(row.n),
                    "VWAP": float(row.vw) if pd.notna(row.vw) else float("nan"),
                }
            )

    hourly = pd.DataFrame(rows)
    if not hourly.empty:
        hourly = hourly.sort_values(["Ticker", "Timestamp"])
    return hourly


def write_summary(two_hour: pd.DataFrame, hourly_path: str, end: datetime) -> None:
    top = two_hour.head(15)
    lines = [
        "# Top ETF Gainers — Last 2 Hours",
        f"_Feed: **{FEED.upper()}**  •  Generated at **{end.isoformat()}**_",
        "",
        "| Rank | Ticker | % Change (2h) | Bars | Last | First |",
        "|---:|:------:|--------------:|----:|-----:|------:|",
    ]
    for i, row in top.reset_index(drop=True).iterrows():
        lines.append(
            "| {rank} | **{ticker}** | {pct:.2f}% | {bars} | {last:.2f} | {first:.2f} |".format(
                rank=i + 1,
                ticker=row["Ticker"],
                pct=row["PctChange_2h"],
                bars=row["Bars"],
                last=row["LastClose"],
                first=row["FirstClose"],
            )
        )

    lines.extend(
        [
            "",
            "## Hourly OHLCV Export",
            f"Saved **48 hours** of hourly OHLCV bars to `{hourly_path}` for all requested symbols.",
            "Data includes open, high, low, close, volume, trades, and VWAP columns.",
            "",
            "### SIP Connection Info",
            "* Live trading base URL: https://api.alpaca.markets",
            "* Provide `ALPACA_KEY_ID` and `ALPACA_SECRET_KEY` as environment variables (SIP credentials).",
            f"* Requested market data feed: `{FEED}`",
        ]
    )

    with open("SUMMARY.md", "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


def main():
    syms_path = sys.argv[1] if len(sys.argv) > 1 else "etfs.txt"
    syms = load_symbols(syms_path)
    if not syms:
        print("No symbols found in etfs.txt", file=sys.stderr)
        sys.exit(1)

    end = datetime.now(timezone.utc)

    minute_start = end - timedelta(hours=2, minutes=5)
    minute_bars = fetch_bars(syms, minute_start, end, timeframe="1Min", limit=2000)
    two_hour = build_two_hour_summary(minute_bars, end)
    two_hour.to_csv("etf_gainers_2h.csv", index=False)

    hourly_start = end - timedelta(hours=48, minutes=5)
    hourly_bars = fetch_bars(syms, hourly_start, end, timeframe="1Hour", limit=400)
    hourly = build_hourly_dataset(hourly_bars, end)
    hourly_path = "etf_hourly_48h.csv"
    hourly.to_csv(hourly_path, index=False)

    write_summary(two_hour, hourly_path, end)

    print("Wrote etf_gainers_2h.csv, etf_hourly_48h.csv, and SUMMARY.md")


if __name__ == "__main__":
    main()

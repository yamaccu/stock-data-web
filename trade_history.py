"""Parse Rakuten Securities Japanese trade history CSV for chart overlays."""

from __future__ import annotations

from io import BytesIO
import unicodedata

import pandas as pd

REQUIRED_COLUMNS = {"約定日", "銘柄コード", "売買区分", "数量[株]", "単価[円]"}


def read_trade_history(raw_bytes: bytes) -> pd.DataFrame:
    """Return per-fill trade rows; accepts UTF-8 (BOM) and Shift-JIS/CP932."""
    if not raw_bytes:
        raise ValueError("取引履歴CSVが空です。")

    frame = None
    for encoding in ("utf-8-sig", "cp932"):
        try:
            frame = pd.read_csv(BytesIO(raw_bytes), dtype=str, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError as exc:
            raise ValueError("取引履歴CSVが空です。") from exc
    if frame is None:
        raise ValueError("文字コードを読み取れません。UTF-8またはShift-JISのCSVを指定してください。")

    frame.columns = [unicodedata.normalize("NFKC", str(c)).strip() for c in frame.columns]
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"楽天証券CSVの必要な列がありません: {', '.join(sorted(missing))}")

    # A settlement date is not a trade date. Use 約定日 for the x-axis.
    dates = pd.to_datetime(frame["約定日"].str.strip(), errors="coerce", format="%Y/%m/%d")
    if dates.isna().any():
        alt_dates = pd.to_datetime(frame.loc[dates.isna(), "約定日"], errors="coerce")
        dates.loc[dates.isna()] = alt_dates

    directions = frame["売買区分"].fillna("").astype(str)
    buy = directions.str.contains("買", regex=False) & ~directions.str.contains("売", regex=False)
    sell = directions.str.contains("売", regex=False) & ~directions.str.contains("買", regex=False)
    quantity = pd.to_numeric(
        frame["数量[株]"].fillna("").str.replace(",", "", regex=False), errors="coerce"
    )
    price = pd.to_numeric(
        frame["単価[円]"].fillna("").str.replace(",", "", regex=False), errors="coerce"
    )

    result = pd.DataFrame(
        {
            "Date": dates,
            "Code": frame["銘柄コード"].fillna("").str.strip().str.upper().str.replace(r"\.T$", "", regex=True),
            "Side": pd.Series("", index=frame.index).mask(buy, "buy").mask(sell, "sell"),
            "Quantity": quantity,
            "Price": price,
        }
    )
    valid = (
        result["Date"].notna()
        & result["Side"].ne("")
        & result["Quantity"].gt(0)
        & result["Price"].gt(0)
        & result["Code"].ne("")
    )
    return result.loc[valid].copy()


def aggregate_chart_trades(
    trades: pd.DataFrame, symbol: str, valid_dates: pd.DatetimeIndex
) -> pd.DataFrame:
    """Sum same-date same-side fills and use quantity-weighted execution price."""
    code = symbol.upper().removesuffix(".T")
    valid_dates = pd.DatetimeIndex(valid_dates).normalize()
    selected = trades.loc[
        trades["Code"].eq(code) & trades["Date"].isin(valid_dates)
    ].copy()
    if selected.empty:
        return pd.DataFrame(columns=["Date", "Side", "Quantity", "Price", "Fills"])

    selected["TradeValue"] = selected["Quantity"] * selected["Price"]
    grouped = (
        selected.groupby(["Date", "Side"], as_index=False)
        .agg(Quantity=("Quantity", "sum"), TradeValue=("TradeValue", "sum"), Fills=("Price", "size"))
        .sort_values(["Date", "Side"])
    )
    grouped["Price"] = grouped["TradeValue"] / grouped["Quantity"]
    return grouped[["Date", "Side", "Quantity", "Price", "Fills"]]

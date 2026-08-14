from __future__ import annotations

import re
import pandas as pd
import streamlit as st
import yfinance as yf

APP_TITLE = "株価データ取得"
DAILY_PERIOD = "1y"
HOURLY_PERIOD = "60d"

st.set_page_config(page_title=APP_TITLE, page_icon="📈", layout="centered")


def normalize_ticker(code: str) -> tuple[str, str]:
    raw = code.strip().upper().replace(" ", "")
    if not raw:
        raise ValueError("銘柄コードを入力してください。")

    if re.fullmatch(r"[0-9A-Z]{4}", raw):
        return raw, f"{raw}.T"

    return raw, raw


def prepare_ohlcv(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    if df.empty:
        raise ValueError(
            "株価データを取得できませんでした。"
            "銘柄コードまたはYahoo Finance側のデータ提供状況を確認してください。"
        )

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"必要な列がありません: {', '.join(missing)}")

    out = df[required].copy()
    out = out.dropna(subset=["Open", "High", "Low", "Close"], how="all")

    idx = pd.DatetimeIndex(out.index)

    if interval == "1h" and idx.tz is not None:
        idx = idx.tz_convert("Asia/Tokyo").tz_localize(None)
    elif idx.tz is not None:
        idx = idx.tz_localize(None)

    out.index = idx

    if interval == "1d":
        out.index.name = "Date"
        out = out.reset_index()
        out["Date"] = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")
    else:
        out.index.name = "Datetime"
        out = out.reset_index()
        out["Datetime"] = pd.to_datetime(out["Datetime"]).dt.strftime("%Y-%m-%d %H:%M")

    return out


@st.cache_data(ttl=300, show_spinner=False)
def fetch_stock_data(code: str):
    file_code, ticker_symbol = normalize_ticker(code)
    ticker = yf.Ticker(ticker_symbol)

    daily_raw = ticker.history(
        period=DAILY_PERIOD,
        interval="1d",
        auto_adjust=False,
        actions=False,
        prepost=False,
        timeout=20,
    )

    hourly_raw = ticker.history(
        period=HOURLY_PERIOD,
        interval="1h",
        auto_adjust=False,
        actions=False,
        prepost=False,
        timeout=20,
    )

    daily = prepare_ohlcv(daily_raw, "1d")
    hourly = prepare_ohlcv(hourly_raw, "1h")
    return daily, hourly, file_code, ticker_symbol


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


st.title("📈 株価データ取得")
st.caption("銘柄コードを入力すると、日足と1時間足のCSVを作成します。")

with st.form("stock_form"):
    code = st.text_input(
        "銘柄コード",
        placeholder="例: 7186",
        help="日本株の4文字コードは自動で .T を付けます。^N225 / GC=F / BTC-USD なども入力できます。",
    )

    submitted = st.form_submit_button(
        "日足・1時間足を取得",
        use_container_width=True,
        type="primary",
    )

if submitted:
    try:
        with st.spinner("株価データを取得しています..."):
            daily, hourly, file_code, ticker_symbol = fetch_stock_data(code)

        st.success(
            f"{ticker_symbol} の取得が完了しました。"
            f" 日足 {len(daily):,}行 / 1時間足 {len(hourly):,}行"
        )

        col1, col2 = st.columns(2)

        with col1:
            st.download_button(
                "日足CSVをダウンロード",
                data=dataframe_to_csv_bytes(daily),
                file_name=f"{file_code}_daily.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with col2:
            st.download_button(
                "1時間足CSVをダウンロード",
                data=dataframe_to_csv_bytes(hourly),
                file_name=f"{file_code}_1h.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with st.expander("取得データを確認"):
            st.markdown("**日足（末尾10行）**")
            st.dataframe(daily.tail(10), use_container_width=True, hide_index=True)

            st.markdown("**1時間足（末尾10行）**")
            st.dataframe(hourly.tail(10), use_container_width=True, hide_index=True)

    except Exception as exc:
        st.error(f"データ取得に失敗しました。\n\n{exc}")

st.divider()
st.caption(
    "日足: 直近1年 / 1時間足: 直近60日 / "
    "CSV列: Date(Datetime), Open, High, Low, Close, Volume"
)

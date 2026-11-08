from __future__ import annotations

import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

st.set_page_config(page_title="株価チャート", page_icon="📈", layout="wide")
MA_PERIODS = (5, 25, 75)
UP_COLOR = "#d9534f"
DOWN_COLOR = "#2476ad"


def normalize_ticker(code: str) -> str:
    raw = code.strip().upper().replace(" ", "")
    if not raw:
        raise ValueError("銘柄コードを入力してください。")
    return f"{raw}.T" if re.fullmatch(r"[0-9A-Z]{4}", raw) else raw


@st.cache_data(ttl=300, show_spinner=False)
def fetch_stock_data(symbol: str) -> pd.DataFrame:
    # 75日線を表示初日から計算するため、内部では1年分取得。
    raw = yf.Ticker(symbol).history(
        period="1y", interval="1d", auto_adjust=False,
        actions=False, prepost=False, timeout=20,
    )
    if raw.empty:
        raise ValueError("株価データを取得できません。銘柄コードやデータ提供状況を確認してください。")

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [name for name in required if name not in raw.columns]
    if missing:
        raise ValueError(f"必要な列がありません: {', '.join(missing)}")

    df = raw[required].dropna(subset=["Open", "High", "Low", "Close"]).copy()
    if df.empty:
        raise ValueError("表示できる日足データがありません。")

    dates = pd.DatetimeIndex(df.index)
    df.index = (dates.tz_localize(None) if dates.tz is not None else dates).normalize()
    df = df.sort_index().loc[lambda x: ~x.index.duplicated(keep="last")]
    for period in MA_PERIODS:
        df[f"MA{period}"] = df["Close"].rolling(period, min_periods=period).mean()
    return df


def build_chart(df: pd.DataFrame) -> go.Figure:
    x = df.index.strftime("%Y-%m-%d").tolist()
    colors = [
        UP_COLOR if close >= open_ else DOWN_COLOR
        for open_, close in zip(df["Open"], df["Close"])
    ]
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.76, 0.24], vertical_spacing=0.04,
    )
    fig.add_trace(
        go.Candlestick(
            x=x, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"], name="日足",
            increasing_line_color=UP_COLOR, decreasing_line_color=DOWN_COLOR,
            increasing_fillcolor=UP_COLOR, decreasing_fillcolor=DOWN_COLOR,
        ),
        row=1, col=1,
    )
    ma_colors = {5: "#e69f00", 25: "#2b8cbe", 75: "#3c9d5d"}
    for period in MA_PERIODS:
        fig.add_trace(
            go.Scatter(
                x=x, y=df[f"MA{period}"], mode="lines",
                name=f"MA{period}", line=dict(color=ma_colors[period], width=2),
            ),
            row=1, col=1,
        )
    fig.add_trace(
        go.Bar(
            x=x, y=df["Volume"], name="出来高",
            marker_color=colors, showlegend=False,
            hovertemplate="%{y:,.0f}<extra>出来高</extra>",
        ),
        row=2, col=1,
    )

    # 休場日の空白をなくし、実際の取引日のみ等間隔で配置する。
    step = max(1, len(x) // 8)
    ticks = x[::step]
    fig.update_xaxes(type="category", showgrid=False)
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(
        tickmode="array", tickvals=ticks,
        ticktext=[date[5:] for date in ticks],
        title_text="日付（月/日）", row=2, col=1,
    )
    fig.update_yaxes(title_text="株価", tickformat=",", row=1, col=1)
    fig.update_yaxes(title_text="出来高", tickformat="~s", row=2, col=1)
    fig.update_layout(
        template="plotly_white", height=690,
        margin=dict(l=30, r=20, t=65, b=25),
        hovermode="x unified", dragmode="zoom",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
    )
    return fig


st.title("📈 株価チャート")
st.caption("直近4か月の日足・5日/25日/75日移動平均線・出来高を表示します。")
with st.form("stock_form"):
    code = st.text_input(
        "銘柄コード", placeholder="例: 7186",
        help="日本株の4文字コードには自動で .T を付けます。^N225 / GC=F / BTC-USD なども指定できます。",
    )
    submitted = st.form_submit_button(
        "チャートを表示", use_container_width=True, type="primary",
    )

if submitted:
    try:
        symbol = normalize_ticker(code)
        with st.spinner("株価データを取得しています..."):
            data = fetch_stock_data(symbol)
        start = data.index.max() - pd.DateOffset(months=4)
        chart_data = data.loc[data.index >= start]
        if chart_data.empty:
            raise ValueError("表示期間内に株価データがありません。")
        st.subheader(symbol)
        st.caption(
            f"表示期間: {chart_data.index.min():%Y/%m/%d} ～ "
            f"{chart_data.index.max():%Y/%m/%d}（{len(chart_data)}営業日）"
        )
        st.plotly_chart(build_chart(chart_data), use_container_width=True)
        if chart_data["MA75"].isna().all():
            st.info("過去データが75営業日未満のため、75日移動平均線は表示されません。")
    except Exception as exc:
        st.error(f"チャート表示に失敗しました。\n\n{exc}")

st.divider()
st.caption("データ提供: Yahoo Finance / 取得結果は5分間キャッシュされます。")

from __future__ import annotations

import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

from trade_history import aggregate_chart_trades, read_trade_history

st.set_page_config(page_title="株価チャート", page_icon="📈", layout="wide")

MA_PERIODS = (5, 25, 75)
CHART_MONTHS = 6
INITIAL_VIEW_MONTHS = 3
RSI_PERIOD = 14
ADX_PERIOD = 14
UP_COLOR = "#d9534f"
DOWN_COLOR = "#2476ad"


def normalize_ticker(code: str) -> str:
    raw = code.strip().upper().replace(" ", "")
    if not raw:
        raise ValueError("銘柄コードを入力してください。")
    return f"{raw}.T" if re.fullmatch(r"[0-9A-Z]{4}", raw) else raw


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ticker_name(symbol: str) -> str:
    """名称取得に失敗しても、株価チャートの表示は妨げない。"""
    try:
        info = yf.Ticker(symbol).get_info()
        if isinstance(info, dict):
            for key in ("longName", "shortName", "displayName"):
                name = info.get(key)
                if isinstance(name, str) and name.strip():
                    return name.strip()
    except Exception:
        pass
    return ""


def calculate_directional_indicators(
    data: pd.DataFrame, period: int = ADX_PERIOD
) -> pd.DataFrame:
    """marketdataと同じWilder方式でADX、+DI、-DIを計算する。"""
    high = data["High"]
    low = data["Low"]
    close = data["Close"]

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where(
        (up_move > down_move) & (up_move > 0), 0.0
    )
    minus_dm = down_move.where(
        (down_move > up_move) & (down_move > 0), 0.0
    )

    true_range = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)

    def wilder_average(values: pd.Series, start: int) -> pd.Series:
        """初回は単純平均、以降はWilder平滑化。"""
        average = pd.Series(float("nan"), index=values.index)
        first = start + period - 1
        if len(values) <= first:
            return average

        average.iloc[first] = values.iloc[start:first + 1].mean()
        for i in range(first + 1, len(values)):
            average.iloc[i] = (
                average.iloc[i - 1] * (period - 1) + values.iloc[i]
            ) / period
        return average

    smoothed_tr = wilder_average(true_range, start=1)
    plus_di = 100 * wilder_average(plus_dm, start=1) / smoothed_tr
    minus_di = 100 * wilder_average(minus_dm, start=1) / smoothed_tr

    di_sum = (plus_di + minus_di).replace(0, float("nan"))
    dx = (100 * (plus_di - minus_di).abs() / di_sum).fillna(0.0)
    return pd.DataFrame(
        {
            f"ADX{period}": wilder_average(dx, start=period),
            f"+DI{period}": plus_di,
            f"-DI{period}": minus_di,
        },
        index=data.index,
    )


def add_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """移動平均と、marketdataと同方式のRSI/ADX/DIを計算する。"""
    result = data.copy()
    close = result["Close"]

    for period in MA_PERIODS:
        result[f"MA{period}"] = close.rolling(
            window=period, min_periods=period
        ).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(
        alpha=1 / RSI_PERIOD, adjust=False, min_periods=RSI_PERIOD
    ).mean()
    avg_loss = loss.ewm(
        alpha=1 / RSI_PERIOD, adjust=False, min_periods=RSI_PERIOD
    ).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.mask((avg_gain == 0) & (avg_loss == 0), 50.0)
    rsi = rsi.mask((avg_gain > 0) & (avg_loss == 0), 100.0)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss > 0), 0.0)
    result[f"RSI{RSI_PERIOD}"] = rsi

    directional = calculate_directional_indicators(result)
    result[directional.columns] = directional
    return result


@st.cache_data(ttl=300, show_spinner=False)
def fetch_stock_data(symbol: str) -> pd.DataFrame:
    # 表示開始時点から75日線やADXを計算するため、過去1年分を取得。
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
    return add_technical_indicators(df)



def add_trade_markers(
    fig: go.Figure, chart_data: pd.DataFrame, trades: pd.DataFrame
) -> None:
    """Draw quantities inside triangles, without displaying execution prices."""
    # Put buy markers below each daily candle and sell markers above it.
    # This keeps both visible when buy and sell executions occur on the same day.
    price_span = float(chart_data["High"].max() - chart_data["Low"].min())
    median_close = float(chart_data["Close"].median())
    padding = max(price_span * 0.05, abs(median_close) * 0.005, 0.01)

    for side, label, symbol, color in (
        ("buy", "買い", "triangle-up", "#c62828"),
        ("sell", "売り", "triangle-down", "#1565c0"),
    ):
        events = trades.loc[trades["Side"].eq(side)]
        if events.empty:
            continue

        x_values = events["Date"].dt.strftime("%Y-%m-%d").tolist()
        marker_y = [
            float(chart_data.loc[date, "Low"] - padding)
            if side == "buy"
            else float(chart_data.loc[date, "High"] + padding)
            for date in events["Date"]
        ]
        quantities = [f"{int(quantity):,}" for quantity in events["Quantity"]]
        sizes = [min(90, max(64, 42 + 6 * len(q))) for q in quantities]
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=marker_y,
                mode="markers+text",
                name=f"{label}（数量）",
                marker=dict(
                    symbol=symbol, size=sizes, color=color,
                    line=dict(color="white", width=1),
                ),
                text=quantities,
                textposition="middle center",
                textfont=dict(size=10, color="white", family="Arial Black"),
                cliponaxis=False,
                customdata=events["Fills"].tolist(),
                hovertemplate=(
                    f"{label}: <b>%{{text}}</b><br>"
                    "約定件数: %{customdata}件<extra></extra>"
                ),
            ),
            row=1, col=1,
        )

def build_chart(
    df: pd.DataFrame, initial_start: pd.Timestamp, trades: pd.DataFrame | None = None
) -> go.Figure:
    x = df.index.strftime("%Y-%m-%d").tolist()
    first_visible_index = int(df.index.searchsorted(initial_start, side="left"))
    volume_colors = [
        UP_COLOR if close >= open_ else DOWN_COLOR
        for open_, close in zip(df["Open"], df["Close"])
    ]
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.50, 0.16, 0.17, 0.17],
        vertical_spacing=0.035,
    )

    fig.add_trace(
        go.Candlestick(
            x=x, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"], name="日足",
            increasing_line_color=UP_COLOR,
            decreasing_line_color=DOWN_COLOR,
            increasing_fillcolor=UP_COLOR,
            decreasing_fillcolor=DOWN_COLOR,
        ),
        row=1, col=1,
    )
    ma_colors = {5: "#e69f00", 25: "#2b8cbe", 75: "#3c9d5d"}
    for period in MA_PERIODS:
        fig.add_trace(
            go.Scatter(
                x=x, y=df[f"MA{period}"], mode="lines",
                name=f"MA{period}",
                line=dict(color=ma_colors[period], width=2),
                hovertemplate="%{y:,.1f}<extra>%{fullData.name}</extra>",
            ),
            row=1, col=1,
        )

    fig.add_trace(
        go.Bar(
            x=x, y=df["Volume"], name="出来高",
            marker_color=volume_colors, showlegend=False,
            hovertemplate="%{y:,.0f}<extra>出来高</extra>",
        ),
        row=2, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=x, y=df["RSI14"], mode="lines",
            name="RSI14", line=dict(color="#7957b3", width=2),
            hovertemplate="%{y:.1f}<extra>RSI14</extra>",
        ),
        row=3, col=1,
    )
    for level in (30, 70):
        fig.add_hline(
            y=level, line_dash="dash", line_color="#999999",
            line_width=1, row=3, col=1,
        )

    di_colors = {
        "ADX14": "#7351ae",
        "+DI14": "#13866b",
        "-DI14": "#d14a57",
    }
    for name, color in di_colors.items():
        fig.add_trace(
            go.Scatter(
                x=x, y=df[name], mode="lines", name=name,
                line=dict(color=color, width=2),
                hovertemplate="%{y:.1f}<extra>%{fullData.name}</extra>",
            ),
            row=4, col=1,
        )
    fig.add_hline(
        y=25, line_dash="dash", line_color="#999999",
        line_width=1, row=4, col=1,
    )

    if trades is not None and not trades.empty:
        add_trade_markers(fig, df, trades)

    # 休場日は空白を作らず、営業日を等間隔に配置する。
    step = max(1, len(x) // 8)
    ticks = x[::step]
    # 6か月分を保持しつつ、初期表示は直近3か月に限定する。
    # category軸のrangeは、日付文字列ではなくカテゴリの位置（0始まり）で指定する。
    fig.update_xaxes(
        type="category",
        range=[first_visible_index - 0.5, len(x) - 0.5],
        showgrid=False,
    )
    for row in (1, 2, 3):
        fig.update_xaxes(showticklabels=False, row=row, col=1)
    fig.update_xaxes(
        tickmode="array", tickvals=ticks,
        ticktext=[date[5:] for date in ticks],
        title_text="日付（月/日）", row=4, col=1,
    )
    fig.update_yaxes(title_text="株価", tickformat=",", row=1, col=1)
    fig.update_yaxes(title_text="出来高", tickformat="~s", row=2, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)
    fig.update_yaxes(title_text="ADX / DI", rangemode="tozero", row=4, col=1)
    fig.update_layout(
        template="plotly_white", height=960,
        margin=dict(l=35, r=20, t=90, b=25),
        hovermode="x unified", dragmode="pan",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.03, x=0),
    )
    return fig


st.title("📈 株価チャート")
st.caption("半年分の日足を描画、直近3か月を初期表示します。楽天証券CSVを指定すると買い▲・売り▼に数量を重ねます。")
with st.form("stock_form"):
    code = st.text_input(
        "銘柄コード", placeholder="例: 7186",
        help="日本株の4文字コードには自動で .T を付けます。^N225 / GC=F / BTC-USD なども指定できます。",
    )
    trade_upload = st.file_uploader(
        "楽天証券の売買履歴CSV（任意）",
        type=["csv"],
        help="ファイルを指定しなければ、売買履歴なしでチャートを表示します。",
    )
    submitted = st.form_submit_button(
        "チャートを表示", use_container_width=True, type="primary",
    )

if submitted:
    try:
        symbol = normalize_ticker(code)
        with st.spinner("株価データを取得しています..."):
            data = fetch_stock_data(symbol)

        # 過去1年から指標を計算し、直近6か月分を描画。初期表示は3か月。
        latest_date = data.index.max()
        start = latest_date - pd.DateOffset(months=CHART_MONTHS)
        initial_start = latest_date - pd.DateOffset(months=INITIAL_VIEW_MONTHS)
        chart_data = data.loc[data.index >= start]
        if chart_data.empty:
            raise ValueError("表示期間内に株価データがありません。")

        name = fetch_ticker_name(symbol)
        st.subheader(f"{symbol}  {name}" if name else symbol)
        st.caption(
            f"描画期間: {chart_data.index.min():%Y/%m/%d} ～ "
            f"{chart_data.index.max():%Y/%m/%d}（{len(chart_data)}営業日） / "
            f"初期表示: 直近{INITIAL_VIEW_MONTHS}か月 / Panで過去に移動できます"
        )
        trade_events = None
        if trade_upload is not None:
            try:
                trade_rows = read_trade_history(trade_upload.getvalue())
                trade_events = aggregate_chart_trades(trade_rows, symbol, chart_data.index)
                if trade_events.empty:
                    st.info("アップロードしたCSVに、この銘柄の描画期間内の売買履歴はありません。")
                else:
                    fill_count = int(trade_events["Fills"].sum())
                    st.caption(
                        f"売買履歴: {fill_count}約定 / "
                        f"{len(trade_events)}マーカー（同日・同方向は合算）"
                    )
            except Exception as exc:
                st.warning(f"売買履歴CSVを読み込めませんでした。チャートのみ表示します。\n\n{exc}")

        st.plotly_chart(
            build_chart(chart_data, initial_start, trade_events),
            use_container_width=True,
        )

        if chart_data["MA75"].isna().all():
            st.info("過去データが75営業日未満のため、75日移動平均線は表示されません。")
        if chart_data["ADX14"].isna().all():
            st.info("過去データが少ないため、ADXはまだ計算できません。")
    except Exception as exc:
        st.error(f"チャート表示に失敗しました。\n\n{exc}")

st.divider()
st.caption("データ提供: Yahoo Finance / 株価取得は5分間、銘柄名は1時間キャッシュされます。")

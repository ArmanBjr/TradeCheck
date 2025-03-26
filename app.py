import streamlit as st
import requests
import pandas as pd
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
import pytz
import time
import re
import platform
import os

# ========== Global Notified Crossovers Tracker ==========
notified_crossovers = {}

# ========== Get Available Markets ==========
@st.cache_data
def list_available_markets():
    url = "https://api.coinex.com/v1/market/list"
    response = requests.get(url)
    data = response.json()
    return data["data"]

# ========== Fetch Kline Data ==========
def get_kline_data(symbol, interval="5min", limit=100):
    url = "https://api.coinex.com/v1/market/kline"
    params = {"market": symbol, "type": interval, "limit": limit}
    response = requests.get(url, params=params)
    data = response.json()["data"]
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume", "amount"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='s').dt.tz_localize('UTC').dt.tz_convert(selected_timezone)
    df.set_index("timestamp", inplace=True)
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df

# ========== Calculate EMAs ==========
def calculate_emas(df, short_window, long_window):
    df["EMA_short"] = EMAIndicator(df["close"], window=short_window).ema_indicator()
    df["EMA_long"] = EMAIndicator(df["close"], window=long_window).ema_indicator()
    df["RSI"] = RSIIndicator(close=df["close"], window=14).rsi()
    return df

# ========== Detect Crossovers ==========
def detect_crossovers(df):
    crossover_points = []
    for i in range(1, len(df)):
        prev_short = df["EMA_short"].iloc[i-1]
        prev_long = df["EMA_long"].iloc[i-1]
        curr_short = df["EMA_short"].iloc[i]
        curr_long = df["EMA_long"].iloc[i]

        if prev_short < prev_long and curr_short > curr_long:
            crossover_points.append((df.index[i], df["close"].iloc[i], df["RSI"].iloc[i], "bullish"))
        elif prev_short > prev_long and curr_short < curr_long:
            crossover_points.append((df.index[i], df["close"].iloc[i], df["RSI"].iloc[i], "bearish"))
    return crossover_points

# ========== Notification System ==========
def send_pushbullet_notification(title, body):
    api_key = st.secrets["pushbullet"]["api_key"]
    data_send = {"type": "note", "title": title, "body": body}
    resp = requests.post(
        'https://api.pushbullet.com/v2/pushes',
        json=data_send,
        headers={'Access-Token': api_key, 'Content-Type': 'application/json'}
    )
    return resp.status_code

def play_local_notification(title, body, repeat=3):
    for _ in range(repeat):
        if platform.system() == "Windows":
            from plyer import notification
            notification.notify(title=title, message=body, timeout=5)
            os.system('echo "\a"')  # Basic beep
        elif platform.system() == "Darwin":  # macOS
            os.system(f"osascript -e 'display notification \"{body}\" with title \"{title}\"'")
        elif platform.system() == "Linux":
            os.system(f'notify-send "{title}" "{body}"')
        time.sleep(1)

# ========== Streamlit UI ==========
st.set_page_config(page_title="EMA Crossover Monitor", layout="wide")
st.title("📈 EMA Crossover Monitor (CoinEx)")

# === Timezone Selection ===
timezones = {
    "Tehran": "Asia/Tehran",
    "UTC": "UTC",
    "London": "Europe/London",
    "New York": "America/New_York"
}
selected_tz_label = st.selectbox("🌍 Select your time zone:", options=list(timezones.keys()), index=0)
selected_timezone = pytz.timezone(timezones[selected_tz_label])

now_local = pd.Timestamp.now(tz=selected_timezone)
st.markdown(f"### 🕒 Current time ({selected_tz_label}): {now_local.strftime('%Y-%m-%d %H:%M:%S')}")

# === Notification Options ===
notif_type = st.radio("🔔 Notification type:", options=["Pushbullet", "Local"])
notif_repeat = st.slider("🔁 Local notification repeat count:", min_value=1, max_value=5, value=3)

# Refresh every N seconds
interval_minutes = st.number_input("⏲️ Refresh interval (minutes)", min_value=1, max_value=60, value=5)
st_autorefresh(interval=interval_minutes * 60 * 1000, key="refresh")

if st.button("🔔 Notif Check"):
    if notif_type == "Pushbullet":
        status = send_pushbullet_notification("🔔 Test Notification", "✅ This is a test push from your EMA monitor app.")
        st.success("Notification sent!" if status == 200 else f"Failed with status code: {status}")
    else:
        play_local_notification("🔔 Test Notification", "✅ This is a test push from your EMA monitor app.", notif_repeat)
        st.success("Local notification played!")

all_markets = list_available_markets()
selected_markets = st.multiselect("🪙 Select markets to monitor:", all_markets, default=["BTCUSDT", "ETHUSDT"])
ema_short = st.number_input("Short EMA period", min_value=1, max_value=50, value=7)
ema_long = st.number_input("Long EMA period", min_value=1, max_value=100, value=30)
color_short = st.color_picker("🎨 Short EMA Color", value="#1f77b4")
color_long = st.color_picker("🎨 Long EMA Color", value="#ff7f0e")
interval = st.selectbox("⏱️ Candle interval", options=["1min", "5min", "15min", "30min", "1hour", "4hour", "1day"], index=1)

# Auto-evaluate on every refresh
for symbol in selected_markets:
    st.subheader(f"{symbol}")
    try:
        df = get_kline_data(symbol, interval=interval)
        df = calculate_emas(df, ema_short, ema_long)

        volume_threshold = df["volume"].quantile(0.75)
        df["is_long"] = df["volume"] >= volume_threshold

        crossovers = detect_crossovers(df)

        match = re.match(r"(\\d+)?(min|hour|day)", interval)
        unit_value = int(match.group(1)) if match and match.group(1) else 1
        unit_type = match.group(2) if match else "min"
        unit_map = {"min": 1, "hour": 60, "day": 1440}
        candle_minutes = unit_value * unit_map.get(unit_type, 1)

        total_minutes = candle_minutes + interval_minutes * 3
        time_window = pd.Timestamp.now(tz=selected_timezone) - pd.Timedelta(minutes=total_minutes)

        for ts, price, rsi, ctype in crossovers:
            if ts > time_window:
                key = f"{symbol}_{ts}_{ctype}"
                notified_crossovers[key] = notified_crossovers.get(key, 0)

                if notified_crossovers[key] < 2:
                    message = f"{symbol} - {ctype.title()} EMA crossover at {price:.2f} (RSI: {rsi:.2f}) at {ts.strftime('%H:%M:%S')}"
                    if notif_type == "Pushbullet":
                        send_pushbullet_notification("📊 EMA Crossover Alert", message)
                    else:
                        play_local_notification("📊 EMA Crossover Alert", message, notif_repeat)
                    notified_crossovers[key] += 1

        fig = go.Figure()
        for i in range(len(df)):
            row = df.iloc[i]
            color = "green" if row["is_long"] else "red"
            fig.add_trace(go.Scatter(x=[row.name, row.name], y=[row["open"], row["close"]], mode="lines", line=dict(color=color, width=6), showlegend=False))
            fig.add_trace(go.Scatter(x=[row.name, row.name], y=[row["low"], row["high"]], mode="lines", line=dict(color=color, width=1), showlegend=False))

        fig.add_trace(go.Scatter(x=df.index, y=df["EMA_short"], line=dict(color=color_short, width=1), name=f"EMA {ema_short}"))
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA_long"], line=dict(color=color_long, width=1), name=f"EMA {ema_long}"))

        for ts, price, rsi, ctype in crossovers:
            fig.add_trace(go.Scatter(
                x=[ts],
                y=[price],
                mode="markers+text",
                marker=dict(size=10, color="lime" if ctype=="bullish" else "red"),
                text=["📈 Bullish" if ctype=="bullish" else "📉 Bearish"],
                textposition="top center",
                name="Crossover"
            ))

        fig.update_layout(title=f"{symbol} - EMA Crossover",
                          xaxis_title="Time",
                          yaxis_title="Price",
                          xaxis_rangeslider_visible=False,
                          template="plotly_dark",
                          height=500)

        st.plotly_chart(fig, use_container_width=True)

        if crossovers:
            st.info(f"⚡ {len(crossovers)} crossover(s) detected")
            for ts, price, rsi, ctype in crossovers[-5:]:
                key = f"{symbol}_{ts}_{ctype}"
                if notified_crossovers.get(key, 0) > 0:
                    st.markdown(f"<span style='color:orange;font-weight:bold'>🔔 {ts.strftime('%Y-%m-%d %H:%M:%S')} - {ctype.title()} crossover at {price:.2f} (RSI: {rsi:.2f})</span>", unsafe_allow_html=True)
                else:
                    st.write(f"{ts.strftime('%Y-%m-%d %H:%M:%S')} - {ctype.title()} crossover at {price:.2f} (RSI: {rsi:.2f})")

    except Exception as e:
        st.error(f"Failed to load {symbol}: {e}")

import os
import re
import json
import time
import platform
import requests
import pandas as pd
import pytz
import streamlit as st
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from streamlit_autorefresh import st_autorefresh

# =================== Persistence ===================
SETTINGS_FILE = "user_settings.json"

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_settings_from_state():
    settings = {
        "selected_markets": st.session_state.get("selected_markets", []),
        "ema_short": st.session_state.get("ema_short", 7),
        "ema_long": st.session_state.get("ema_long", 30),
        "interval": st.session_state.get("interval", "5min"),
        "timezone": st.session_state.get("selected_tz_label", "Tehran"),
        "notif_type": st.session_state.get("notif_type", "Local"),
        "notif_sound_repeat": st.session_state.get("notif_sound_repeat", 3),
        "max_alerts_per_cross": st.session_state.get("max_alerts_per_cross", 2),
        "refresh_minutes": st.session_state.get("interval_minutes", 5),
    }
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
        st.toast("💾 Settings saved", icon="💾")
    except Exception as e:
        st.warning(f"Couldn't save settings: {e}")

_saved = load_settings()

# =================== App state ===================
notified_crossovers = {}

# =================== CoinEx API ===================
@st.cache_data(show_spinner=False, ttl=600)
def list_available_markets():
    url = "https://api.coinex.com/v1/market/list"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.json()["data"]

def get_kline_data(symbol, interval="5min", limit=200, _tz=pytz.UTC) -> pd.DataFrame:
    url = "https://api.coinex.com/v1/market/kline"
    params = {"market": symbol, "type": interval, "limit": int(limit)}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()["data"]

    df = pd.DataFrame(
        data,
        columns=["timestamp", "open", "high", "low", "close", "volume", "amount"]
    )
    # time & types
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s").dt.tz_localize("UTC").dt.tz_convert(_tz)
    df.set_index("timestamp", inplace=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_index()

# =================== Indicators / Signals ===================
def calculate_emas(df, short_window, long_window):
    df["EMA_short"] = EMAIndicator(df["close"], window=short_window).ema_indicator()
    df["EMA_long"]  = EMAIndicator(df["close"], window=long_window).ema_indicator()
    df["RSI"]       = RSIIndicator(close=df["close"], window=14).rsi()
    return df

def detect_crossovers(df):
    out = []
    for i in range(1, len(df)):
        ps, pl = df["EMA_short"].iloc[i-1], df["EMA_long"].iloc[i-1]
        cs, cl = df["EMA_short"].iloc[i],   df["EMA_long"].iloc[i]
        if pd.notna(ps) and pd.notna(pl) and pd.notna(cs) and pd.notna(cl):
            if ps < pl and cs > cl:
                out.append((df.index[i], df["close"].iloc[i], df["RSI"].iloc[i], "bullish"))
            elif ps > pl and cs < cl:
                out.append((df.index[i], df["close"].iloc[i], df["RSI"].iloc[i], "bearish"))
    return out

# =================== Notifications ===================
def send_pushbullet_notification(title, body):
    api_key = (st.secrets.get("pushbullet") or {}).get("api_key")
    if not api_key:
        return 0, "Missing Pushbullet API key in .streamlit/secrets.toml"
    resp = requests.post(
        "https://api.pushbullet.com/v2/pushes",
        json={"type": "note", "title": title, "body": body},
        headers={"Access-Token": api_key, "Content-Type": "application/json"},
        timeout=15,
    )
    try:
        return resp.status_code, resp.json()
    except Exception:
        return resp.status_code, resp.text

def play_local_notification(title, body, repeat=3):
    for _ in range(repeat):
        if platform.system() == "Windows":
            try:
                from plyer import notification
                notification.notify(title=title, message=body, timeout=5)
            except Exception:
                pass
            os.system('echo "\a"')
        elif platform.system() == "Darwin":
            os.system(f"osascript -e 'display notification \"{body}\" with title \"{title}\"'")
        elif platform.system() == "Linux":
            os.system(f'notify-send "{title}" "{body}"')
        time.sleep(1)

# =================== UI ===================
st.set_page_config(page_title="EMA Crossover Monitor", layout="wide")
st.title("📋 EMA Crossover Monitor — Table View (CoinEx)")

# Timezone
timezones = {"Tehran":"Asia/Tehran","UTC":"UTC","London":"Europe/London","New York":"America/New_York"}
default_tz = _saved.get("timezone", "Tehran")
st.session_state.setdefault("selected_tz_label", default_tz)
selected_tz_label = st.selectbox(
    "🌍 Time zone:",
    options=list(timezones.keys()),
    index=list(timezones.keys()).index(st.session_state["selected_tz_label"]),
    key="selected_tz_label",
    on_change=save_settings_from_state
)
selected_timezone = pytz.timezone(timezones[selected_tz_label])
now_local = pd.Timestamp.now(tz=selected_timezone)
st.markdown(f"**🕒 Current time ({selected_tz_label})**: {now_local.strftime('%Y-%m-%d %H:%M:%S')}")

# Notification options
colA, colB, colC = st.columns([1,1,1])
with colA:
    notif_default = _saved.get("notif_type", "Local")
    notif_type = st.radio("🔔 Notification type:", ["Local", "Pushbullet"],
                          index=0 if notif_default=="Local" else 1,
                          key="notif_type", on_change=save_settings_from_state)
with colB:
    max_alerts_per_cross = st.slider("Max alerts per crossover", 1, 5,
                                     _saved.get("max_alerts_per_cross", 2),
                                     key="max_alerts_per_cross", on_change=save_settings_from_state)
with colC:
    notif_sound_repeat = st.slider("Local sound repeats", 1, 5,
                                   _saved.get("notif_sound_repeat", 3),
                                   key="notif_sound_repeat", on_change=save_settings_from_state)

# Auto-refresh
interval_minutes = st.number_input("⏲️ Auto-refresh every (minutes)", 1, 60,
                                   _saved.get("refresh_minutes", 5),
                                   key="interval_minutes", on_change=save_settings_from_state)
st_autorefresh(interval=interval_minutes * 60 * 1000, key="refresh")

# Test notif
if st.button("🔔 Notif Check"):
    if notif_type == "Pushbullet":
        st.info("Pushbullet is disabled for now (only local notifs).")
    else:
        play_local_notification("🔔 Test Notification", "✅ Test local notification", notif_sound_repeat)
        st.success("Local notification played!")

# Inputs
all_markets = list_available_markets()
default_markets = [m for m in _saved.get("selected_markets", ["BTCUSDT","ETHUSDT"]) if m in all_markets]
selected_markets = st.multiselect("🪙 Markets to monitor", all_markets, default=default_markets,
                                  key="selected_markets", on_change=save_settings_from_state)
ema_short = st.number_input("Short EMA", 1, 50, _saved.get("ema_short", 7),
                            key="ema_short", on_change=save_settings_from_state)
ema_long  = st.number_input("Long EMA", 1, 200, _saved.get("ema_long", 30),
                            key="ema_long", on_change=save_settings_from_state)
interval  = st.selectbox("📏 Candle interval",
                         ["1min","5min","15min","30min","1hour","4hour","1day"],
                         index=["1min","5min","15min","30min","1hour","4hour","1day"].index(
                             _saved.get("interval", "5min")),
                         key="interval", on_change=save_settings_from_state)

# =================== Table (no charts) ===================
records = []

pattern = r"(\d+)?(min|hour|day)"
m = re.match(pattern, interval)
unit_value = int(m.group(1)) if m and m.group(1) else 1
unit_type  = m.group(2) if m else "min"
unit_map   = {"min":1, "hour":60, "day":1440}
candle_minutes = unit_value * unit_map.get(unit_type, 1)
window_minutes = candle_minutes + interval_minutes*3
recent_window  = pd.Timestamp.now(tz=selected_timezone) - pd.Timedelta(minutes=window_minutes)

for symbol in selected_markets:
    try:
        df = get_kline_data(symbol, interval=interval, limit=200, _tz=selected_timezone)
        df = calculate_emas(df, ema_short, ema_long)
        cross = detect_crossovers(df)

        last_close = float(df["close"].iloc[-1])
        ema_s = float(df["EMA_short"].iloc[-1])
        ema_l = float(df["EMA_long"].iloc[-1])
        rsi    = float(df["RSI"].iloc[-1])

        last_ts = None; last_type = "-"; alerts_sent = 0; status_flag = "OK"
        if cross:
            last_ts, last_price, last_rsi, last_type = cross[-1]
            key = f"{symbol}_{last_ts}_{last_type}"
            alerts_sent = notified_crossovers.get(key, 0)
            if last_ts > recent_window and alerts_sent < max_alerts_per_cross:
                status_flag = "ALERT"
                msg = f"{symbol} - {last_type.title()} EMA crossover at {last_price:.6f} (RSI: {last_rsi:.2f}) at {last_ts.strftime('%H:%M:%S')}"
                if notif_type == "Pushbullet":
                    st.info("Pushbullet is disabled for now (only local notifs).")
                else:
                    play_local_notification("📊 EMA Crossover Alert", msg, notif_sound_repeat)
                notified_crossovers[key] = alerts_sent + 1
                alerts_sent += 1

        records.append({
            "Symbol": symbol,
            "Last Price": last_close,
            f"EMA {ema_short}": ema_s,
            f"EMA {ema_long}": ema_l,
            "RSI": rsi,
            "Last Crossover": last_ts.strftime('%Y-%m-%d %H:%M') if last_ts else "-",
            "Type": last_type.title() if last_type != "-" else "-",
            "Alerts Sent": alerts_sent,
            "Status": status_flag
        })
    except Exception as e:
        st.error(f"Failed to load {symbol}: {e}")

summary_df = pd.DataFrame(records)

if not summary_df.empty:
    def highlight_alert(row):
        if row.get("Status") == "ALERT":
            if row.get("Type") == "Bullish":
                return ["background-color: #d4edda" for _ in row]  # green
            if row.get("Type") == "Bearish":
                return ["background-color: #f8d7da" for _ in row]  # red
        return ["" for _ in row]

    st.dataframe(
        summary_df.style.apply(highlight_alert, axis=1).format({
            "Last Price":"{:.6f}",
            f"EMA {ema_short}":"{:.6f}",
            f"EMA {ema_long}":"{:.6f}",
            "RSI":"{:.2f}"
        }),
        use_container_width=True,
        height=420
    )
else:
    st.info("No markets selected.")

# test with new things
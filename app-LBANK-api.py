import os
import re
import json
import time
import platform
import pathlib
import requests
import pandas as pd
import pytz
import streamlit as st
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from streamlit_autorefresh import st_autorefresh
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import AverageTrueRange
import numpy as np


# =================== Persistence ===================
SETTINGS_FILE = "user_settings.json"
LBANK_BASE = "https://api.lbkex.com"

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

SEEN_FILE = "seen_crossovers.json"

def _load_seen_from_disk():
    try:
        if os.path.exists(SEEN_FILE):
            with open(SEEN_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def _save_seen_to_disk(d):
    try:
        with open(SEEN_FILE, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

# init once
if "seen_crossovers" not in st.session_state:
    st.session_state["seen_crossovers"] = _load_seen_from_disk()

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

# --- add full indicator set to a kline DataFrame (OHLCV, datetime index) ---
def add_full_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # EMAs for strategy score (9/21) – independent from your table’s EMA_short/EMA_long
    df["ema9"]  = EMAIndicator(df["close"], window=9).ema_indicator()
    df["ema21"] = EMAIndicator(df["close"], window=21).ema_indicator()

    # RSI(14)
    df["rsi"] = RSIIndicator(close=df["close"], window=14).rsi()

    # MACD(12, 26, 9)
    macd = MACD(close=df["close"], window_fast=12, window_slow=26, window_sign=9)
    df["macd"]        = macd.macd()
    df["macd_signal"] = macd.macd_signal()

    # Volume avg (5)
    df["avg_volume"] = df["volume"].rolling(5, min_periods=1).mean()

    # ATR(14) + its rolling average (14)
    atr = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=14)
    df["atr"]     = atr.average_true_range()
    df["atr_avg"] = df["atr"].rolling(14, min_periods=1).mean()

    # Stochastic %K / %D
    stoch = StochasticOscillator(high=df["high"], low=df["low"], close=df["close"],
                                 window=14, smooth_window=3)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    # Simple candle classifier (engulfing/pinbar/none)
    df["candle_type"] = _detect_candle_type(df)

    return df


def _detect_candle_type(df: pd.DataFrame) -> pd.Series:
    types = []
    prev_o = df["open"].shift(1)
    prev_c = df["close"].shift(1)

    for o, h, l, c, po, pc in zip(df["open"], df["high"], df["low"], df["close"], prev_o, prev_c):
        t = "none"
        if pd.notna(po) and pd.notna(pc):
            # bullish/bearish engulfing
            if (c > o) and (pc < po) and (c >= po) and (o <= pc):
                t = "engulfing_bull"
            elif (c < o) and (pc > po) and (c <= po) and (o >= pc):
                t = "engulfing_bear"
        # pin bar (very small body vs total range, long tail)
        body = abs(c - o)
        rng  = h - l
        if rng > 0:
            upper = h - max(c, o)
            lower = min(c, o) - l
            if body / rng < 0.3 and (upper > 2 * body or lower > 2 * body):
                t = "pinbar_bull" if lower > upper else "pinbar_bear"
        types.append(t)

    return pd.Series(types, index=df.index)

def _pullback_state_from_parts(trend_slope, broke, retested, confirmed):
    if abs(trend_slope) < 1e-12:
        return "No trend"
    if not broke:
        return "Trend, no break"
    if broke and not retested:
        return "Break, waiting retest"
    if broke and retested and not confirmed:
        return "Retest, waiting confirm"
    return "—"  # should not happen, covered by main positive labels

def _detect_pullback_once(df: pd.DataFrame, lookback: int, tol_mult: float):
    if len(df) < lookback + 3:
        return {"label": "Too short", "price": None, "score": 0, "parts": (0,0,0), "slope": 0.0}

    seg = df.iloc[-(lookback+2):].copy()
    closes = seg["close"].to_numpy()
    x = np.arange(len(closes), dtype=float)

    a, b = np.polyfit(x, closes, 1)             # trendline y = a x + b
    line = a * x + b

    i_prev, i_break, i_conf = len(seg)-3, len(seg)-2, len(seg)-1
    prev_c, break_c, conf_c = closes[i_prev], closes[i_break], closes[i_conf]
    line_prev, line_break, line_conf = line[i_prev], line[i_break], line[i_conf]

    # tolerance (ATR-based if possible, else 0.2%)
    try:
        atr_last = float(seg["atr"].iloc[-1])
        tol = atr_last * tol_mult if np.isfinite(atr_last) and atr_last > 0 else float(closes[-1]) * 0.002
    except Exception:
        tol = float(closes[-1]) * 0.002

    # parts
    if a > 0:
        broke     = (break_c < line_break) and (prev_c >= line_prev)
        retested  = (abs(seg["open"].iloc[i_conf] - line_conf) <= tol) or (seg["high"].iloc[i_conf] >= line_conf - tol)
        confirmed = conf_c <= break_c
        if broke and retested and confirmed:
            return {"label": "Bearish PB", "price": conf_c, "score": 3, "parts": (1,1,1), "slope": a}
        return {"label": _pullback_state_from_parts(a, broke, retested, confirmed),
                "price": None, "score": int(broke)+int(retested)+int(confirmed), "parts": (broke, retested, confirmed), "slope": a}

    elif a < 0:
        broke     = (break_c > line_break) and (prev_c <= line_prev)
        retested  = (abs(seg["open"].iloc[i_conf] - line_conf) <= tol) or (seg["low"].iloc[i_conf] <= line_conf + tol)
        confirmed = conf_c >= break_c
        if broke and retested and confirmed:
            return {"label": "Bullish PB", "price": conf_c, "score": 3, "parts": (1,1,1), "slope": a}
        return {"label": _pullback_state_from_parts(a, broke, retested, confirmed),
                "price": None, "score": int(broke)+int(retested)+int(confirmed), "parts": (broke, retested, confirmed), "slope": a}

    else:
        return {"label": "No trend", "price": None, "score": 0, "parts": (0,0,0), "slope": a}

def detect_trendline_pullback_any(df: pd.DataFrame,
                                  lookbacks=(20, 35, 50),
                                  tol_mult: float = 0.25) -> dict:
    """
    Try several lookbacks; return the first confirmed PB if any.
    Otherwise return the 'best partial' state so the column is never '-'.
    """
    best = {"label": "No trend", "price": None, "score": -1, "parts": (0,0,0), "slope": 0.0}
    for lb in lookbacks:
        res = _detect_pullback_once(df, lb, tol_mult)
        # immediate return on a full signal
        if res["label"] in ("Bullish PB", "Bearish PB"):
            return res
        # keep the best partial (highest parts score)
        if res["score"] > best["score"]:
            best = res
    return best




# --- pick the higher timeframe for a given interval ---
def _higher_tf(interval: str) -> str:
    mapping = {
        "1min": "5min", "5min": "15min", "15min": "1hour",
        "30min": "4hour", "1hour": "4hour", "4hour": "1day", "1day": "1day"
    }
    return mapping.get(interval, "15min")


def get_higher_tf_trend(symbol: str, interval: str, tz) -> str:
    hi = _higher_tf(interval)
    df_h = get_kline_data(symbol, interval=hi, limit=200, _tz=tz)
    df_h["ema9"]  = EMAIndicator(df_h["close"], 9).ema_indicator()
    df_h["ema21"] = EMAIndicator(df_h["close"], 21).ema_indicator()
    last = df_h.iloc[-1]
    return "bullish" if last["ema9"] > last["ema21"] else "bearish"


# --- final scoring (uses all indicators if present) ---
def evaluate_signal_token(t: dict) -> tuple[int, str]:
    score = 0

    if t.get("ema_fast") is not None and t.get("ema_slow") is not None and t["ema_fast"] > t["ema_slow"]:
        score += 2

    # 2) RSI in sweet spot
    if t.get("rsi") is not None and 50 <= t["rsi"] <= 70:
        score += 1

    # 3) MACD bullish (+ histogram rising via macd > signal)
    if t.get("macd") is not None and t.get("macd_signal") is not None and t["macd"] > 0 and t["macd"] > t["macd_signal"]:
        score += 2

    # 4) Volume above recent average
    if t.get("volume") is not None and t.get("avg_volume") is not None and t["volume"] > t["avg_volume"]:
        score += 1

    # 5) ATR in a “reasonable” range vs its average (0.8–1.5×)
    if t.get("atr") and t.get("atr_avg") and (0.8 * t["atr_avg"] <= t["atr"] <= 1.5 * t["atr_avg"]):
        score += 1

    # 6) Stochastic bullish cross in oversold
    if t.get("stoch_k") is not None and t.get("stoch_d") is not None and t["stoch_k"] > t["stoch_d"] and t["stoch_k"] < 30:
        score += 1

    # 7) Candle confirmation
    if t.get("candle_type") in ("engulfing_bull", "pinbar_bull"):
        score += 2

    # 8) Higher-TF trend alignment
    if t.get("highertf_trend") == "bullish":
        score += 1

    strength = "strong ✅" if score >= 8 else ("mid ⚠️" if score >= 5 else "weak ❌")
    return score, strength

# =================== App state ===================
if "seen_crossovers" not in st.session_state:
    st.session_state["seen_crossovers"] = {}  # key -> True once notified

# =================== LBANK API ===================


def _to_ui_sym(lbank_sym: str) -> str:
    # "btc_usdt" -> "BTCUSDT" (your app’s current display style)
    return lbank_sym.replace("_", "").upper()

def _to_lbank_sym(ui_sym: str) -> str:
    return st.session_state.get("lbank_symbol_map", {}).get(ui_sym, ui_sym.lower())


#LBANK_BASE = "https://api.lbkex.com"  # or https://www.lbkex.net

@st.cache_data(show_spinner=False, ttl=600)
def list_available_markets():
    url = f"{LBANK_BASE}/v2/currencyPairs.do"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    js = r.json()

    # LBank wraps data; validate
    if not isinstance(js, dict) or not js.get("result", False):
        raise RuntimeError(f"LBank pairs error: {js}")

    raw_pairs = js.get("data", [])  # list like ["btc_usdt", ...]
    ui_symbols, sym_map = [], {}

    for p in raw_pairs:
        if isinstance(p, str) and "_" in p:
            ui = p.replace("_", "").upper()  # "btc_usdt" -> "BTCUSDT"
            sym_map[ui] = p                  # UI -> API map
            ui_symbols.append(ui)

    st.session_state["lbank_symbol_map"] = sym_map
    return sorted(ui_symbols)



#LBANK_BASE = "https://api.lbkex.com"



# App → LBank interval names
LBANK_INTERVAL = {
    "1min":  "minute1",
    "5min":  "minute5",
    "15min": "minute15",
    "30min": "minute30",
    "1hour": "hour1",
    "4hour": "hour4",
    "1day":  "day1",
}

def get_kline_data(symbol, interval="5min", limit=200, _tz=pytz.UTC) -> pd.DataFrame:
    # UI → API symbol (e.g. "A2ZUSDT" -> "a2z_usdt")
    api_symbol = st.session_state.get("lbank_symbol_map", {}).get(symbol, symbol).lower()

    # LBank accepts different 'type' keywords depending on cluster/version.
    _pref = {
        "1min":  ["minute1", "1min",  "kline_1min"],
        "5min":  ["minute5", "5min",  "kline_5min"],
        "15min": ["minute15","15min", "kline_15min"],
        "30min": ["minute30","30min", "kline_30min"],
        "1hour": ["hour1",  "1hour", "kline_1hour"],
        "4hour": ["hour4",  "4hour", "kline_4hour"],
        "1day":  ["day1",   "1day",  "kline_1day"],
    }.get(interval, ["minute5"])

    # Some clusters serve klines on api.lbank.info (while pairs come from api.lbkex.com)
    # add all known hosts (some symbols only work on .net)
    bases = [
        "https://api.lbank.info",
        "https://api.lbkex.com",
        "https://api.lbkex.net",
        "https://www.lbkex.net",
    ]

    # LBank rejects large sizes on some clusters -> keep <= 100
    req_size = min(int(limit), 100)

    j = None
    for base in bases:
        for t in _pref:  # your list of candidate interval types
            try:
                r = requests.get(
                    f"{base}/v2/kline.do",
                    params={"symbol": api_symbol, "size": req_size, "type": t},
                    timeout=20,
                )
                j = r.json()
            except Exception:
                j = None
            if j and j.get("result") and j.get("data"):
                break
        if j and j.get("result") and j.get("data"):
            break

    if not j or not j.get("result") or not j.get("data"):
        raise ValueError(f"LBank kline empty/err for {api_symbol} ({interval}): {j}")

    # LBank returns arrays: [timestamp, open, close, high, low, volume]
    df = pd.DataFrame(j["data"], columns=["ts","open","close","high","low","volume"])

    # seconds vs ms
    ts = pd.to_datetime(df["ts"], unit="s", errors="coerce")
    if df["ts"].astype("int64").max() > 10**12:
        ts = pd.to_datetime(df["ts"], unit="ms", errors="coerce")

    df["timestamp"] = ts.dt.tz_localize("UTC").dt.tz_convert(_tz)
    df.set_index("timestamp", inplace=True)

    for c in ["open","close","high","low","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # reorder to open/high/low/close
    df = df[["open","high","low","close","volume"]].sort_index()
    return df


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
    base = pathlib.Path(__file__).parent
    mp3_path = base / "notifSound.mp3"  # your MP3 file in the same folder
    for _ in range(repeat):
        # Toast/notification
        if platform.system() == "Windows":
            try:
                from plyer import notification
                notification.notify(title=title, message=body, timeout=5)
            except Exception:
                pass
        elif platform.system() == "Darwin":  # macOS
            os.system(f"osascript -e 'display notification \"{body}\" with title \"{title}\"'")
        elif platform.system() == "Linux":
            os.system(f'notify-send "{title}" "{body}"')
        # Sound
        if mp3_path.exists():
            try:
                import pygame
                pygame.mixer.init()
                pygame.mixer.music.load(str(mp3_path))
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
            except Exception as e:
                print(f"Error playing MP3: {e}")
        time.sleep(1)


# =================== UI ===================
st.set_page_config(page_title="EMA Crossover Monitor", layout="wide")
st.title("📋 EMA Crossover Monitor — Table View (LBank)")

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
# --- Pullback tuning knobs ---
colPB1, colPB2 = st.columns(2)
with colPB1:
    pb_lookback = st.slider("Pullback lookback", 15, 80, 30, key="pb_lookback")
with colPB2:
    pb_tol_mult = st.slider("Pullback tol (ATR×)", 0.05, 0.50, 0.25, step=0.05, key="pb_tol_mult")

selected_timezone = pytz.timezone(timezones[selected_tz_label])
now_local = pd.Timestamp.now(tz=selected_timezone)
st.markdown(f"**🕒 Current time ({selected_tz_label})**: {now_local.strftime('%Y-%m-%d %H:%M:%S')}")

# Notification options
colA, colB, colC = st.columns([1,1,1])
with colA:
    notif_default = _saved.get("notif_type", "Local")
    notif_type = st.radio(
        "🔔 Notification type:", ["Local", "Pushbullet"],
        index=0 if notif_default == "Local" else 1,
        key="notif_type", on_change=save_settings_from_state
    )
with colB:
    max_alerts_per_cross = st.slider(
        "Max alerts per crossover", 1, 5, _saved.get("max_alerts_per_cross", 2),
        key="max_alerts_per_cross", on_change=save_settings_from_state
    )
with colC:
    notif_sound_repeat = st.slider(
        "Local sound repeats", 1, 5, _saved.get("notif_sound_repeat", 3),
        key="notif_sound_repeat", on_change=save_settings_from_state
    )

# Auto-refresh
interval_minutes = st.number_input(
    "⏲️ Auto-refresh every (minutes)", 1, 60, _saved.get("refresh_minutes", 5),
    key="interval_minutes", on_change=save_settings_from_state
)
st_autorefresh(interval=interval_minutes * 60 * 1000, key="refresh")

# Test notif
if st.button("🔔 Notif Check"):
    if notif_type == "Pushbullet":
        st.info("Pushbullet is disabled for now (only local notifs).")
    else:
        play_local_notification("🔔 Test Notification", "✅ Test local notification", notif_sound_repeat)
        st.success("Local notification played!")

# ---- Markets list (no pre-scan) ----
if "cache_cleared" not in st.session_state:
    st.cache_data.clear()
    st.session_state["cache_cleared"] = True

all_markets = list_available_markets()

# Remove any previously-saved symbols that aren't on LBank anymore
bad_saved = [s for s in _saved.get("selected_markets", []) if s not in all_markets]
if bad_saved:
    st.warning(f"Not on LBank (removed): {', '.join(bad_saved)}")

default_markets = [m for m in _saved.get("selected_markets", ["BTCUSDT","ETHUSDT"]) if m in all_markets]

selected_markets = st.multiselect(
    "🪙 Markets to monitor",
    options=all_markets,
    default=default_markets,
    key="selected_markets",
    on_change=save_settings_from_state
)

# ---- Validate ONLY the chosen symbols (fast check) ----
@st.cache_data(ttl=180, show_spinner=False)
def has_kline(symbol: str, interval: str) -> bool:
    try:
        _ = get_kline_data(symbol, interval=interval, limit=5, _tz=pytz.UTC)
        return True
    except Exception:
        return False

ok_syms, missing_syms = [], []
for s in selected_markets:
    (ok_syms if has_kline(s, st.session_state.get("interval","5min")) else missing_syms).append(s)

if missing_syms:
    st.warning(f"No kline for this interval: {', '.join(missing_syms)}")
selected_markets = ok_syms  # use only workable ones below

col1, _ = st.columns([1,5])
with col1:
    if st.button("↻ Refresh markets"):
        list_available_markets.clear()  # clears only this cached function


ema_short = st.number_input("Short EMA", 1, 50, _saved.get("ema_short", 7),
                            key="ema_short", on_change=save_settings_from_state)
ema_long  = st.number_input("Long EMA", 1, 200, _saved.get("ema_long", 30),
                            key="ema_long", on_change=save_settings_from_state)
interval  = st.selectbox(
    "📏 Candle interval",
    ["1min","5min","15min","30min","1hour","4hour","1day"],
    index=["1min","5min","15min","30min","1hour","4hour","1day"].index(_saved.get("interval", "5min")),
    key="interval", on_change=save_settings_from_state
)

# =================== Table (no charts) ===================
records = []

pattern = r"(\d+)?(min|hour|day)"
m = re.match(pattern, interval)
unit_value = int(m.group(1)) if m and m.group(1) else 1
unit_type  = m.group(2) if m else "min"
unit_map   = {"min":1, "hour":60, "day":1440}
candle_minutes = unit_value * unit_map.get(unit_type, 1)
window_minutes = candle_minutes + interval_minutes * 3
recent_window  = pd.Timestamp.now(tz=selected_timezone) - pd.Timedelta(minutes=window_minutes)

for symbol in selected_markets:
    try:
        df = get_kline_data(symbol, interval=interval, limit=200, _tz=selected_timezone)
        df = add_full_indicators(df)                                  # << add all indicators
        last = df.iloc[-1]

        # after fetching df
        # df = add_full_indicators(df)                 # keeps macd/rsi/stoch/... in df
        df = calculate_emas(df, ema_short, ema_long) # adds EMA_short / EMA_long
        pb = detect_trendline_pullback_any(df, lookbacks=(pb_lookback, max(15, pb_lookback-15), min(80, pb_lookback+15)),
                                        tol_mult=pb_tol_mult)
        pullback_label = pb["label"]



        # latest values
        ema_s = float(df["EMA_short"].iloc[-1])
        ema_l = float(df["EMA_long"].iloc[-1])

        token_dict = {
            "symbol": symbol,
            "ema_fast": ema_s,                         # ← use the user-selected EMAs
            "ema_slow": ema_l,                         # ← use the user-selected EMAs
            "rsi":        float(df["rsi"].iloc[-1]),
            "macd":       float(df["macd"].iloc[-1]),
            "macd_signal":float(df["macd_signal"].iloc[-1]),
            "volume":     float(df["volume"].iloc[-1]),
            "avg_volume": float(df["avg_volume"].iloc[-1]),
            "atr":        float(df["atr"].iloc[-1]),
            "atr_avg":    float(df["atr_avg"].iloc[-1]),
            "stoch_k":    float(df["stoch_k"].iloc[-1]),
            "stoch_d":    float(df["stoch_d"].iloc[-1]),
            "candle_type":      df["candle_type"].iloc[-1],
            "highertf_trend":   get_higher_tf_trend(symbol, interval, selected_timezone),
        }

        score, strength = evaluate_signal_token(token_dict)  # ← now uses selected EMAs
        df = calculate_emas(df, ema_short, ema_long)
        cross = detect_crossovers(df)

        last_close = float(df["close"].iloc[-1])
        # ema_s = float(df["EMA_short"].iloc[-1])
        # ema_l = float(df["EMA_long"].iloc[-1])
        rsi   = float(df["RSI"].iloc[-1])

        last_ts = None
        last_type = "-"
        alerts_sent = 0
        status_flag = "OK"
        key = None

        if cross:
            last_ts, last_price, last_rsi, last_type = cross[-1]
            # stable key for this exact crossover
            key = f"{symbol}|{last_ts.isoformat()}|{last_type}"
            seen_before = st.session_state["seen_crossovers"].get(key, False)

            if last_ts > recent_window:
                status_flag = "ALERT"
                if not seen_before:
                    msg = (
                        f"{symbol} - {last_type.title()} EMA crossover at "
                        f"{last_price:.6f} (RSI: {last_rsi:.2f}) at {last_ts.strftime('%H:%M:%S')}"
                    )
                    if notif_type == "Pushbullet":
                        st.info("Pushbullet is disabled for now (only local notifs).")
                    else:
                        play_local_notification("📊 EMA Crossover Alert", msg, notif_sound_repeat)
                    # mark as seen so future reruns won't notify again
                    st.session_state["seen_crossovers"][key] = True
                    # persist to disk
                    _save_seen_to_disk(st.session_state["seen_crossovers"])



        # compute percent change since last crossover (if any)
        cross_price = None
        change_pct = None
        if cross:
            cross_price = float(last_price)
            if cross_price not in (None, 0):
                change_pct = (last_close - cross_price) / cross_price * 100.0
        records.append({
            "Symbol": symbol,
            "Last Price": last_close,
            f"EMA {ema_short}": ema_s,
            f"EMA {ema_long}": ema_l,
            "RSI": rsi,
            "Last Crossover": last_ts.strftime('%Y-%m-%d %H:%M') if last_ts else "-",
            "Type": last_type.title() if last_type != "-" else "-",
            "Change Since Cross %": change_pct,

            # NEW columns
            "Score": score,
            "Signal Strength": strength,

            # backend fields
            "Alerts Sent": alerts_sent,
            "Status": status_flag,
            "Seen Before": "Yes" if (key and st.session_state["seen_crossovers"].get(key, False)) else "No",
            "Pullback": pullback_label,

        })
    except Exception as e:
        st.error(f"Failed to load {symbol}: {e}")
        api_symbol_dbg = st.session_state.get("lbank_symbol_map", {}).get(symbol, symbol).lower()
        st.caption(f"Debug: symbol='{symbol}' → '{api_symbol_dbg}', interval='{interval}'")


summary_df = pd.DataFrame(records)

if summary_df.empty:
    st.info("No markets selected.")
else:
    # series we can reference by row.index in the styler
    _status_s = summary_df["Status"].copy()
    _type_s   = summary_df["Type"].copy()

    # def highlight_alert(row):
    #     # use the outer series with row.name
    #     if _status_s.loc[row.name] == "ALERT":
    #         if _type_s.loc[row.name] == "Bullish":
    #             return ["background-color: #d4edda" for _ in row]  # green
    #         if _type_s.loc[row.name] == "Bearish":
    #             return ["background-color: #f8d7da" for _ in row]  # red
    #     return ["" for _ in row]


    def color_last_price_by_change(row):
        v = row.get("Change Since Cross %")
        if pd.isna(v): return [""]
        if v > 0:  return ["color: #2ecc71; font-weight: 600;"]
        if v < 0:  return ["color: #e74c3c; font-weight: 600;"]
        return [""]
    
    # cols_to_show = [
    #     "Symbol", "Last Price", f"EMA {ema_short}", f"EMA {ema_long}",
    #     "RSI", "Last Crossover", "Type", "Change Since Cross %",
    #     "Score", "Signal Strength",
    # ]

    # display_df = summary_df[cols_to_show]  # no Status here

    cols_to_show = [
    "Symbol", "Last Price", f"EMA {ema_short}", f"EMA {ema_long}",
    "RSI", "Last Crossover", "Type", "Change Since Cross %",
    "Score", "Signal Strength", "Pullback",           # <-- NEW
    ]

    display_df = summary_df[cols_to_show]

    def highlight_alert(row):
        if _status_s.loc[row.name] == "ALERT":
            if _type_s.loc[row.name] == "Bullish":
                return ["background-color: #d4edda"] * len(display_df.columns)
            if _type_s.loc[row.name] == "Bearish":
                return ["background-color: #f8d7da"] * len(display_df.columns)
        return [""] * len(display_df.columns)

    def color_type_cell(v):
        if v == "Bullish": return "color:#2ecc71;font-weight:600;"
        if v == "Bearish": return "color:#e74c3c;font-weight:600;"
        return ""

    def color_change_cell(v):
        if pd.isna(v): return ""
        if v > 0:  return "color:#2ecc71;font-weight:600;"
        if v < 0:  return "color:#e74c3c;font-weight:600;"
        return ""

    # <<< KEY FIX: color Last Price based on the row's % change >>>
    # index of Last Price column
    _lp_idx = display_df.columns.get_loc("Last Price")

    def color_last_price_row(row):
        styles = [""] * len(display_df.columns)
        v = row.get("Change Since Cross %")
        if not pd.isna(v):
            if v > 0:
                styles[_lp_idx] = "color:#2ecc71;font-weight:700;font-size:24px;"
            elif v < 0:
                styles[_lp_idx] = "color:#e74c3c;font-weight:700;font-size:24px;"
        return styles


    def color_pullback_cell(v):
        if v == "Bullish PB": return "color:#2ecc71;font-weight:700;"
        if v == "Bearish PB": return "color:#e74c3c;font-weight:700;"
        if v in ("Break, waiting retest", "Retest, waiting confirm"): return "color:#f1c40f;font-weight:600;"
        if v in ("Trend, no break", "No trend", "Too short"): return "color:#bdc3c7;"
        return ""

    
    styled = (
        display_df
        .style
        .apply(highlight_alert, axis=1)
        .apply(color_last_price_row, axis=1)
        .map(color_type_cell, subset=["Type"])                      # <- was applymap
        .map(color_change_cell, subset=["Change Since Cross %"])    # <- was applymap
        .map(color_pullback_cell, subset=["Pullback"])              # <- was applymap
        .format({
            "Last Price": "{:.6f}",
            f"EMA {ema_short}": "{:.6f}",
            f"EMA {ema_long}": "{:.6f}",
            "RSI": "{:.2f}",
            "Change Since Cross %": "{:+.2f}%",
        })
    )


    st.dataframe(styled, use_container_width=True, height=420)


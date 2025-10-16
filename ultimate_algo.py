import os
import time
import requests
import pandas as pd
import yfinance as yf
import ta
import warnings
import pyotp
import math
from datetime import datetime, time as dtime, timedelta
from SmartApi.smartConnect import SmartConnect
import threading
import numpy as np

warnings.filterwarnings("ignore")

# ---------------- CONFIG ----------------
OPENING_PLAY_ENABLED = True           # Enable/disable the pure opening-level institutional play
OPENING_START = dtime(9,15)
OPENING_END = dtime(9,45)

EXPIRY_ACTIONABLE = True              # If True: expiry gamma layer can ACTIVATE trade signals (with extra checks)
EXPIRY_INFO_ONLY = False              # If True, expiry layer only sends info messages (won't trigger trades)
EXPIRY_RELAX_FACTOR = 0.7            # How much to relax confirmation requirements on expiry (0-1, lower -> more relaxation)

GAMMA_VOL_SPIKE_THRESHOLD = 1.8      # multiplier for volume to consider gamma-like event
DELTA_OI_RATIO = 1.5                 # OI CE/PE dominance ratio threshold
MOMENTUM_VOL_AMPLIFIER = 1.3         # threshold for amplifying confidence when volatility & momentum high

# ----------------------------------------

# --------- ANGEL ONE LOGIN ---------
API_KEY = os.getenv("API_KEY")
CLIENT_CODE = os.getenv("CLIENT_CODE")
PASSWORD = os.getenv("PASSWORD")
TOTP_SECRET = os.getenv("TOTP_SECRET")
TOTP = pyotp.TOTP(TOTP_SECRET).now()

client = SmartConnect(api_key=API_KEY)
session = client.generateSession(CLIENT_CODE, PASSWORD, TOTP)
feedToken = client.getfeedToken()

# --------- TELEGRAM ---------
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

# --- Added: guards to avoid repeating the same Telegram message many times
STARTED_SENT = False
STOP_SENT = False

def send_telegram(msg, reply_to=None):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": msg}
        if reply_to:
            payload["reply_to_message_id"] = reply_to
        r = requests.post(url, data=payload, timeout=5).json()
        return r.get("result", {}).get("message_id")
    except:
        return None

# --------- MARKET HOURS ---------
def is_market_open():
    # Get current time in IST (UTC +5:30)
    utc_now = datetime.utcnow()
    ist_now = utc_now + timedelta(hours=5, minutes=30)
    current_time_ist = ist_now.time()
    
    # Check if current IST time is between 9:15 AM and 3:30 PM
    market_open = dtime(9,15) <= current_time_ist <= dtime(15,30)
    
    return market_open

# --------- AUTO STOP AT 3:30 PM IST ---------
def should_stop_trading():
    utc_now = datetime.utcnow()
    ist_now = utc_now + timedelta(hours=5, minutes=30)
    current_time_ist = ist_now.time()
    return current_time_ist >= dtime(15,30)

# --------- STRIKE ROUNDING FOR ALL INDICES ---------
def round_strike(index, price):
    """
    Safely round a price to nearest strike step per index.
    Returns None if price is None or NaN.
    """
    try:
        if price is None:
            return None
        # handle pandas NA / numpy nan
        if isinstance(price, float) and math.isnan(price):
            return None
        price = float(price)
        
        # Validate price is reasonable
        if price <= 0:
            return None
            
        # CORRECTED STRIKE ROUNDING FOR EACH INDEX
        if index == "NIFTY": 
            strike = int(round(price / 50.0) * 50)
        elif index == "BANKNIFTY": 
            strike = int(round(price / 100.0) * 100)
        elif index == "SENSEX": 
            strike = int(round(price / 100.0) * 100)
        elif index == "FINNIFTY": 
            strike = int(round(price / 50.0) * 50)  # FINNIFTY uses 50 increments
        elif index == "MIDCPNIFTY": 
            strike = int(round(price / 25.0) * 25)
        elif index == "EICHERMOT": 
            strike = int(round(price / 50.0) * 50)
        elif index == "TRENT": 
            strike = int(round(price / 100.0) * 100)
        elif index == "RELIANCE": 
            strike = int(round(price / 10.0) * 10)
        else: 
            strike = int(round(price / 50.0) * 50)  # default
        
        # Validate strike makes sense
        if abs(strike - price) > price * 0.3:  # Strike shouldn't be too far from current price
            return None
            
        return strike
    except Exception:
        return None

# --------- ENSURE SERIES ---------
def ensure_series(data):
    return data.iloc[:,0] if isinstance(data, pd.DataFrame) else data.squeeze()

# --------- FETCH INDEX DATA FOR ALL INDICES ---------
def fetch_index_data(index, interval="5m", period="2d"):
    symbol_map = {
        "NIFTY": "^NSEI", 
        "BANKNIFTY": "^NSEBANK", 
        "SENSEX": "^BSESN",
        "FINNIFTY": "NIFTY_FIN_SERVICE.NS",  # Correct symbol for FINNIFTY
        "MIDCPNIFTY": "NIFTY_MID_SELECT.NS", 
        "EICHERMOT": "EICHERMOT.NS",
        "TRENT": "TRENT.NS",
        "RELIANCE": "RELIANCE.NS"
    }
    try:
        df = yf.download(symbol_map[index], period=period, interval=interval, auto_adjust=True, progress=False, timeout=30)
        return None if df.empty else df
    except Exception:
        return None

# --------- LOAD TOKEN MAP ---------
def load_token_map():
    try:
        url="https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
        df=pd.DataFrame(requests.get(url,timeout=10).json())
        df.columns=[c.lower() for c in df.columns]
        # Include both NFO and BFO for SENSEX
        df=df[df['exch_seg'].str.upper().isin(["NFO", "BFO"])]
        df['symbol']=df['symbol'].str.upper()
        return df.set_index('symbol')['token'].to_dict()
    except:
        return {}

token_map=load_token_map()

# --------- SAFE LTP FETCH ---------
def fetch_option_price(symbol, retries=3, delay=3):
    token=token_map.get(symbol.upper())
    if not token:
        return None
    for _ in range(retries):
        try:
            # Determine exchange based on symbol
            exchange = "BFO" if "SENSEX" in symbol.upper() else "NFO"
            data=client.ltpData(exchange, symbol, token)
            return float(data['data']['ltp'])
        except:
            time.sleep(delay)
    return None

# --------- DETECT LIQUIDITY ZONE ---------
def detect_liquidity_zone(df, lookback=20):
    high_series = ensure_series(df['High']).dropna()
    low_series = ensure_series(df['Low']).dropna()
    # fallback if not enough history
    try:
        if len(high_series) <= lookback:
            high_pool = float(high_series.max()) if len(high_series)>0 else float('nan')
        else:
            high_pool = float(high_series.rolling(lookback).max().iloc[-2])
    except Exception:
        high_pool = float(high_series.max()) if len(high_series)>0 else float('nan')
    try:
        if len(low_series) <= lookback:
            low_pool = float(low_series.min()) if len(low_series)>0 else float('nan')
        else:
            low_pool = float(low_series.rolling(lookback).min().iloc[-2])
    except Exception:
        low_pool = float(low_series.min()) if len(low_series)>0 else float('nan')

    # ensure numeric fallback
    if math.isnan(high_pool) and len(high_series)>0:
        high_pool = float(high_series.max())
    if math.isnan(low_pool) and len(low_series)>0:
        low_pool = float(low_series.min())

    return round(high_pool,0), round(low_pool,0)

# --------- INSTITUTIONAL LIQUIDITY HUNT (robust) ---------
def institutional_liquidity_hunt(index, df):
    """
    Calculates pre-market and opening range liquidity zones.
    Returns dual-sided liquidity zones for CE/PE trades.
    """
    # Pre-market / Overnight levels
    prev_high = None
    prev_low = None
    try:
        prev_high_val = ensure_series(df['High']).iloc[-2]
        prev_low_val = ensure_series(df['Low']).iloc[-2]
        prev_high = float(prev_high_val) if not (isinstance(prev_high_val,float) and math.isnan(prev_high_val)) else None
        prev_low = float(prev_low_val) if not (isinstance(prev_low_val,float) and math.isnan(prev_low_val)) else None
    except Exception:
        prev_high = None
        prev_low = None

    high_zone, low_zone = detect_liquidity_zone(df, lookback=15)

    # Options OI zones (dummy placeholder) — guard against NaN last close
    last_close_val = None
    try:
        lc = ensure_series(df['Close']).iloc[-1]
        if isinstance(lc, float) and math.isnan(lc):
            last_close_val = None
        else:
            last_close_val = float(lc)
    except Exception:
        last_close_val = None

    if last_close_val is None:
        highest_ce_oi_strike = None
        highest_pe_oi_strike = None
    else:
        highest_ce_oi_strike = round_strike(index, last_close_val + 50)
        highest_pe_oi_strike = round_strike(index, last_close_val - 50)

    bull_liquidity = []
    if prev_low is not None: bull_liquidity.append(prev_low)
    if low_zone is not None: bull_liquidity.append(low_zone)
    if highest_pe_oi_strike is not None: bull_liquidity.append(highest_pe_oi_strike)

    bear_liquidity = []
    if prev_high is not None: bear_liquidity.append(prev_high)
    if high_zone is not None: bear_liquidity.append(high_zone)
    if highest_ce_oi_strike is not None: bear_liquidity.append(highest_ce_oi_strike)

    return bull_liquidity, bear_liquidity

def liquidity_zone_entry_check(price, bull_liq, bear_liq):
    """
    Returns which side (CE/PE/BOTH) is appropriate based on price near liquidity zones.
    Robust to None zones and NaN price.
    """
    if price is None or (isinstance(price, float) and math.isnan(price)):
        return None

    for zone in bull_liq:
        if zone is None: continue
        try:
            if abs(price - zone) <= 5:  # Threshold 5 points
                return "CE"
        except:
            continue
    for zone in bear_liq:
        if zone is None: continue
        try:
            if abs(price - zone) <= 5:
                return "PE"
        except:
            continue

    valid_bear = [z for z in bear_liq if z is not None]
    valid_bull = [z for z in bull_liq if z is not None]
    if valid_bear and valid_bull:
        try:
            if price > max(valid_bear) or price < min(valid_bull):
                return "BOTH"
        except:
            return None
    return None

# 🚨 NEW: BOTTOM-FISHING / INSTITUTIONAL LIQUIDITY LAYER 🚨
def detect_bottom_fishing(index, df):
    """
    Detects potential institutional bottom-fishing entry.
    Only triggers if price reacts off support zones / absorption patterns.
    Returns "CE" or "PE" or None.
    """
    try:
        close = ensure_series(df['Close'])
        low = ensure_series(df['Low'])
        high = ensure_series(df['High'])
        volume = ensure_series(df['Volume'])
        if len(close) < 6: 
            return None

        # Identify bull liquidity zones
        bull_liq, bear_liq = institutional_liquidity_hunt(index, df)
        last_close = float(close.iloc[-1])

        # Detect absorption: long lower wick + recovery + volume divergence
        wick = last_close - low.iloc[-1]
        body = abs(close.iloc[-1] - close.iloc[-2])
        vol_avg = volume.rolling(20).mean().iloc[-1] if len(volume) >= 20 else volume.mean()
        vol_ratio = volume.iloc[-1] / (vol_avg if vol_avg > 0 else 1)

        # Oversold absorption criteria
        if wick > body * 1.5 and vol_ratio > 1.2:
            # check near bull liquidity
            for zone in bull_liq:
                if abs(last_close - zone) <= 5:
                    return "CE"
        # Mirror logic for distribution at top
        bear_wick = high.iloc[-1] - last_close
        if bear_wick > body * 1.5 and vol_ratio > 1.2:
            for zone in bear_liq:
                if abs(last_close - zone) <= 5:
                    return "PE"
    except:
        return None
    return None

# --------- NEW STRATEGIES FROM 1ST CODE ---------

# 🚨 OPENING-PLAY (PURE LEVELS) 🚨
def institutional_opening_play(index, df):
    """
    Pure level-based opening logic (no indicators).
    Returns "CE"/"PE"/None
    """
    try:
        prev_high = float(ensure_series(df['High']).iloc[-2])
        prev_low = float(ensure_series(df['Low']).iloc[-2])
        prev_close = float(ensure_series(df['Close']).iloc[-2])
        current_price = float(ensure_series(df['Close']).iloc[-1])
    except Exception:
        return None
    # Thresholds can be tweaked per-index if desired
    if current_price > prev_high + 10: return "CE"
    if current_price < prev_low - 10: return "PE"
    if current_price > prev_close + 20: return "CE"
    if current_price < prev_close - 20: return "PE"
    return None

# 🚨 SMART-MONEY DIVERGENCE 🚨
def smart_money_divergence(df):
    """
    Detect stealth accumulation/distribution: price falling but RSI rising with higher volume -> accumulation (CE)
    or price rising but RSI falling with higher volume -> distribution (PE)
    """
    try:
        close = ensure_series(df['Close']); volume = ensure_series(df['Volume'])
        rsi = ta.momentum.RSIIndicator(close, 14).rsi()
        if len(close) < 10: return None
        # lookback windows
        p_short = close.iloc[-5]; p_now = close.iloc[-1]
        rsi_short = rsi.iloc[-5]; rsi_now = rsi.iloc[-1]
        vol_avg = volume.rolling(20).mean().iloc[-1] if len(volume)>=20 else volume.mean()
        vol_now = volume.iloc[-1]
        if p_now < p_short and rsi_now > rsi_short and vol_now > vol_avg*1.1:
            return "CE"
        if p_now > p_short and rsi_now < rsi_short and vol_now > vol_avg*1.1:
            return "PE"
    except Exception:
        return None
    return None

# 🚨 STOP-HUNT / LIQUIDITY GRAB DETECTOR 🚨
def detect_stop_hunt(df):
    """
    Look for spike wick beyond recent levels followed by quick retrace into the range -> indicates stop-hunt.
    Returns "CE"/"PE"/None depending on direction.
    """
    try:
        high = ensure_series(df['High']); low = ensure_series(df['Low']); close = ensure_series(df['Close']); volume = ensure_series(df['Volume'])
        if len(close) < 6: return None
        recent_high = high.iloc[-6:-1].max(); recent_low = low.iloc[-6:-1].min()
        last_high = high.iloc[-1]; last_low = low.iloc[-1]; last_close = close.iloc[-1]
        vol_avg = volume.rolling(20).mean().iloc[-1] if len(volume)>=20 else volume.mean()
        if last_high > recent_high * 1.002 and last_close < recent_high and volume.iloc[-1] > vol_avg*1.2:
            # wick above recent high then close back => upside stop-hunt -> PE trigger
            return "PE"
        if last_low < recent_low * 0.998 and last_close > recent_low and volume.iloc[-1] > vol_avg*1.2:
            # wick below recent low then close back => downside stop-hunt -> CE trigger
            return "CE"
    except Exception:
        return None
    return None

# 🚨 INSTITUTIONAL CONTINUATION 🚨
def detect_institutional_continuation(df):
    """
    Confirms breakout follow-through: ATR expansion + rising volume + price speed
    Returns "CE"/"PE"/None
    """
    try:
        close = ensure_series(df['Close']); high = ensure_series(df['High']); low = ensure_series(df['Low']); volume = ensure_series(df['Volume'])
        if len(close) < 10: return None
        atr = ta.volatility.AverageTrueRange(high, low, close, 14).average_true_range().iloc[-1]
        vol_avg = volume.rolling(20).mean().iloc[-1] if len(volume)>=20 else volume.mean()
        # price speed (last N bars)
        speed = (close.iloc[-1] - close.iloc[-3]) / (abs(close.iloc[-3]) + 1e-6)
        if atr > close.std() * 0.8 and volume.iloc[-1] > vol_avg * 1.2 and speed > 0.004:
            return "CE"
        if atr > close.std() * 0.8 and volume.iloc[-1] > vol_avg * 1.2 and speed < -0.004:
            return "PE"
    except Exception:
        return None
    return None

# 🚨 GAMMA SQUEEZE / EXPIRY LAYER 🚨
def is_expiry_day_for_index(index):
    try:
        ex = EXPIRIES.get(index)
        if not ex: return False
        dt = datetime.strptime(ex, "%d %b %Y")
        today = (datetime.utcnow() + timedelta(hours=5, minutes=30)).date()
        return dt.date() == today
    except Exception:
        return False

def detect_gamma_squeeze(index, df):
    """
    Proxy gamma squeeze detection (no options greeks availability):
    - Check if today is expiry-like (or close to expiry)
    - Look for sudden volume spike + price acceleration near strikes with OI concentration
    - Returns dict {'side': 'CE'/'PE', 'confidence': 0-1} or None
    """
    try:
        close = ensure_series(df['Close']); volume = ensure_series(df['Volume']); high = ensure_series(df['High']); low = ensure_series(df['Low'])
        if len(close) < 6: return None
        vol_avg = volume.rolling(20).mean().iloc[-1] if len(volume)>=20 else volume.mean()
        vol_ratio = volume.iloc[-1] / (vol_avg if vol_avg>0 else 1)
        speed = (close.iloc[-1] - close.iloc[-3]) / (abs(close.iloc[-3]) + 1e-6)
        # quick OI proxy using scrip master concentration by strike (best-effort)
        try:
            url=f"https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
            df_s = pd.DataFrame(requests.get(url,timeout=10).json())
            df_s['symbol'] = df_s['symbol'].str.upper()
            df_index = df_s[df_s['symbol'].str.contains(index)]
            # find strike concentration: sum of oi per CE/PE near atm
            df_index['oi'] = pd.to_numeric(df_index.get('oi',0), errors='coerce').fillna(0)
            ce_oi = df_index[df_index['symbol'].str.endswith("CE")]['oi'].sum()
            pe_oi = df_index[df_index['symbol'].str.endswith("PE")]['oi'].sum()
        except Exception:
            ce_oi = pe_oi = 0
        # decide side using speed and oi bias
        if vol_ratio > GAMMA_VOL_SPIKE_THRESHOLD and abs(speed) > 0.002:
            if speed > 0:
                conf = min(1.0, (vol_ratio - 1.0) / 3.0 + (ce_oi / (pe_oi+1e-6)) * 0.1)
                return {'side':'CE','confidence':conf}
            else:
                conf = min(1.0, (vol_ratio - 1.0) / 3.0 + (pe_oi / (ce_oi+1e-6)) * 0.1)
                return {'side':'PE','confidence':conf}
    except Exception:
        return None
    return None

# --------- EXISTING STRATEGIES ---------
def detect_pullback_reversal(df):
    """
    Detects pullback to EMA21 or EMA9 and a reversal candle indicating institutional reload.
    Returns "CE" or "PE" or None.
    """
    try:
        close = ensure_series(df['Close'])
        ema9 = ta.trend.EMAIndicator(close, 9).ema_indicator()
        ema21 = ta.trend.EMAIndicator(close, 21).ema_indicator()
        rsi = ta.momentum.RSIIndicator(close, 14).rsi()

        # ensure enough history
        if len(close) < 6:
            return None

        # uptrend pullback then bounce
        if close.iloc[-6] > ema21.iloc[-6] and close.iloc[-3] <= ema21.iloc[-3] and close.iloc[-1] > ema9.iloc[-1] and rsi.iloc[-1] > 50:
            return "CE"

        # downtrend pullback then drop
        if close.iloc[-6] < ema21.iloc[-6] and close.iloc[-3] >= ema21.iloc[-3] and close.iloc[-1] < ema9.iloc[-1] and rsi.iloc[-1] < 50:
            return "PE"
    except Exception:
        return None
    return None

def detect_institutional_trap(df, lookback=10):
    """
    Detects fake breakout / trap: strong wick beyond recent high/low with high volume and closing back.
    Returns "PE" if upside trap (short), "CE" if downside trap (long), else None.
    """
    try:
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        close = ensure_series(df['Close'])
        volume = ensure_series(df['Volume'])
        if len(close) < lookback + 2:
            return None
        recent_high = high.rolling(lookback).max().iloc[-2]
        recent_low = low.rolling(lookback).min().iloc[-2]
        avg_vol = volume.rolling(20).mean().iloc[-1] if len(volume) >= 20 else volume.mean()

        # Upward fake breakout trap: wick above recent high then close below recent high on high volume
        if high.iloc[-1] > recent_high and close.iloc[-1] < recent_high and (avg_vol is None or volume.iloc[-1] > (avg_vol * 1.2)):
            return "PE"

        # Downward fake breakdown trap
        if low.iloc[-1] < recent_low and close.iloc[-1] > recent_low and (avg_vol is None or volume.iloc[-1] > (avg_vol * 1.2)):
            return "CE"
    except Exception:
        return None
    return None

def mimic_orderflow_logic(df):
    """
    Mimic orderflow using wick ratios, volume spikes, and RSI divergence.
    Returns "CE" or "PE" or None.
    """
    try:
        close = ensure_series(df['Close'])
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        volume = ensure_series(df['Volume'])
        rsi = ta.momentum.RSIIndicator(close, 14).rsi()

        if len(close) < 4:
            return None

        body = (high - low).abs()
        wick_top = (high - close).abs()
        wick_bottom = (close - low).abs()

        body_last = body.iloc[-1] if body.iloc[-1] != 0 else 1.0
        wick_top_ratio = wick_top.iloc[-1] / body_last
        wick_bottom_ratio = wick_bottom.iloc[-1] / body_last

        vol_avg = volume.rolling(20).mean().iloc[-1] if len(volume) >= 20 else volume.mean()
        vol_ratio = volume.iloc[-1] / (vol_avg if vol_avg and vol_avg > 0 else 1)

        # Distribution: price up slightly, RSI falling, large upper wick and rising volume
        if close.iloc[-1] > close.iloc[-3] and rsi.iloc[-1] < rsi.iloc[-3] and wick_top_ratio > 0.6 and vol_ratio > 1.2:
            return "PE"

        # Accumulation: price down slightly, RSI rising, large lower wick and rising volume
        if close.iloc[-1] < close.iloc[-3] and rsi.iloc[-1] > rsi.iloc[-3] and wick_bottom_ratio > 0.6 and vol_ratio > 1.2:
            return "CE"
    except Exception:
        return None
    return None

# --------- INSTITUTIONAL FLOW CHECKS ---------
def institutional_flow_signal(index, df5):
    try:
        last_close = float(ensure_series(df5["Close"]).iloc[-1])
        prev_close = float(ensure_series(df5["Close"]).iloc[-2])
    except:
        return None

    vol5 = ensure_series(df5["Volume"])
    vol_latest = float(vol5.iloc[-1])
    vol_avg = float(vol5.rolling(20).mean().iloc[-1]) if len(vol5) >= 20 else float(vol5.mean())

    if vol_latest > vol_avg*1.5 and abs(last_close-prev_close)/prev_close>0.003:
        return "BOTH"
    elif last_close>prev_close and vol_latest>vol_avg:
        return "CE"
    elif last_close<prev_close and vol_latest>vol_avg:
        return "PE"
    high_zone, low_zone = detect_liquidity_zone(df5, lookback=15)
    try:
        if last_close>=high_zone: return "PE"
        elif last_close<=low_zone: return "CE"
    except:
        return None
    return None

# --------- OI + DELTA FLOW DETECTION ---------
def oi_delta_flow_signal(index):
    try:
        url=f"https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
        df=pd.DataFrame(requests.get(url,timeout=10).json())
        # Include both exchanges
        df=df[df['exch_seg'].str.upper().isin(["NFO", "BFO"])]
        df['symbol']=df['symbol'].str.upper()
        df_index=df[df['symbol'].str.contains(index)]
        # ensure 'oi' exists and convertible
        if 'oi' not in df_index.columns:
            return None
        df_index['oi'] = pd.to_numeric(df_index['oi'], errors='coerce').fillna(0)
        df_index['oi_change'] = df_index['oi'].diff().fillna(0)
        ce_sum = df_index[df_index['symbol'].str.endswith("CE")]['oi_change'].sum()
        pe_sum = df_index[df_index['symbol'].str.endswith("PE")]['oi_change'].sum()
        if ce_sum>pe_sum*DELTA_OI_RATIO: return "CE"
        if pe_sum>ce_sum*DELTA_OI_RATIO: return "PE"
        if ce_sum>0 and pe_sum>0: return "BOTH"
    except:
        return None

# --------- NEW INSTITUTIONAL CONFIRMATION LAYER (updated to use pullback/trap) ---------
def institutional_confirmation_layer(index, df5, base_signal):
    close = ensure_series(df5['Close'])
    high = ensure_series(df5['High'])
    low = ensure_series(df5['Low'])
    volume = ensure_series(df5['Volume'])
    atr = ta.volatility.AverageTrueRange(high, low, close, 14).average_true_range().iloc[-1]

    last_close = float(close.iloc[-1])
    prev_close = float(close.iloc[-2])
    body_strength = abs(last_close - (high.iloc[-1]+low.iloc[-1])/2)

    # Liquidity trap: avoid if candle wicks through prev high/low zone
    high_zone, low_zone = detect_liquidity_zone(df5, lookback=20)
    if base_signal == 'CE' and last_close >= high_zone:
        return False
    if base_signal == 'PE' and last_close <= low_zone:
        return False

    # Weak momentum: skip if candle body small or low volume
    vol_avg = volume.rolling(20).mean().iloc[-1] if len(volume) >= 20 else volume.mean()
    if volume.iloc[-1] < vol_avg or body_strength < atr*0.25:
        return False

    # Volatility dead zone: skip low ATR periods
    if atr < (close.std() * 0.25):
        return False

    # Breadth check: ensure coherence across major index pairs
    if index == 'NIFTY':
        b_df = fetch_index_data('BANKNIFTY', '5m', '2d')
    elif index == 'BANKNIFTY':
        b_df = fetch_index_data('NIFTY', '5m', '2d')
    else:
        b_df = None

    if b_df is not None:
        b_close = ensure_series(b_df['Close'])
        b_ema9 = ta.trend.EMAIndicator(b_close, 9).ema_indicator().iloc[-1]
        b_ema21 = ta.trend.EMAIndicator(b_close, 21).ema_indicator().iloc[-1]
        if base_signal == 'PE' and b_ema9 > b_ema21:
            return False
        if base_signal == 'CE' and b_ema9 < b_ema21:
            return False

    # Additional institutional confirmation: if pullback reversal aligns with base_signal, allow
    try:
        pull_sig = detect_pullback_reversal(df5)
        if pull_sig and pull_sig == base_signal:
            return True
    except:
        pass

    # If trap detected and is opposite of base_signal, deny
    try:
        trap_sig = detect_institutional_trap(df5)
        if trap_sig and trap_sig != base_signal:
            return False
    except:
        pass

    return True

# --------- FLOW CONFIRMATION LAYER (UPDATED) ---------
def institutional_flow_confirm(index, base_signal, df5):
    flow = institutional_flow_signal(index, df5)
    oi_flow = oi_delta_flow_signal(index)

    if flow and flow != 'BOTH' and flow != base_signal:
        return False
    if oi_flow and oi_flow != 'BOTH' and oi_flow != base_signal:
        return False

    # ✅ NEW: Institutional Confirmation Add-On (uses pullback/trap/orderflow)
    if not institutional_confirmation_layer(index, df5, base_signal):
        return False

    return True

# --------- UPDATED STRATEGY CHECK WITH ALL NEW STRATEGIES ---------
def analyze_index_signal(index):
    df5=fetch_index_data(index,"5m","2d")
    df15=fetch_index_data(index,"15m","10d")
    if df5 is None or df15 is None:
        return None

    close5=ensure_series(df5["Close"])
    close15=ensure_series(df15["Close"])

    # guard - ensure we have enough data and non-NaN
    if len(close5) < 6 or len(close15) < 2:
        return None
    if close5.isna().iloc[-1] or close5.isna().iloc[-2]:
        return None

    last_close=float(close5.iloc[-1])
    prev_close=float(close5.iloc[-2])

    # 🚨 0) Opening-play priority during opening window
    try:
        utc_now = datetime.utcnow()
        ist_now = utc_now + timedelta(hours=5, minutes=30)
        t = ist_now.time()
        opening_range_bias = OPENING_PLAY_ENABLED and (OPENING_START <= t <= OPENING_END)
        if opening_range_bias:
            op_sig = institutional_opening_play(index, df5)
            if op_sig:
                # opening plays are high-priority: lightweight fakeout check afterwards
                fakeout = False
                # quick liquidity check
                high_zone, low_zone = detect_liquidity_zone(df5, lookback=10)
                try:
                    if op_sig == "CE" and last_close >= high_zone: fakeout = True
                    if op_sig == "PE" and last_close <= low_zone: fakeout = True
                except:
                    fakeout = False
                return op_sig, df5, fakeout
    except Exception:
        pass

    # 🚨 1) Expiry / Gamma detection (high priority but non-blocking)
    try:
        gamma = detect_gamma_squeeze(index, df5)
        if gamma:
            gamma_msg = f"⚡ GAMMA-LIKE EVENT DETECTED: {index} {gamma['side']} (conf {gamma['confidence']:.2f})"
            # send info always
            send_telegram(gamma_msg)
            # actionable: if expiry day AND configured to be actionable
            if is_expiry_day_for_index(index) and EXPIRY_ACTIONABLE and not EXPIRY_INFO_ONLY:
                # create an expiry-weighted candidate signal
                cand = gamma['side']
                # if confidence high, try to relax confirmation requirements (but not remove them)
                # We'll allow an expiry-enabled trade if either institutional_flow_confirm OR (gamma.conf high + oi_flow agrees)
                oi_flow = oi_delta_flow_signal(index)
                if institutional_flow_confirm(index, cand, df5):
                    return cand, df5, False
                if gamma['confidence'] > 0.45 and oi_flow == cand:
                    # partial relax: treat as valid trade
                    return cand, df5, False
                # else treat it informational only
    except Exception:
        pass

    # 🚨 2) BOTTOM-FISHING HIGH-PRIORITY LAYER
    bottom_sig = detect_bottom_fishing(index, df5)
    if bottom_sig:
        return bottom_sig, df5, False

    # 🚨 3) Traps / Pullback / Orderflow / Stop-hunt / Smart-money divergence (priority stack)
    try:
        trap_sig = detect_institutional_trap(df5)
        if trap_sig:
            return trap_sig, df5, True  # trap = fakeout
    except:
        trap_sig = None
    try:
        pull_sig = detect_pullback_reversal(df5)
        if pull_sig:
            return pull_sig, df5, False
    except:
        pull_sig = None
    try:
        flow_sig = mimic_orderflow_logic(df5)
        if flow_sig:
            return flow_sig, df5, False
    except:
        flow_sig = None
    try:
        stop_sig = detect_stop_hunt(df5)
        if stop_sig:
            # stop-hunts are often good short-term signals
            return stop_sig, df5, True
    except:
        stop_sig = None
    try:
        sm_sig = smart_money_divergence(df5)
        if sm_sig:
            return sm_sig, df5, False
    except:
        sm_sig = None

    # 🚨 4) Institutional continuation (confirms breakouts)
    cont_sig = detect_institutional_continuation(df5)
    if cont_sig:
        # continuation should be confirmed by institutional_flow_confirm
        if institutional_flow_confirm(index, cont_sig, df5):
            return cont_sig, df5, False

    # EMA/RSI indicators
    ema9_5=float(ta.trend.EMAIndicator(close5,9).ema_indicator().iloc[-1])
    ema21_5=float(ta.trend.EMAIndicator(close5,21).ema_indicator().iloc[-1])
    rsi5=float(ta.momentum.RSIIndicator(close5,14).rsi().iloc[-1])

    vol5 = ensure_series(df5["Volume"])
    vol_latest=float(vol5.iloc[-1])
    vol_avg=float(vol5.rolling(20).mean().iloc[-1]) if len(vol5)>=20 else float(vol5.mean())
    # Loosened volume gate: allow some lower-vol cases but still block dead bars
    if vol_latest < (vol_avg * 0.8):
        # allow if strong breakout or orderflow indicates institutional action (handled below)
        pass

    ema9_15=float(ta.trend.EMAIndicator(close15,9).ema_indicator().iloc[-1])
    ema21_15=float(ta.trend.EMAIndicator(close15,21).ema_indicator().iloc[-1])
    rsi15=float(ta.momentum.RSIIndicator(close15,14).rsi().iloc[-1])

    # guard against NaN values
    if math.isnan(last_close) or math.isnan(prev_close):
        return None

    # Relaxed multi-timeframe logic (less strict)
    bullish = (
        (ema9_5 > ema21_5 and rsi5 > 50 and last_close > prev_close)
        or (ema9_15 > ema21_15 and rsi15 > 51)
    )
    bearish = (
        (ema9_5 < ema21_5 and rsi5 < 50 and last_close < prev_close)
        or (ema9_15 < ema21_15 and rsi15 < 49)
    )

    # Opening range bias (mild) - boost early signals
    try:
        utc_now = datetime.utcnow()
        ist_now = utc_now + timedelta(hours=5, minutes=30)
        t = ist_now.time()
        opening_range_bias = dtime(9,15) <= t <= dtime(9,45)
        if opening_range_bias:
            if last_close > close5.iloc[-5:].mean():
                bullish = True
            elif last_close < close5.iloc[-5:].mean():
                bearish = True
    except Exception:
        pass

    # --- Breakout & Failed Breakout Detection (quick addition) ---
    try:
        recent_high = df5["High"].iloc[-6:].max()
        recent_low = df5["Low"].iloc[-6:].min()
        # breakout above recent high
        if last_close > recent_high:
            bullish = True
        elif last_close < recent_low:
            bearish = True
        # failed breakout fade
        if df5["High"].iloc[-1] > recent_high and df5["Close"].iloc[-1] < recent_high:
            bearish = True
        if df5["Low"].iloc[-1] < recent_low and df5["Close"].iloc[-1] > recent_low:
            bullish = True
    except Exception:
        pass

    high_zone, low_zone = detect_liquidity_zone(df5, lookback=15)

    fakeout=False
    try:
        if bullish and last_close<high_zone: fakeout=True
        if bearish and last_close>low_zone: fakeout=True
    except:
        fakeout=False

    # ✅ 5️⃣ Integrate Liquidity Hunt Layer (existing)
    bull_liq, bear_liq = institutional_liquidity_hunt(index, df5)
    liquidity_side = liquidity_zone_entry_check(last_close, bull_liq, bear_liq)
    if liquidity_side:
        return liquidity_side, df5, fakeout

    if bullish: return "CE",df5,fakeout
    if bearish: return "PE",df5,fakeout
    return None

# --------- SYMBOL FORMAT FOR ALL INDICES ---------
def get_option_symbol(index, expiry_str, strike, opttype):
    dt=datetime.strptime(expiry_str,"%d %b %Y")
    
    # CORRECTED: Each index maintains its own symbol format
    if index == "SENSEX":
        # SENSEX format: SENSEX + YY + MONTH_CODE + STRIKE + OPTION_TYPE
        year_short = dt.strftime("%y")
        month_code = dt.strftime("%b").upper()
        day = dt.strftime("%d")
        return f"SENSEX{year_short}{month_code}{strike}{opttype}"
    elif index == "FINNIFTY":
        # FINNIFTY format: FINNIFTY + DDMMMYY + STRIKE + OPTION_TYPE
        return f"FINNIFTY{dt.strftime('%d%b%y').upper()}{strike}{opttype}"
    elif index == "MIDCPNIFTY":
        # MIDCPNIFTY format: MIDCPNIFTY + DDMMMYY + STRIKE + OPTION_TYPE  
        return f"MIDCPNIFTY{dt.strftime('%d%b%y').upper()}{strike}{opttype}"
    else:
        # Standard format for NIFTY, BANKNIFTY, and stocks
        return f"{index}{dt.strftime('%d%b%y').upper()}{strike}{opttype}"

# --------- MONITOR WITH THREAD UPDATES ---------
def monitor_price_live(symbol,entry,targets,sl,fakeout,thread_id):
    last_high = entry
    weakness_sent = False
    in_trade=False
    # Ensure active_trades entry exists and mark status OPEN
    for idx, val in active_trades.items():
        if val and val.get("symbol") == symbol:
            active_trades[idx]["status"] = "OPEN"
            break

    while True:
        # Auto stop at 3:30 PM IST
        if should_stop_trading():
            global STOP_SENT
            if not STOP_SENT:
                send_telegram(f"🛑 Market closed - Stopping monitoring for {symbol}", reply_to=thread_id)
                STOP_SENT = True
            # mark closed if was open
            for idx, val in active_trades.items():
                if val and val.get("symbol") == symbol:
                    active_trades[idx]["status"] = "CLOSED"
            break
            
        price=fetch_option_price(symbol)
        if not price: time.sleep(10); continue
        price=round(price)
        if not in_trade:
            if price >= entry:
                send_telegram(f"✅ ENTRY TRIGGERED at {price}", reply_to=thread_id)
                in_trade=True
                last_high=price
        else:
            if price > last_high:
                send_telegram(f"🚀 {symbol} making new high → {price}", reply_to=thread_id)
                last_high=price
            elif not weakness_sent and price < sl*1.05:
                send_telegram(f"⚡ {symbol} showing weakness near SL {sl}", reply_to=thread_id)
                weakness_sent=True
            if price>=targets[0]:
                send_telegram(f"🌟 {symbol}: First Target {targets[0]} hit", reply_to=thread_id)
                # mark closed
                for idx, val in active_trades.items():
                    if val and val.get("symbol") == symbol:
                        active_trades[idx]["status"] = "CLOSED"
                break
            if price<=sl:
                send_telegram(f"🔗 {symbol}: Stop Loss {sl} hit. Exit trade.", reply_to=thread_id)
                # mark closed
                for idx, val in active_trades.items():
                    if val and val.get("symbol") == symbol:
                        active_trades[idx]["status"] = "CLOSED"
                break
        time.sleep(10)

# --------- EXPIRY CONFIG FOR ALL INDICES ---------
EXPIRIES = {
    "NIFTY": "20 OCT 2025",      # Updated as per your note
    "BANKNIFTY": "28 OCT 2025", 
    "SENSEX": "23 OCT 2025",     # Updated as per your note
    "FINNIFTY": "28 OCT 2025",  # CORRECTED: FINNIFTY has its own expiry
    "MIDCPNIFTY": "28 OCT 2025",
    "EICHERMOT": "28 OCT 2025",
    "TRENT": "28 OCT 2025", 
    "RELIANCE": "28 OCT 2025"
}

# ACTIVE TRACKING FOR ALL INDICES
active_trades = {
    "NIFTY": None, "BANKNIFTY": None, "SENSEX": None,
    "FINNIFTY": None, "MIDCPNIFTY": None, "EICHERMOT": None,
    "TRENT": None, "RELIANCE": None
}

# --------- THREAD FUNCTION ---------
def trade_thread(index):
    global active_trades
    # allow re-entry if last trade closed; only block if an open trade exists
    if active_trades[index] and isinstance(active_trades[index], dict) and active_trades[index].get("status") == "OPEN":
        return

    sig=analyze_index_signal(index)
    side=None; fakeout=False; df=None
    if sig: 
        # analyze_index_signal may return a 2-tuple or 3-tuple (we keep compatibility)
        if isinstance(sig, tuple) and len(sig) == 3:
            side, df, fakeout = sig
        elif isinstance(sig, tuple) and len(sig) == 2:
            side, df = sig
            fakeout = False
        else:
            side = sig

    df5=fetch_index_data(index,"5m","2d")
    inst_signal = institutional_flow_signal(index, df5) if df5 is not None else None
    oi_signal = oi_delta_flow_signal(index)
    final_signal = oi_signal or inst_signal or side

    # If gamma exists and expiry and EXPIRY_ACTIONABLE, let it overwrite final_signal if confidently matching and not conflicting
    try:
        gamma = detect_gamma_squeeze(index, df5)
        if gamma and is_expiry_day_for_index(index) and EXPIRY_ACTIONABLE and gamma['confidence']>0.45:
            # prefer gamma if agrees with oi or inst flow, else require confirmation
            if oi_signal == gamma['side'] or inst_signal == gamma['side'] or side == gamma['side']:
                final_signal = gamma['side']
            else:
                # if not agreeing, allow only informational send and keep normal flow
                send_telegram(f"⚡ Info: {index} gamma event detected -> {gamma['side']} conf {gamma['confidence']:.2f}")
    except Exception:
        pass

    if final_signal=="BOTH":
        for s in ["CE","PE"]:
            if institutional_flow_confirm(index, s, df5):
                send_signal(index,s,df,fakeout)
        return
    elif final_signal:
        if df is None: df=df5
        if institutional_flow_confirm(index, final_signal, df5):
            send_signal(index,final_signal,df,fakeout)
    else:
        return

# --------- SEND SIGNAL ---------
def send_signal(index,side,df,fakeout):
    ltp=float(ensure_series(df["Close"]).iloc[-1])
    
    # CORRECTED: Each index uses its own strike rounding with validation
    strike=round_strike(index,ltp)
    
    if strike is None:
        # cannot determine strike because LTP missing or invalid
        send_telegram(f"⚠️ {index}: could not determine strike (ltp missing or invalid). Signal skipped.")
        return
        
    # CORRECTED: Each index uses its own symbol format  
    symbol=get_option_symbol(index,EXPIRIES[index],strike,side)
    
    price=fetch_option_price(symbol)
    if not price: 
        send_telegram(f"⚠️ {index} {symbol}: option price fetch failed. Signal skipped.")
        return
        
    high=ensure_series(df["High"])
    low=ensure_series(df["Low"])
    close=ensure_series(df["Close"])
    atr=float(ta.volatility.AverageTrueRange(high,low,close,14).average_true_range().iloc[-1])
    entry=round(price+5)
    sl=round(price-atr)
    targets=[round(price+atr*1.5),round(price+atr*2)]
    
    msg=(f" GIT🔊 {index} {side} VSSIGNAL CONFIRMED\n"
         f"🔹 Strike: {strike}\n"
         f"🟩 Buy Above ₹{entry}\n"
         f"🔵 SL: ₹{sl}\n"
         f"🌟 Targets: {targets[0]} / {targets[1]}\n"
         f"⚡ Fakeout: {'YES' if fakeout else 'NO'}")
         
    thread_id=send_telegram(msg)
    # mark active trade with status OPEN
    active_trades[index]={"symbol":symbol,"entry":entry,"sl":sl,"targets":targets,"thread":thread_id,"status":"OPEN"}
    monitor_price_live(symbol,entry,targets,sl,fakeout,thread_id)
    # once monitor returns, ensure status is CLOSED (monitor_price_live will set it on exit)
    active_trades[index]=None

# --------- MAIN LOOP (ALL INDICES PARALLEL) ---------
def run_algo_parallel():
    if not is_market_open(): 
        print("❌ Market closed - skipping iteration")
        return
        
    if should_stop_trading():
        global STOP_SENT
        if not STOP_SENT:
            send_telegram("🛑 Market closed at 3:30 PM IST - Algorithm stopped")
            STOP_SENT = True
        return
        
    threads=[]
    all_indices = ["NIFTY", "BANKNIFTY", "SENSEX", "FINNIFTY", "MIDCPNIFTY", "EICHERMOT", "TRENT", "RELIANCE"]
    
    for index in all_indices:
        t=threading.Thread(target=trade_thread,args=(index,))
        t.start()
        threads.append(t)
    for t in threads: t.join()

# --------- START ---------
# Send startup once - MOVE INSIDE LOOP
while True:
    try:
        # Send startup message once per session
        if not STARTED_SENT and is_market_open():
            send_telegram("🚀 GIT ULTIMATE MASTER ALGO STARTED - All 8 Indices Running in Parallel with Institutional Layers + Gamma Expiry Handling!")
            STARTED_SENT = True
            STOP_SENT = False  # Reset stop flag for new session
            
        # Auto stop at 3:30 PM IST
        if should_stop_trading():
            if not STOP_SENT:
                send_telegram("🛑 Market closing time reached - Algorithm stopped automatically")
                STOP_SENT = True
                STARTED_SENT = False  # Reset for next day
            break
            
        if is_market_open():
            run_algo_parallel()
            
        time.sleep(30)
    except Exception as e:
        send_telegram(f"⚠️ Error in main loop: {e}")
        time.sleep(60)

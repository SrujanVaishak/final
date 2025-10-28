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
OPENING_PLAY_ENABLED = True
OPENING_START = dtime(9,15)
OPENING_END = dtime(9,45)

EXPIRY_ACTIONABLE = True
EXPIRY_INFO_ONLY = False
EXPIRY_RELAX_FACTOR = 0.7
GAMMA_VOL_SPIKE_THRESHOLD = 2.0
DELTA_OI_RATIO = 2.0
MOMENTUM_VOL_AMPLIFIER = 1.5

# STRONGER CONFIRMATION THRESHOLDS
VCP_CONTRACTION_RATIO = 0.6
FAULTY_BASE_BREAK_THRESHOLD = 0.25
WYCKOFF_VOLUME_SPRING = 2.2
LIQUIDITY_SWEEP_DISTANCE = 0.005
PEAK_REJECTION_WICK_RATIO = 0.8
FVG_GAP_THRESHOLD = 0.0025
VOLUME_GAP_IMBALANCE = 2.5
OTE_RETRACEMENT_LEVELS = [0.618, 0.786]
DEMAND_SUPPLY_ZONE_LOOKBACK = 20

# NEW: ACCUMULATION PHASE DETECTION
ACCUMULATION_VOLUME_RATIO = 2.0
ACCUMULATION_PRICE_RANGE = 0.02
ACCUMULATION_DAYS_LOOKBACK = 10

# --------- EXPIRIES FOR ALL INDICES ---------
EXPIRIES = {
    "NIFTY": "28 OCT 2025",
    "BANKNIFTY": "28 OCT 2025",
    "SENSEX": "30 OCT 2025",
    "FINNIFTY": "28 OCT 2025",
    "MIDCPNIFTY": "28 OCT 2025",
    "EICHERMOT": "28 OCT 2025",
    "TRENT": "28 OCT 2025",
    "RELIANCE": "28 OCT 2025"
}

# --------- STRATEGY TRACKING ---------
STRATEGY_NAMES = {
    "INSTITUTIONAL_PRICE_ACTION": "INSTITUTIONAL PRICE ACTION",
    "OPENING_PLAY": "OPENING PLAY", 
    "GAMMA_SQUEEZE": "GAMMA SQUEEZE",
    "SMART_MONEY_DIVERGENCE": "SMART MONEY DIVERGENCE",
    "STOP_HUNT": "STOP HUNT",
    "INSTITUTIONAL_CONTINUATION": "INSTITUTIONAL CONTINUATION",
    "PULLBACK_REVERSAL": "PULLBACK REVERSAL",
    "ORDERFLOW_MIMIC": "ORDERFLOW MIMIC",
    "VCP_PATTERN": "VCP PATTERN",
    "FAULTY_BASES": "FAULTY BASES",
    "WYCKOFF_SCHEMATIC": "WYCKOFF SCHEMATIC",
    "LIQUIDITY_SWEEPS": "LIQUIDITY SWEEP",
    "PEAK_REJECTION": "PEAK REJECTION",
    "FAIR_VALUE_GAP": "FAIR VALUE GAP",
    "VOLUME_GAP_IMBALANCE": "VOLUME GAP IMBALANCE",
    "OTE_RETRACEMENT": "OTE RETRACEMENT",
    "DEMAND_SUPPLY_ZONES": "DEMAND SUPPLY ZONES",
    "BOTTOM_FISHING": "BOTTOM FISHING"
}

# Track all signals for end-of-day report
daily_signals = []
signal_counter = 0

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
    utc_now = datetime.utcnow()
    ist_now = utc_now + timedelta(hours=5, minutes=30)
    current_time_ist = ist_now.time()
    return dtime(9,15) <= current_time_ist <= dtime(15,30)

def should_stop_trading():
    utc_now = datetime.utcnow()
    ist_now = utc_now + timedelta(hours=5, minutes=30)
    current_time_ist = ist_now.time()
    return current_time_ist >= dtime(15,30)

# --------- STRIKE ROUNDING FOR ALL INDICES ---------
def round_strike(index, price):
    try:
        if price is None:
            return None
        if isinstance(price, float) and math.isnan(price):
            return None
        price = float(price)
        
        if index == "NIFTY": 
            return int(round(price / 50.0) * 50)
        elif index == "BANKNIFTY": 
            return int(round(price / 100.0) * 100)
        elif index == "SENSEX": 
            return int(round(price / 100.0) * 100)
        elif index == "FINNIFTY": 
            return int(round(price / 50.0) * 50)
        elif index == "MIDCPNIFTY": 
            return int(round(price / 25.0) * 25)
        elif index == "EICHERMOT": 
            return int(round(price / 50.0) * 50)
        elif index == "TRENT": 
            return int(round(price / 100.0) * 100)
        elif index == "RELIANCE": 
            return int(round(price / 10.0) * 10)
        else: 
            return int(round(price / 50.0) * 50)
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
        "FINNIFTY": "NIFTY_FIN_SERVICE.NS",
        "MIDCPNIFTY": "NIFTY_MID_SELECT.NS", 
        "EICHERMOT": "EICHERMOT.NS",
        "TRENT": "TRENT.NS",
        "RELIANCE": "RELIANCE.NS"
    }
    df = yf.download(symbol_map[index], period=period, interval=interval, auto_adjust=True, progress=False)
    return None if df.empty else df

# --------- LOAD TOKEN MAP ---------
def load_token_map():
    try:
        url="https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
        df=pd.DataFrame(requests.get(url,timeout=10).json())
        df.columns=[c.lower() for c in df.columns]
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

    if math.isnan(high_pool) and len(high_series)>0:
        high_pool = float(high_series.max())
    if math.isnan(low_pool) and len(low_series)>0:
        low_pool = float(low_series.min())

    return round(high_pool,0), round(low_pool,0)

# --------- INSTITUTIONAL LIQUIDITY HUNT ---------
def institutional_liquidity_hunt(index, df):
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
    if price is None or (isinstance(price, float) and math.isnan(price)):
        return None

    for zone in bull_liq:
        if zone is None: continue
        try:
            if abs(price - zone) <= 5:
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

# 🚨 NEW: INSTITUTIONAL PRICE ACTION LAYER 🚨
def institutional_price_action_signal(df):
    """
    Pure price action based institutional signals
    Focuses on breakouts, rejections, and momentum
    """
    try:
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        close = ensure_series(df['Close'])
        volume = ensure_series(df['Volume'])
        
        if len(close) < 10:
            return None
            
        # Recent price range
        recent_high = high.iloc[-10:-1].max()
        recent_low = low.iloc[-10:-1].min()
        current_close = close.iloc[-1]
        
        # Volume analysis
        vol_avg = volume.rolling(20).mean().iloc[-1]
        current_vol = volume.iloc[-1]
        
        # 🚨 INSTITUTIONAL BREAKOUT DETECTION
        if (current_close > recent_high and 
            current_vol > vol_avg * 1.8 and
            current_close > close.iloc[-2] and
            close.iloc[-2] > close.iloc[-3]):
            return "CE"
            
        # 🚨 INSTITUTIONAL BREAKDOWN DETECTION  
        if (current_close < recent_low and
            current_vol > vol_avg * 1.8 and
            current_close < close.iloc[-2] and
            close.iloc[-2] < close.iloc[-3]):
            return "PE"
            
        # 🚨 STRONG REJECTION PATTERNS
        current_body = abs(close.iloc[-1] - close.iloc[-2])
        upper_wick = high.iloc[-1] - max(close.iloc[-1], close.iloc[-2])
        lower_wick = min(close.iloc[-1], close.iloc[-2]) - low.iloc[-1]
        
        # Strong rejection at highs
        if (upper_wick > current_body * 1.5 and
            current_vol > vol_avg * 1.5 and
            close.iloc[-1] < close.iloc[-2]):
            return "PE"
            
        # Strong rejection at lows
        if (lower_wick > current_body * 1.5 and
            current_vol > vol_avg * 1.5 and
            close.iloc[-1] > close.iloc[-2]):
            return "CE"
            
    except Exception:
        return None
    return None

# 🚨 NEW: INSTITUTIONAL MOMENTUM CONFIRMATION 🚨
def institutional_momentum_confirmation(index, df, proposed_signal):
    """
    Final institutional confirmation layer
    """
    try:
        close = ensure_series(df['Close'])
        volume = ensure_series(df['Volume'])
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        
        if len(close) < 5:
            return False
            
        # Price momentum confirmation
        if proposed_signal == "CE":
            # For CE: require upward momentum
            if not (close.iloc[-1] > close.iloc[-2] and close.iloc[-2] > close.iloc[-3]):
                return False
            # Strong bullish candle
            if (high.iloc[-1] - low.iloc[-1]) < (high.iloc[-2] - low.iloc[-2]) * 0.7:
                return False
                
        elif proposed_signal == "PE":
            # For PE: require downward momentum
            if not (close.iloc[-1] < close.iloc[-2] and close.iloc[-2] < close.iloc[-3]):
                return False
            # Strong bearish candle
            if (high.iloc[-1] - low.iloc[-1]) < (high.iloc[-2] - low.iloc[-2]) * 0.7:
                return False
                
        return True
        
    except Exception:
        return False

# 🚨 LAYER 1: OPENING-RANGE INSTITUTIONAL PLAY 🚨
def institutional_opening_play(index, df):
    try:
        prev_high = float(ensure_series(df['High']).iloc[-2])
        prev_low = float(ensure_series(df['Low']).iloc[-2])
        prev_close = float(ensure_series(df['Close']).iloc[-2])
        current_price = float(ensure_series(df['Close']).iloc[-1])
    except Exception:
        return None
        
    # STRONGER CONFIRMATION: Require volume confirmation
    volume = ensure_series(df['Volume'])
    vol_avg = volume.rolling(10).mean().iloc[-1] if len(volume) >= 10 else volume.mean()
    vol_ratio = volume.iloc[-1] / (vol_avg if vol_avg > 0 else 1)
    
    if current_price > prev_high + 15 and vol_ratio > 1.3: return "CE"
    if current_price < prev_low - 15 and vol_ratio > 1.3: return "PE"
    if current_price > prev_close + 25 and vol_ratio > 1.2: return "CE"
    if current_price < prev_close - 25 and vol_ratio > 1.2: return "PE"
    return None

# 🚨 LAYER 2: GAMMA SQUEEZE / EXPIRY LAYER 🚨
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
    try:
        close = ensure_series(df['Close']); volume = ensure_series(df['Volume']); 
        high = ensure_series(df['High']); low = ensure_series(df['Low'])
        if len(close) < 6: return None
        
        vol_avg = volume.rolling(20).mean().iloc[-1] if len(volume)>=20 else volume.mean()
        vol_ratio = volume.iloc[-1] / (vol_avg if vol_avg>0 else 1)
        speed = (close.iloc[-1] - close.iloc[-3]) / (abs(close.iloc[-3]) + 1e-6)
        
        try:
            url=f"https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
            df_s = pd.DataFrame(requests.get(url,timeout=10).json())
            df_s['symbol'] = df_s['symbol'].str.upper()
            df_index = df_s[df_s['symbol'].str.contains(index)]
            df_index['oi'] = pd.to_numeric(df_index.get('oi',0), errors='coerce').fillna(0)
            ce_oi = df_index[df_index['symbol'].str.endswith("CE")]['oi'].sum()
            pe_oi = df_index[df_index['symbol'].str.endswith("PE")]['oi'].sum()
        except Exception:
            ce_oi = pe_oi = 0
        
        # STRICTER GAMMA CONDITIONS
        if vol_ratio > GAMMA_VOL_SPIKE_THRESHOLD and abs(speed) > 0.003:
            if speed > 0:
                conf = min(1.0, (vol_ratio - 1.0) / 3.0 + (ce_oi / (pe_oi+1e-6)) * 0.1)
                return {'side':'CE','confidence':conf}
            else:
                conf = min(1.0, (vol_ratio - 1.0) / 3.0 + (pe_oi / (ce_oi+1e-6)) * 0.1)
                return {'side':'PE','confidence':conf}
    except Exception:
        return None
    return None

# 🚨 LAYER 3: SMART-MONEY DIVERGENCE 🚨
def smart_money_divergence(df):
    try:
        close = ensure_series(df['Close']); volume = ensure_series(df['Volume'])
        rsi = ta.momentum.RSIIndicator(close, 14).rsi()
        if len(close) < 10: return None
        
        p_short = close.iloc[-5]; p_now = close.iloc[-1]
        rsi_short = rsi.iloc[-5]; rsi_now = rsi.iloc[-1]
        vol_avg = volume.rolling(20).mean().iloc[-1] if len(volume)>=20 else volume.mean()
        vol_now = volume.iloc[-1]
        
        # STRICTER DIVERGENCE CONDITIONS
        if p_now < p_short and rsi_now > rsi_short + 5 and vol_now > vol_avg*1.3:
            return "CE"
        if p_now > p_short and rsi_now < rsi_short - 5 and vol_now > vol_avg*1.3:
            return "PE"
    except Exception:
        return None
    return None

# 🚨 LAYER 4: STOP-HUNT DETECTOR 🚨
def detect_stop_hunt(df):
    try:
        high = ensure_series(df['High']); low = ensure_series(df['Low']); 
        close = ensure_series(df['Close']); volume = ensure_series(df['Volume'])
        if len(close) < 6: return None
        
        recent_high = high.iloc[-6:-1].max(); recent_low = low.iloc[-6:-1].min()
        last_high = high.iloc[-1]; last_low = low.iloc[-1]; last_close = close.iloc[-1]
        vol_avg = volume.rolling(20).mean().iloc[-1] if len(volume)>=20 else volume.mean()
        
        # STRICTER STOP HUNT CONDITIONS
        if last_high > recent_high * 1.003 and last_close < recent_high and volume.iloc[-1] > vol_avg*1.5:
            return "PE"
        if last_low < recent_low * 0.997 and last_close > recent_low and volume.iloc[-1] > vol_avg*1.5:
            return "CE"
    except Exception:
        return None
    return None

# 🚨 LAYER 5: INSTITUTIONAL CONTINUATION 🚨
def detect_institutional_continuation(df):
    try:
        close = ensure_series(df['Close']); high = ensure_series(df['High']); 
        low = ensure_series(df['Low']); volume = ensure_series(df['Volume'])
        if len(close) < 10: return None
        
        atr = ta.volatility.AverageTrueRange(high, low, close, 14).average_true_range().iloc[-1]
        vol_avg = volume.rolling(20).mean().iloc[-1] if len(volume)>=20 else volume.mean()
        
        speed = (close.iloc[-1] - close.iloc[-3]) / (abs(close.iloc[-3]) + 1e-6)
        
        # STRICTER CONTINUATION CONDITIONS
        if atr > close.std() * 0.8 and volume.iloc[-1] > vol_avg * 1.5 and speed > 0.006:
            return "CE"
        if atr > close.std() * 0.8 and volume.iloc[-1] > vol_avg * 1.5 and speed < -0.006:
            return "PE"
    except Exception:
        return None
    return None

# 🚨 LAYER 6: PULLBACK REVERSAL 🚨
def detect_pullback_reversal(df):
    try:
        close = ensure_series(df['Close'])
        ema9 = ta.trend.EMAIndicator(close, 9).ema_indicator()
        ema21 = ta.trend.EMAIndicator(close, 21).ema_indicator()
        rsi = ta.momentum.RSIIndicator(close, 14).rsi()

        if len(close) < 6:
            return None

        # STRICTER PULLBACK CONDITIONS
        if (close.iloc[-6] > ema21.iloc[-6] and close.iloc[-3] <= ema21.iloc[-3] and 
            close.iloc[-1] > ema9.iloc[-1] and rsi.iloc[-1] > 55 and 
            close.iloc[-1] > close.iloc[-2]):
            return "CE"

        if (close.iloc[-6] < ema21.iloc[-6] and close.iloc[-3] >= ema21.iloc[-3] and 
            close.iloc[-1] < ema9.iloc[-1] and rsi.iloc[-1] < 45 and 
            close.iloc[-1] < close.iloc[-2]):
            return "PE"
    except Exception:
        return None
    return None

# 🚨 LAYER 7: ORDERFLOW MIMIC LOGIC 🚨
def mimic_orderflow_logic(df):
    try:
        close = ensure_series(df['Close']); high = ensure_series(df['High']); 
        low = ensure_series(df['Low']); volume = ensure_series(df['Volume'])
        rsi = ta.momentum.RSIIndicator(close, 14).rsi()

        if len(close) < 4:
            return None

        body = (high - low).abs(); wick_top = (high - close).abs(); wick_bottom = (close - low).abs()
        body_last = body.iloc[-1] if body.iloc[-1] != 0 else 1.0
        wick_top_ratio = wick_top.iloc[-1] / body_last
        wick_bottom_ratio = wick_bottom.iloc[-1] / body_last
        vol_avg = volume.rolling(20).mean().iloc[-1] if len(volume) >= 20 else volume.mean()
        vol_ratio = volume.iloc[-1] / (vol_avg if vol_avg and vol_avg > 0 else 1)

        # STRICTER ORDERFLOW CONDITIONS
        if (close.iloc[-1] > close.iloc[-3] and rsi.iloc[-1] < rsi.iloc[-3] - 3 and 
            wick_top_ratio > 0.7 and vol_ratio > 1.5):
            return "PE"

        if (close.iloc[-1] < close.iloc[-3] and rsi.iloc[-1] > rsi.iloc[-3] + 3 and 
            wick_bottom_ratio > 0.7 and vol_ratio > 1.5):
            return "CE"
    except Exception:
        return None
    return None

# 🚨 LAYER 8: VCP (Volatility Contraction Pattern) 🚨
def detect_vcp_pattern(df):
    try:
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        close = ensure_series(df['Close'])
        volume = ensure_series(df['Volume'])
        
        if len(close) < 10:
            return None
            
        atr = ta.volatility.AverageTrueRange(high, low, close, 5).average_true_range()
        
        recent_atr = atr.iloc[-1]
        prev_atr = atr.iloc[-5]
        
        recent_vol = volume.iloc[-5:].mean()
        prev_vol = volume.iloc[-10:-5].mean()
        
        # STRICTER VCP CONDITIONS
        if (recent_atr < prev_atr * VCP_CONTRACTION_RATIO and 
            recent_vol < prev_vol * 0.8 and
            close.iloc[-1] > close.iloc[-5] and
            volume.iloc[-1] > recent_vol * 1.3):
            return "CE"
        elif (recent_atr < prev_atr * VCP_CONTRACTION_RATIO and 
              recent_vol < prev_vol * 0.8 and
              close.iloc[-1] < close.iloc[-5] and
              volume.iloc[-1] > recent_vol * 1.3):
            return "PE"
    except Exception:
        return None
    return None

# 🚨 LAYER 9: FAULTY BASES 🚨
def detect_faulty_bases(df):
    try:
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        close = ensure_series(df['Close'])
        volume = ensure_series(df['Volume'])
        
        if len(close) < 8:
            return None
            
        recent_high = high.iloc[-8:-3].max()
        recent_low = low.iloc[-8:-3].min()
        current_close = close.iloc[-1]
        
        # STRICTER FAULTY BASE CONDITIONS
        if (high.iloc[-4] > recent_high * (1 + FAULTY_BASE_BREAK_THRESHOLD/100) and
            current_close < recent_high * 0.998 and
            volume.iloc[-4] > volume.iloc[-5:].mean() * 1.4):
            return "PE"
            
        if (low.iloc[-4] < recent_low * (1 - FAULTY_BASE_BREAK_THRESHOLD/100) and
            current_close > recent_low * 1.002 and
            volume.iloc[-4] > volume.iloc[-5:].mean() * 1.4):
            return "CE"
    except Exception:
        return None
    return None

# 🚨 LAYER 10: WYCKOFF SCHEMATICS 🚨
def detect_wyckoff_schematic(df):
    try:
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        close = ensure_series(df['Close'])
        volume = ensure_series(df['Volume'])
        
        if len(close) < 15:
            return None
            
        spring_low = low.iloc[-5]
        support_level = low.iloc[-10:-5].min()
        spring_volume = volume.iloc[-5]
        avg_volume = volume.iloc[-10:].mean()
        
        # STRICTER WYCKOFF CONDITIONS
        if (spring_low < support_level * 0.992 and
            close.iloc[-1] > support_level * 1.005 and
            spring_volume > avg_volume * WYCKOFF_VOLUME_SPRING and
            volume.iloc[-1] > avg_volume * 1.2):
            return "CE"
            
        upthrust_high = high.iloc[-5]
        resistance_level = high.iloc[-10:-5].max()
        upthrust_volume = volume.iloc[-5]
        
        if (upthrust_high > resistance_level * 1.008 and
            close.iloc[-1] < resistance_level * 0.995 and
            upthrust_volume > avg_volume * WYCKOFF_VOLUME_SPRING and
            volume.iloc[-1] > avg_volume * 1.2):
            return "PE"
    except Exception:
        return None
    return None

# 🚨 LAYER 11: LIQUIDITY SWEEPS 🚨
def detect_liquidity_sweeps(df):
    try:
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        close = ensure_series(df['Close'])
        volume = ensure_series(df['Volume'])
        
        if len(close) < 10:
            return None
            
        recent_highs = high.iloc[-10:-2]
        recent_lows = low.iloc[-10:-2]
        
        liquidity_high = recent_highs.max()
        liquidity_low = recent_lows.min()
        
        current_high = high.iloc[-1]
        current_low = low.iloc[-1]
        current_close = close.iloc[-1]
        
        # STRICTER LIQUIDITY SWEEP CONDITIONS
        if (current_high > liquidity_high * (1 + LIQUIDITY_SWEEP_DISTANCE) and
            current_close < liquidity_high * 0.998 and
            volume.iloc[-1] > volume.iloc[-10:-1].mean() * 1.6):
            return "PE"
            
        if (current_low < liquidity_low * (1 - LIQUIDITY_SWEEP_DISTANCE) and
            current_close > liquidity_low * 1.002 and
            volume.iloc[-1] > volume.iloc[-10:-1].mean() * 1.6):
            return "CE"
    except Exception:
        return None
    return None

# 🚨 LAYER 12: PEAK REJECTION 🚨
def detect_peak_rejection(df):
    try:
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        close = ensure_series(df['Close'])
        volume = ensure_series(df['Volume'])
        
        if len(close) < 5:
            return None
            
        current_high = high.iloc[-1]
        current_low = low.iloc[-1]
        current_close = close.iloc[-1]
        
        body_size = abs(current_close - close.iloc[-2])
        upper_wick = current_high - max(close.iloc[-1], close.iloc[-2])
        lower_wick = min(close.iloc[-1], close.iloc[-2]) - current_low
        
        # STRICTER PEAK REJECTION CONDITIONS
        if (upper_wick > body_size * PEAK_REJECTION_WICK_RATIO and
            current_close < (current_high + current_low) / 2 * 0.995 and
            volume.iloc[-1] > volume.iloc[-5:].mean() * 1.3):
            return "PE"
            
        if (lower_wick > body_size * PEAK_REJECTION_WICK_RATIO and
            current_close > (current_high + current_low) / 2 * 1.005 and
            volume.iloc[-1] > volume.iloc[-5:].mean() * 1.3):
            return "CE"
    except Exception:
        return None
    return None

# 🚨 LAYER 13: FAIR VALUE GAP (FVG) 🚨
def detect_fair_value_gap(df):
    try:
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        close = ensure_series(df['Close'])
        
        if len(close) < 3:
            return None
            
        # STRICTER FVG CONDITIONS
        if (low.iloc[-1] > high.iloc[-2] * (1 + FVG_GAP_THRESHOLD) and
            close.iloc[-1] > close.iloc[-2] and
            close.iloc[-1] > (high.iloc[-2] + low.iloc[-2]) / 2):
            return "CE"
            
        if (high.iloc[-1] < low.iloc[-2] * (1 - FVG_GAP_THRESHOLD) and
            close.iloc[-1] < close.iloc[-2] and
            close.iloc[-1] < (high.iloc[-2] + low.iloc[-2]) / 2):
            return "PE"
    except Exception:
        return None
    return None

# 🚨 LAYER 14: VOLUME GAP IMBALANCE 🚨
def detect_volume_gap_imbalance(df):
    try:
        volume = ensure_series(df['Volume'])
        close = ensure_series(df['Close'])
        
        if len(volume) < 20:
            return None
            
        current_volume = volume.iloc[-1]
        avg_volume = volume.iloc[-20:].mean()
        price_change = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]
        
        # STRICTER VOLUME GAP CONDITIONS
        if (current_volume > avg_volume * VOLUME_GAP_IMBALANCE and
            abs(price_change) > 0.004):  # Increased from 0.002
            if price_change > 0:
                return "CE"
            else:
                return "PE"
    except Exception:
        return None
    return None

# 🚨 LAYER 15: OTE (Optimal Trade Entry) 🚨
def detect_ote_retracement(df):
    try:
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        close = ensure_series(df['Close'])
        
        if len(close) < 15:
            return None
            
        swing_high = high.iloc[-15:-5].max()
        swing_low = low.iloc[-15:-5].min()
        swing_range = swing_high - swing_low
        
        current_price = close.iloc[-1]
        
        for level in OTE_RETRACEMENT_LEVELS:
            ote_level = swing_high - (swing_range * level)
            
            if (abs(current_price - ote_level) / ote_level < 0.0015 and  # Tighter tolerance
                close.iloc[-1] > close.iloc[-2] and
                close.iloc[-1] > close.iloc[-3]):
                return "CE"
                
            ote_level = swing_low + (swing_range * level)
            if (abs(current_price - ote_level) / ote_level < 0.0015 and  # Tighter tolerance
                close.iloc[-1] < close.iloc[-2] and
                close.iloc[-1] < close.iloc[-3]):
                return "PE"
    except Exception:
        return None
    return None

# 🚨 LAYER 16: DEMAND AND SUPPLY ZONES 🚨
def detect_demand_supply_zones(df):
    try:
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        close = ensure_series(df['Close'])
        volume = ensure_series(df['Volume'])
        
        if len(close) < DEMAND_SUPPLY_ZONE_LOOKBACK + 5:
            return None
            
        lookback = DEMAND_SUPPLY_ZONE_LOOKBACK
        
        demand_lows = low.rolling(3, center=True).min().dropna()
        significant_demand = demand_lows[demand_lows == demand_lows.rolling(5).min()]
        
        supply_highs = high.rolling(3, center=True).max().dropna()
        significant_supply = supply_highs[supply_highs == supply_highs.rolling(5).max()]
        
        current_price = close.iloc[-1]
        
        # STRICTER ZONE CONDITIONS
        for zone in significant_demand.iloc[-5:]:
            if (abs(current_price - zone) / zone < 0.002 and  # Tighter tolerance
                close.iloc[-1] > close.iloc[-2] and
                close.iloc[-1] > close.iloc[-3] and  # Additional confirmation
                volume.iloc[-1] > volume.iloc[-5:].mean() * 1.4):
                return "CE"
                
        for zone in significant_supply.iloc[-5:]:
            if (abs(current_price - zone) / zone < 0.002 and  # Tighter tolerance
                close.iloc[-1] < close.iloc[-2] and
                close.iloc[-1] < close.iloc[-3] and  # Additional confirmation
                volume.iloc[-1] > volume.iloc[-5:].mean() * 1.4):
                return "PE"
    except Exception:
        return None
    return None

# 🚨 LAYER 17: BOTTOM-FISHING 🚨
def detect_bottom_fishing(index, df):
    try:
        close = ensure_series(df['Close'])
        low = ensure_series(df['Low'])
        high = ensure_series(df['High'])
        volume = ensure_series(df['Volume'])
        if len(close) < 6: 
            return None

        bull_liq, bear_liq = institutional_liquidity_hunt(index, df)
        last_close = float(close.iloc[-1])

        wick = last_close - low.iloc[-1]
        body = abs(close.iloc[-1] - close.iloc[-2])
        vol_avg = volume.rolling(20).mean().iloc[-1] if len(volume) >= 20 else volume.mean()
        vol_ratio = volume.iloc[-1] / (vol_avg if vol_avg > 0 else 1)

        # STRICTER BOTTOM FISHING CONDITIONS
        if wick > body * 2.0 and vol_ratio > 1.5:  # Increased ratios
            for zone in bull_liq:
                if zone and abs(last_close - zone) <= 3:  # Tighter zone
                    return "CE"

        bear_wick = high.iloc[-1] - last_close
        if bear_wick > body * 2.0 and vol_ratio > 1.5:  # Increased ratios
            for zone in bear_liq:
                if zone and abs(last_close - zone) <= 3:  # Tighter zone
                    return "PE"
    except:
        return None
    return None

# --------- UPDATED STRATEGY CHECK WITH INSTITUTIONAL LAYERS ---------
def analyze_index_signal(index):
    df5 = fetch_index_data(index, "5m", "2d")
    if df5 is None:
        return None

    close5 = ensure_series(df5["Close"])
    if len(close5) < 20 or close5.isna().iloc[-1] or close5.isna().iloc[-2]:
        return None

    last_close = float(close5.iloc[-1])
    prev_close = float(close5.iloc[-2])

    # 🚨 NEW: TIME-BASED FILTER - Avoid late day unreliable signals
    try:
        utc_now = datetime.utcnow()
        ist_now = utc_now + timedelta(hours=5, minutes=30)
        current_time = ist_now.time()
        # Avoid signals in last 45 minutes (low reliability)
        if current_time >= dtime(14, 45):
            return None
    except:
        pass

    # 🚨 ALL STRATEGIES RUN SIMULTANEOUSLY - NO PRIORITY 🚨
    strategies = [
        ("INSTITUTIONAL_PRICE_ACTION", institutional_price_action_signal(df5)),
        ("OPENING_PLAY", institutional_opening_play(index, df5)),
        ("GAMMA_SQUEEZE", detect_gamma_squeeze(index, df5)),
        ("SMART_MONEY_DIVERGENCE", smart_money_divergence(df5)),
        ("STOP_HUNT", detect_stop_hunt(df5)),
        ("INSTITUTIONAL_CONTINUATION", detect_institutional_continuation(df5)),
        ("PULLBACK_REVERSAL", detect_pullback_reversal(df5)),
        ("ORDERFLOW_MIMIC", mimic_orderflow_logic(df5)),
        ("VCP_PATTERN", detect_vcp_pattern(df5)),
        ("FAULTY_BASES", detect_faulty_bases(df5)),
        ("WYCKOFF_SCHEMATIC", detect_wyckoff_schematic(df5)),
        ("LIQUIDITY_SWEEPS", detect_liquidity_sweeps(df5)),
        ("PEAK_REJECTION", detect_peak_rejection(df5)),
        ("FAIR_VALUE_GAP", detect_fair_value_gap(df5)),
        ("VOLUME_GAP_IMBALANCE", detect_volume_gap_imbalance(df5)),
        ("OTE_RETRACEMENT", detect_ote_retracement(df5)),
        ("DEMAND_SUPPLY_ZONES", detect_demand_supply_zones(df5)),
        ("BOTTOM_FISHING", detect_bottom_fishing(index, df5))
    ]

    # Collect all valid signals
    valid_signals = []
    for strategy_name, signal in strategies:
        if signal:
            if isinstance(signal, dict):  # Gamma squeeze returns dict
                signal = signal.get('side')
            if signal in ["CE", "PE"]:
                if institutional_momentum_confirmation(index, df5, signal):
                    valid_signals.append((strategy_name, signal))

    # Return first valid signal (all strategies have equal priority)
    if valid_signals:
        strategy_name, signal = valid_signals[0]
        return signal, df5, False, strategy_name  # 🚨 ADDED STRATEGY NAME

    return None

# --------- SYMBOL FORMAT FOR ALL INDICES ---------
def get_option_symbol(index, expiry_str, strike, opttype):
    dt=datetime.strptime(expiry_str,"%d %b %Y")
    
    if index == "SENSEX":
        year_short = dt.strftime("%y")
        month_code = dt.strftime("%b").upper()
        day = dt.strftime("%d")
        return f"SENSEX{year_short}{month_code}{strike}{opttype}"
    elif index == "FINNIFTY":
        return f"FINNIFTY{dt.strftime('%d%b%y').upper()}{strike}{opttype}"
    elif index == "MIDCPNIFTY":
        return f"MIDCPNIFTY{dt.strftime('%d%b%y').upper()}{strike}{opttype}"
    else:
        return f"{index}{dt.strftime('%d%b%y').upper()}{strike}{opttype}"

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

    # STRICTER FLOW CONDITIONS
    if vol_latest > vol_avg*2.0 and abs(last_close-prev_close)/prev_close>0.005:
        return "BOTH"
    elif last_close>prev_close and vol_latest>vol_avg*1.5:
        return "CE"
    elif last_close<prev_close and vol_latest>vol_avg*1.5:
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
        df=df[df['exch_seg'].str.upper().isin(["NFO", "BFO"])]
        df['symbol']=df['symbol'].str.upper()
        df_index=df[df['symbol'].str.contains(index)]
        if 'oi' not in df_index.columns:
            return None
        df_index['oi'] = pd.to_numeric(df_index['oi'], errors='coerce').fillna(0)
        df_index['oi_change'] = df_index['oi'].diff().fillna(0)
        ce_sum = df_index[df_index['symbol'].str.endswith("CE")]['oi_change'].sum()
        pe_sum = df_index[df_index['symbol'].str.endswith("PE")]['oi_change'].sum()
        # STRICTER OI CONDITIONS
        if ce_sum>pe_sum*DELTA_OI_RATIO: return "CE"
        if pe_sum>ce_sum*DELTA_OI_RATIO: return "PE"
        if ce_sum>0 and pe_sum>0: return "BOTH"
    except:
        return None

# --------- SIMPLIFIED CONFIRMATION ---------
def institutional_confirmation_layer(index, df5, base_signal):
    try:
        close = ensure_series(df5['Close'])
        last_close = float(close.iloc[-1])
        
        high_zone, low_zone = detect_liquidity_zone(df5, lookback=20)
        if base_signal == 'CE' and last_close >= high_zone:
            return False
        if base_signal == 'PE' and last_close <= low_zone:
            return False

        return True
    except Exception:
        return False

def institutional_flow_confirm(index, base_signal, df5):
    flow = institutional_flow_signal(index, df5)
    oi_flow = oi_delta_flow_signal(index)

    if flow and flow != 'BOTH' and flow != base_signal:
        return False
    if oi_flow and oi_flow != 'BOTH' and oi_flow != base_signal:
        return False

    if not institutional_confirmation_layer(index, df5, base_signal):
        return False

    return True

# --------- TRADE MONITORING AND TRACKING ---------
active_trades = {}

def monitor_price_live(symbol, entry, targets, sl, fakeout, thread_id, strategy_name, signal_data):
    """Run monitoring in separate thread without blocking main signal generation"""
    def monitoring_thread():
        global daily_signals
        
        last_high = entry
        weakness_sent = False
        in_trade = False
        entry_price_achieved = False
        max_price_reached = entry
        targets_hit = [False] * len(targets)
        
        while True:
            if should_stop_trading():
                # Update signal data before stopping
                signal_data.update({
                    "entry_status": "NOT_ENTERED" if not entry_price_achieved else "ENTERED",
                    "targets_hit": sum(targets_hit),
                    "max_price_reached": max_price_reached,
                    "zero_targets": sum(targets_hit) == 0,
                    "no_new_highs": max_price_reached <= entry,
                    "final_pnl": calculate_pnl(entry, max_price_reached, targets_hit, sl)
                })
                daily_signals.append(signal_data)
                send_telegram(f"🛑 Market closed - Stopping monitoring for {symbol}", reply_to=thread_id)
                break
                
            price = fetch_option_price(symbol)
            if not price: 
                time.sleep(10)
                continue
                
            price = round(price)
            
            # Track max price reached
            if price > max_price_reached:
                max_price_reached = price
            
            if not in_trade:
                if price >= entry:
                    send_telegram(f"✅ ENTRY TRIGGERED at {price}", reply_to=thread_id)
                    in_trade = True
                    entry_price_achieved = True
                    last_high = price
                    # Update signal data
                    signal_data["entry_status"] = "ENTERED"
            else:
                if price > last_high:
                    send_telegram(f"🚀 {symbol} making new high → {price}", reply_to=thread_id)
                    last_high = price
                elif not weakness_sent and price < sl * 1.05:
                    send_telegram(f"⚡ {symbol} showing weakness near SL {sl}", reply_to=thread_id)
                    weakness_sent = True
                
                # Check all targets
                for i, target in enumerate(targets):
                    if price >= target and not targets_hit[i]:
                        send_telegram(f"🎯 {symbol}: Target {i+1} hit at ₹{target}", reply_to=thread_id)
                        targets_hit[i] = True
                
                if price <= sl:
                    send_telegram(f"🔗 {symbol}: Stop Loss {sl} hit. Exit trade.", reply_to=thread_id)
                    # Update final signal data
                    signal_data.update({
                        "targets_hit": sum(targets_hit),
                        "max_price_reached": max_price_reached,
                        "zero_targets": sum(targets_hit) == 0,
                        "no_new_highs": max_price_reached <= entry,
                        "final_pnl": calculate_pnl(entry, max_price_reached, targets_hit, sl)
                    })
                    daily_signals.append(signal_data)
                    break
                    
                # If all targets hit, exit
                if all(targets_hit):
                    send_telegram(f"🏆 {symbol}: ALL TARGETS HIT! Trade completed successfully!", reply_to=thread_id)
                    # Update final signal data
                    signal_data.update({
                        "targets_hit": len(targets),
                        "max_price_reached": max_price_reached,
                        "zero_targets": False,
                        "no_new_highs": False,
                        "final_pnl": calculate_pnl(entry, max_price_reached, targets_hit, sl)
                    })
                    daily_signals.append(signal_data)
                    break
            
            time.sleep(10)
    
    # Start monitoring in separate thread
    thread = threading.Thread(target=monitoring_thread)
    thread.daemon = True
    thread.start()

def calculate_pnl(entry, max_price, targets_hit, sl):
    """Calculate P&L based on targets hit and max price reached"""
    if max_price <= sl:
        return f"-{entry - sl}"
    
    targets_achieved = sum(targets_hit)
    if targets_achieved == 0:
        if max_price > entry:
            return f"+{max_price - entry}"
        else:
            return "0"
    
    # Calculate average target price for achieved targets
    achieved_prices = [target for i, target in enumerate(targets) if targets_hit[i]]
    avg_exit = sum(achieved_prices) / len(achieved_prices)
    return f"+{avg_exit - entry}"

# --------- NEW READABLE END OF DAY REPORT ---------
def generate_end_of_day_report():
    """Generate comprehensive end-of-day report in readable format"""
    if not daily_signals:
        return "📊 END OF DAY REPORT\nNo signals generated today."
    
    report = "📊 GIT ULTIMATE MASTER - END OF DAY REPORT\n\n"
    report += f"📅 Date: {datetime.now().strftime('%d-%b-%Y')}\n"
    report += f"📈 Total Signals: {len(daily_signals)}\n"
    report += "=" * 50 + "\n\n"
    
    total_pnl = 0
    successful_trades = 0
    
    for i, signal in enumerate(daily_signals, 1):
        report += f"🔰 SIGNAL #{i}\n"
        report += f"   • Timestamp: {signal['timestamp']}\n"
        report += f"   • Index: {signal['index']}\n"
        report += f"   • Strike: {signal['strike']}\n"
        report += f"   • Type: {signal['option_type']}\n"
        report += f"   • Strategy: {signal['strategy']}\n"
        report += f"   • Entry Price: ₹{signal['entry_price']}\n"
        report += f"   • Target 1: ₹{signal['targets'][0]}\n"
        report += f"   • Target 2: ₹{signal['targets'][1]}\n"
        report += f"   • Target 3: ₹{signal['targets'][2]}\n"
        report += f"   • Target 4: ₹{signal['targets'][3]}\n"
        report += f"   • Stop Loss: ₹{signal['sl']}\n"
        report += f"   • Fakeout: {signal['fakeout']}\n"
        report += f"   • Index Price: {signal['index_price']}\n"
        report += f"   • Entry Status: {signal.get('entry_status', 'PENDING')}\n"
        report += f"   • Targets Hit: {signal.get('targets_hit', 0)}/4\n"
        report += f"   • Max Price Reached: ₹{signal.get('max_price_reached', signal['entry_price'])}\n"
        report += f"   • Zero Targets Hit: {'Yes' if signal.get('zero_targets', True) else 'No'}\n"
        report += f"   • No New Highs: {'Yes' if signal.get('no_new_highs', True) else 'No'}\n"
        report += f"   • Final P&L: {signal.get('final_pnl', '0')}\n"
        report += f"   • Signal ID: {signal['signal_id']}\n"
        report += "-" * 40 + "\n\n"
        
        # Calculate P&L for summary
        pnl_str = signal.get("final_pnl", "0")
        try:
            if pnl_str.startswith("+"):
                total_pnl += float(pnl_str[1:])
                successful_trades += 1
            elif pnl_str.startswith("-"):
                total_pnl -= float(pnl_str[1:])
        except:
            pass
    
    # Summary Section
    report += "📈 SUMMARY\n"
    report += "=" * 30 + "\n"
    report += f"• Total Signals: {len(daily_signals)}\n"
    report += f"• Successful Trades: {successful_trades}\n"
    report += f"• Success Rate: {(successful_trades/len(daily_signals))*100:.1f}%\n"
    report += f"• Total P&L: ₹{total_pnl:+.2f}\n\n"
    
    # Strategy Performance
    strategy_stats = {}
    for signal in daily_signals:
        strat = signal['strategy']
        if strat not in strategy_stats:
            strategy_stats[strat] = {'count': 0, 'success': 0}
        strategy_stats[strat]['count'] += 1
        pnl = signal.get("final_pnl", "0")
        if pnl.startswith("+"):
            strategy_stats[strat]['success'] += 1
    
    report += "🎯 STRATEGY PERFORMANCE\n"
    report += "=" * 30 + "\n"
    for strat, stats in strategy_stats.items():
        success_rate = (stats['success']/stats['count'])*100 if stats['count'] > 0 else 0
        report += f"• {strat}: {stats['count']} signals, {success_rate:.1f}% success rate\n"
    
    return report

# --------- UPDATED SIGNAL SENDING WITH STRATEGY TRACKING ---------
def send_signal(index, side, df, fakeout, strategy_key):
    global signal_counter
    
    # Get ACTUAL index price where pattern was detected
    signal_detection_price = float(ensure_series(df["Close"]).iloc[-1])
    
    # Calculate strike based on ACTUAL detection price
    strike = round_strike(index, signal_detection_price)
    
    if strike is None:
        send_telegram(f"⚠️ {index}: could not determine strike (price missing). Signal skipped.")
        return
        
    symbol = get_option_symbol(index, EXPIRIES[index], strike, side)
    
    # Get ACTUAL option price for that strike
    option_price = fetch_option_price(symbol)
    if not option_price: 
        return
    
    # 🚨 INSTITUTIONAL ENTRY: Use actual detected price (no fixed +5)
    entry = round(option_price)
    
    # Calculate ATR for risk management
    high = ensure_series(df["High"])
    low = ensure_series(df["Low"])
    close = ensure_series(df["Close"])
    atr = float(ta.volatility.AverageTrueRange(high, low, close, 14).average_true_range().iloc[-1])
    
    # 🚨 CONFIRMED SMALL TARGETS
    atr_multiplier = 0.3
    base_target = option_price
    
    # Progressive small targets
    targets = [
        round(base_target + (atr * atr_multiplier * 1.0)),
        round(base_target + (atr * atr_multiplier * 1.5)),
        round(base_target + (atr * atr_multiplier * 2.2)),
        round(base_target + (atr * atr_multiplier * 3.0))
    ]
    
    # Stop Loss
    sl = round(option_price - (atr * 0.8))
    
    # Format targets
    targets_str = "//".join(str(t) for t in targets) + "++"
    
    # Get strategy name
    strategy_name = STRATEGY_NAMES.get(strategy_key, strategy_key.upper())
    
    # Create signal data for tracking
    signal_id = f"SIG{signal_counter:04d}"
    signal_counter += 1
    
    signal_data = {
        "signal_id": signal_id,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "index": index,
        "strike": strike,
        "option_type": side,  # 🚨 NOW INCLUDES CE/PE
        "strategy": strategy_name,
        "entry_price": entry,
        "targets": targets,
        "sl": sl,
        "fakeout": fakeout,
        "index_price": signal_detection_price,
        "entry_status": "PENDING",
        "targets_hit": 0,
        "max_price_reached": entry,
        "zero_targets": True,
        "no_new_highs": True,
        "final_pnl": "0"
    }
    
    # 🚨 FIXED SIGNAL FORMAT - NOW INCLUDES CE/PE
    msg = (f"{index} {strike} {side}\n"  # 🚨 ADDED CE/PE
           f"ABOVE {entry}\n"
           f"TARGETS: {targets_str}\n"
           f"SL: {sl}\n"
           f"FAKEOUT: {'YES' if fakeout else 'NO'}\n"
           f"STRATEGY: {strategy_name}\n"  # 🚨 ADDED STRATEGY
           f"SIGNAL ID: {signal_id}")  # 🚨 ADDED SIGNAL ID
         
    thread_id = send_telegram(msg)
    
    # Store trade info
    trade_id = f"{symbol}_{int(time.time())}"
    active_trades[trade_id] = {
        "symbol": symbol, 
        "entry": entry, 
        "sl": sl, 
        "targets": targets, 
        "thread": thread_id, 
        "status": "OPEN",
        "index": index,
        "signal_data": signal_data
    }
    
    # Start monitoring in SEPARATE thread
    monitor_price_live(symbol, entry, targets, sl, fakeout, thread_id, strategy_name, signal_data)

# --------- UPDATED TRADE THREAD ---------
def trade_thread(index):
    """Generate signals without blocking on active trades"""
    result = analyze_index_signal(index)
    
    if not result:
        return
        
    # Unpack result with strategy
    if len(result) == 4:
        side, df, fakeout, strategy_key = result  # 🚨 NOW GETS STRATEGY
    else:
        # Fallback for old format
        side, df, fakeout = result
        strategy_key = "UNKNOWN"
    
    df5 = fetch_index_data(index, "5m", "2d")
    inst_signal = institutional_flow_signal(index, df5) if df5 is not None else None
    oi_signal = oi_delta_flow_signal(index)
    final_signal = oi_signal or inst_signal or side

    if final_signal == "BOTH":
        for s in ["CE", "PE"]:
            if institutional_flow_confirm(index, s, df5):
                send_signal(index, s, df, fakeout, strategy_key)
        return
    elif final_signal:
        if df is None: 
            df = df5
        if institutional_flow_confirm(index, final_signal, df5):
            send_signal(index, final_signal, df, fakeout, strategy_key)
    else:
        return

# --------- MAIN LOOP (ALL INDICES PARALLEL) ---------
def run_algo_parallel():
    if not is_market_open(): 
        print("❌ Market closed - skipping iteration")
        return
        
    if should_stop_trading():
        global STOP_SENT
        if not STOP_SENT:
            # Generate end of day report
            report = generate_end_of_day_report()
            send_telegram(report)
            send_telegram("🛑 Market closed at 3:30 PM IST - Algorithm stopped")
            STOP_SENT = True
        return
        
    threads = []
    all_indices = ["NIFTY", "BANKNIFTY", "SENSEX", "FINNIFTY", "MIDCPNIFTY", "EICHERMOT", "TRENT", "RELIANCE"]
    
    for index in all_indices:
        t = threading.Thread(target=trade_thread, args=(index,))
        t.start()
        threads.append(t)
    
    for t in threads: 
        t.join()

# --------- START ---------
STARTED_SENT = False
STOP_SENT = False
MARKET_CLOSED_SENT = False

while True:
    try:
        # Check if market is open
        market_open = is_market_open()
        
        # Market CLOSED behavior - send only ONE message
        if not market_open:
            if not MARKET_CLOSED_SENT and not STARTED_SENT:
                send_telegram("🔴 Market is currently closed. Algorithm waiting for 9:15 AM...")
                MARKET_CLOSED_SENT = True
                STOP_SENT = False
            
            # Just sleep, don't send repeated messages
            time.sleep(30)
            continue
        
        # Market OPEN behavior - original logic
        if not STARTED_SENT:
            send_telegram("🚀 GIT ULTIMATE MASTER ALGO STARTED - All 8 Indices Running with COMPLETE REPORTING:\n"
                         "✅ CE/PE Identification in Every Signal\n"
                         "✅ Strategy Name Tracking\n"  
                         "✅ End-of-Day Performance Report\n"
                         "✅ Max Price Reached Tracking\n"
                         "✅ Real-time Trade Monitoring")
            STARTED_SENT = True
            STOP_SENT = False
            MARKET_CLOSED_SENT = False  # Reset for next day
            
        if should_stop_trading():
            if not STOP_SENT:
                # Generate final report before stopping
                report = generate_end_of_day_report()
                send_telegram(report)
                send_telegram("🛑 Market closing time reached - Algorithm stopped automatically")
                STOP_SENT = True
                STARTED_SENT = False
            # Don't break, just sleep until next day
            time.sleep(60)
            continue
            
        # Run the main algorithm
        run_algo_parallel()
        time.sleep(30)
        
    except Exception as e:
        send_telegram(f"⚠️ Error in main loop: {e}")
        time.sleep(60)

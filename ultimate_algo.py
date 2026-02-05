#INDEXBASED + EOD NOT COMMING - FIXED VERSION

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

# NEW: INSTITUTIONAL SUPPORT/RESISTANCE CONFIG
KEY_LEVEL_DAYS = [1, 5, 20]  # 1 day, 5 days, 1 month (approx 20 trading days)
KEY_LEVEL_TOLERANCE = 0.003  # 0.3% tolerance for key levels
RETEST_CONFIRMATION_BARS = 3  # Number of bars to confirm retest
BREAKOUT_CONFIRMATION_VOLUME = 1.8  # Volume multiplier for breakout confirmation
HAMMER_WICK_TO_BODY_RATIO = 2.0  # Minimum wick-to-body ratio for hammer patterns

# --------- EXPIRIES FOR KEPT INDICES ---------
EXPIRIES = {
    "NIFTY": "10 FEB 2026",
    "BANKNIFTY": "24 FEB 2026", 
    "SENSEX": "12 FEB 2026",
    "MIDCPNIFTY": "24 FEB 2026"
}

# --------- STRATEGY TRACKING ---------
# 🚨 KEEP ONLY 2 STRATEGIES AS REQUESTED 🚨
STRATEGY_NAMES = {
    "liquidity_zone": "LIQUIDITY ZONE",
    "ote_retracement": "OTE RETRACEMENT"
}

# --------- ENHANCED TRACKING FOR REPORTS ---------
all_generated_signals = []  # Track ALL signals for EOD reporting
strategy_performance = {}
signal_counter = 0
daily_signals = []

# --------- NEW: SIGNAL DEDUPLICATION AND COOLDOWN TRACKING ---------
active_strikes = {}  # Track active strikes to prevent duplicates
last_signal_time = {}  # Track last signal time per index
signal_cooldown = 1200  # 20 minutes in seconds

def initialize_strategy_tracking():
    """Initialize strategy performance tracking"""
    global strategy_performance
    # 🚨 ONLY 2 STRATEGIES NOW 🚨
    strategy_performance = {
        "LIQUIDITY ZONE": {"total": 0, "success_2_targets": 0, "success_3_4_targets": 0, "total_pnl": 0},
        "OTE RETRACEMENT": {"total": 0, "success_2_targets": 0, "success_3_4_targets": 0, "total_pnl": 0}
    }

# Initialize tracking
initialize_strategy_tracking()

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
MARKET_CLOSED_SENT = False
EOD_REPORT_SENT = False

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

# --------- STRIKE ROUNDING FOR KEPT INDICES ---------
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
        elif index == "MIDCPNIFTY": 
            return int(round(price / 25.0) * 25)
        else: 
            return int(round(price / 50.0) * 50)
    except Exception:
        return None

# --------- ENSURE SERIES ---------
def ensure_series(data):
    return data.iloc[:,0] if isinstance(data, pd.DataFrame) else data.squeeze()

# --------- FETCH INDEX DATA FOR KEPT INDICES ---------
def fetch_index_data(index, interval="5m", period="2d"):
    symbol_map = {
        "NIFTY": "^NSEI", 
        "BANKNIFTY": "^NSEBANK", 
        "SENSEX": "^BSESN",
        "MIDCPNIFTY": "NIFTY_MID_SELECT.NS"
    }
    df = yf.download(symbol_map[index], period=period, interval=interval, auto_adjust=True, progress=False)
    return None if df.empty else df

# --------- FETCH LONGER TIME FRAME DATA FOR KEY LEVELS ---------
def fetch_key_level_data(index, days):
    """Fetch data for key level analysis (daily timeframe)"""
    symbol_map = {
        "NIFTY": "^NSEI", 
        "BANKNIFTY": "^NSEBANK", 
        "SENSEX": "^BSESN",
        "MIDCPNIFTY": "NIFTY_MID_SELECT.NS"
    }
    
    # Convert days to period string for yfinance
    if days <= 5:
        period = "5d"
    elif days <= 20:
        period = "1mo"
    elif days <= 60:
        period = "3mo"
    else:
        period = "6mo"
    
    try:
        df = yf.download(symbol_map[index], period=period, interval="1d", auto_adjust=True, progress=False)
        return None if df.empty else df
    except:
        return None

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

# 🚨 FIXED: STRICT EXPIRY VALIDATION FUNCTIONS 🚨
def validate_option_symbol(index, symbol, strike, opttype):
    """STRICT validation to ensure ONLY specified expiry symbols are used"""
    try:
        # Get the expected expiry for this index
        expected_expiry = EXPIRIES.get(index)
        if not expected_expiry:
            return False
            
        # Parse expected expiry date
        expected_dt = datetime.strptime(expected_expiry, "%d %b %Y")
        
        # STRICT CHECK: For SENSEX: SENSEX25NOV25000CE format
        if index == "SENSEX":
            year_short = expected_dt.strftime("%y")  # 25
            month_code = expected_dt.strftime("%b").upper()  # NOV
            day = expected_dt.strftime("%d")  # 25
            expected_pattern = f"SENSEX{day}{month_code}{year_short}"
            symbol_upper = symbol.upper()
            
            # Check if symbol contains EXACTLY this pattern
            if expected_pattern in symbol_upper:
                return True
            else:
                print(f"❌ SENSEX expiry mismatch: Expected {expected_pattern}, Got {symbol_upper}")
                return False
        else:
            # STRICT CHECK: For NIFTY/BANKNIFTY/MIDCPNIFTY: NIFTY25NOV2521500CE format
            expected_pattern = expected_dt.strftime("%d%b%y").upper()  # 25NOV25
            symbol_upper = symbol.upper()
            
            # Check if symbol contains EXACTLY this pattern
            if expected_pattern in symbol_upper:
                return True
            else:
                print(f"❌ {index} expiry mismatch: Expected {expected_pattern}, Got {symbol_upper}")
                return False
                
    except Exception as e:
        print(f"Symbol validation error: {e}")
        return False

# 🚨 FIXED: GET OPTION SYMBOL WITH STRICT EXPIRY VALIDATION 🚨
def get_option_symbol(index, expiry_str, strike, opttype):
    """STRICTLY generates symbols ONLY with specified expiries"""
    try:
        dt = datetime.strptime(expiry_str, "%d %b %Y")
        
        if index == "SENSEX":
            year_short = dt.strftime("%y")  # 25
            month_code = dt.strftime("%b").upper()  # NOV
            day = dt.strftime("%d")  # 25
            symbol = f"SENSEX{day}{month_code}{year_short}{strike}{opttype}"
        elif index == "MIDCPNIFTY":
            symbol = f"MIDCPNIFTY{dt.strftime('%d%b%y').upper()}{strike}{opttype}"
        else:
            symbol = f"{index}{dt.strftime('%d%b%y').upper()}{strike}{opttype}"
        
        # STRICT VALIDATION: Validate the generated symbol
        if validate_option_symbol(index, symbol, strike, opttype):
            print(f"✅ Valid symbol generated: {symbol}")
            return symbol
        else:
            print(f"❌ Generated symbol validation FAILED: {symbol}")
            return None
            
    except Exception as e:
        print(f"Error generating symbol: {e}")
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

# 🚨 NEW: INSTITUTIONAL SUPPORT/RESISTANCE ZONE ANALYSIS 🚨
def analyze_institutional_key_levels(index, current_price):
    """
    Analyze key support/resistance levels from 1 day, 5 days, and 1 month
    Returns: dict with support/resistance levels and retest analysis
    """
    key_levels = {
        'support_levels': [],
        'resistance_levels': [],
        'nearest_support': None,
        'nearest_resistance': None,
        'support_retest': False,
        'resistance_retest': False,
        'breakout_trap': False,
        'hammer_at_resistance': False
    }
    
    try:
        for days in KEY_LEVEL_DAYS:
            df_daily = fetch_key_level_data(index, days)
            if df_daily is None or len(df_daily) < 10:
                continue
                
            close = ensure_series(df_daily['Close'])
            high = ensure_series(df_daily['High'])
            low = ensure_series(df_daily['Low'])
            
            # Identify recent swing highs and lows
            if len(high) >= 5:
                recent_highs = high.iloc[-5:]
                recent_lows = low.iloc[-5:]
                
                # Significant highs (potential resistance)
                for level in recent_highs:
                    if abs(current_price - level) / level <= KEY_LEVEL_TOLERANCE:
                        key_levels['resistance_levels'].append(level)
                        if key_levels['nearest_resistance'] is None or abs(level - current_price) < abs(key_levels['nearest_resistance'] - current_price):
                            key_levels['nearest_resistance'] = level
                
                # Significant lows (potential support)
                for level in recent_lows:
                    if abs(current_price - level) / level <= KEY_LEVEL_TOLERANCE:
                        key_levels['support_levels'].append(level)
                        if key_levels['nearest_support'] is None or abs(level - current_price) < abs(key_levels['nearest_support'] - current_price):
                            key_levels['nearest_support'] = level
    except Exception as e:
        print(f"Error analyzing key levels: {e}")
    
    return key_levels

# 🚨 NEW: INSTITUTIONAL RETEST PATTERN DETECTION 🚨
def detect_institutional_retest(index, df_5min, key_levels):
    """
    Detect institutional retest patterns at key levels
    Returns: "CE" for support retest bullish, "PE" for resistance retest bearish, None otherwise
    """
    try:
        close = ensure_series(df_5min['Close'])
        high = ensure_series(df_5min['High'])
        low = ensure_series(df_5min['Low'])
        volume = ensure_series(df_5min['Volume'])
        
        if len(close) < RETEST_CONFIRMATION_BARS + 2:
            return None
            
        current_price = close.iloc[-1]
        current_volume = volume.iloc[-1]
        avg_volume = volume.rolling(20).mean().iloc[-1] if len(volume) >= 20 else volume.mean()
        
        # Check resistance retest (bearish scenario)
        if key_levels['nearest_resistance'] is not None:
            resistance = key_levels['nearest_resistance']
            
            # Check if price is near resistance
            if abs(current_price - resistance) / resistance <= KEY_LEVEL_TOLERANCE:
                # Look for bearish signs: price approaching resistance with volume
                if current_volume > avg_volume * 1.5:
                    
                    # Check for green hammer at resistance (bearish reversal)
                    if len(close) >= 3:
                        # Hammer pattern detection
                        body_size = abs(close.iloc[-1] - close.iloc[-2])
                        lower_wick = min(close.iloc[-1], close.iloc[-2]) - low.iloc[-2]
                        
                        if (close.iloc[-2] > close.iloc[-3] and  # Green candle
                            lower_wick > body_size * HAMMER_WICK_TO_BODY_RATIO and  # Long lower wick
                            close.iloc[-1] < close.iloc[-2] and  # Current candle is red
                            current_price <= resistance * 1.01):  # Near resistance
                            
                            key_levels['hammer_at_resistance'] = True
                            return "PE"  # Bearish signal
                    
                    # General resistance rejection
                    if (current_price <= resistance * 1.005 and  # Near resistance
                        close.iloc[-1] < close.iloc[-2] and  # Bearish candle
                        high.iloc[-1] >= resistance * 0.995):  # Touched resistance
                        
                        key_levels['resistance_retest'] = True
                        return "PE"  # Bearish signal
        
        # Check support retest (bullish scenario)
        if key_levels['nearest_support'] is not None:
            support = key_levels['nearest_support']
            
            # Check if price is near support
            if abs(current_price - support) / support <= KEY_LEVEL_TOLERANCE:
                # Look for bullish signs: price bouncing from support with volume
                if current_volume > avg_volume * 1.5:
                    
                    # Check for bullish reversal at support
                    if len(close) >= 3:
                        # Bullish reversal pattern
                        if (close.iloc[-2] < close.iloc[-3] and  # Red candle
                            close.iloc[-1] > close.iloc[-2] and  # Current candle is green
                            low.iloc[-2] <= support * 1.01 and  # Touched support
                            current_price >= support * 0.995):  # Bounced from support
                            
                            key_levels['support_retest'] = True
                            return "CE"  # Bullish signal
    except Exception as e:
        print(f"Error detecting retest patterns: {e}")
    
    return None

# 🚨 NEW: BREAKOUT TRAP DETECTION 🚨
def detect_breakout_trap(index, df_5min, key_levels):
    """
    Detect false breakouts (traps) at key levels
    Returns: "CE" for bear trap (false breakdown), "PE" for bull trap (false breakout)
    """
    try:
        close = ensure_series(df_5min['Close'])
        high = ensure_series(df_5min['High'])
        low = ensure_series(df_5min['Low'])
        volume = ensure_series(df_5min['Volume'])
        
        if len(close) < 5:
            return None
            
        current_price = close.iloc[-1]
        current_volume = volume.iloc[-1]
        avg_volume = volume.rolling(20).mean().iloc[-1] if len(volume) >= 20 else volume.mean()
        
        # Check for bull trap (false breakout above resistance)
        if key_levels['nearest_resistance'] is not None:
            resistance = key_levels['nearest_resistance']
            
            # Check if price broke above resistance but closed below
            if (high.iloc[-2] > resistance * 1.005 and  # Broke above resistance
                close.iloc[-2] < resistance and  # Closed below resistance
                current_price < resistance and  # Still below resistance
                current_volume > avg_volume * BREAKOUT_CONFIRMATION_VOLUME):
                
                key_levels['breakout_trap'] = True
                return "PE"  # Bearish signal (false breakout)
        
        # Check for bear trap (false breakdown below support)
        if key_levels['nearest_support'] is not None:
            support = key_levels['nearest_support']
            
            # Check if price broke below support but closed above
            if (low.iloc[-2] < support * 0.995 and  # Broke below support
                close.iloc[-2] > support and  # Closed above support
                current_price > support and  # Still above support
                current_volume > avg_volume * BREAKOUT_CONFIRMATION_VOLUME):
                
                key_levels['breakout_trap'] = True
                return "CE"  # Bullish signal (false breakdown)
                
    except Exception as e:
        print(f"Error detecting breakout traps: {e}")
    
    return None

# 🚨 NEW: INSTITUTIONAL PRICE ACTION WITH KEY LEVELS 🚨
def institutional_price_action_with_key_levels(index, df_5min):
    """
    Main institutional price action analysis with key level confirmation
    """
    try:
        close = ensure_series(df_5min['Close'])
        if len(close) < 10:
            return None
            
        current_price = close.iloc[-1]
        
        # Analyze key levels
        key_levels = analyze_institutional_key_levels(index, current_price)
        
        # Check for retest patterns first (highest priority)
        retest_signal = detect_institutional_retest(index, df_5min, key_levels)
        if retest_signal:
            # Additional confirmation for retest signals
            if key_levels['hammer_at_resistance'] and retest_signal == "PE":
                return retest_signal, df_5min, False, "institutional_retest"
            elif key_levels['support_retest'] and retest_signal == "CE":
                return retest_signal, df_5min, False, "institutional_retest"
            elif key_levels['resistance_retest'] and retest_signal == "PE":
                return retest_signal, df_5min, False, "institutional_retest"
        
        # Check for breakout traps (second priority)
        trap_signal = detect_breakout_trap(index, df_5min, key_levels)
        if trap_signal:
            return trap_signal, df_5min, True, "breakout_trap"  # True for fakeout
        
    except Exception as e:
        print(f"Error in institutional price action: {e}")
    
    return None

# 🚨 LAYER 1: OTE RETRACEMENT (KEEP AS REQUESTED) 🚨
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
                return "CE", df, False, "ote_retracement"
                
            ote_level = swing_low + (swing_range * level)
            if (abs(current_price - ote_level) / ote_level < 0.0015 and  # Tighter tolerance
                close.iloc[-1] < close.iloc[-2] and
                close.iloc[-1] < close.iloc[-3]):
                return "PE", df, False, "ote_retracement"
    except Exception:
        return None
    return None

# 🚨 LAYER 2: LIQUIDITY ZONE (KEEP AS REQUESTED) 🚨
def detect_liquidity_zone_signal(index, df):
    """
    Liquidity zone detection with institutional confirmation
    """
    try:
        close = ensure_series(df['Close'])
        if len(close) < 10:
            return None
            
        last_close = float(close.iloc[-1])
        bull_liq, bear_liq = institutional_liquidity_hunt(index, df)
        
        # Check if price is at liquidity zones with volume confirmation
        volume = ensure_series(df['Volume'])
        current_volume = volume.iloc[-1]
        avg_volume = volume.rolling(20).mean().iloc[-1] if len(volume) >= 20 else volume.mean()
        
        # Check bullish liquidity (support zone)
        for zone in bull_liq:
            if zone is not None and abs(last_close - zone) <= 5:
                if current_volume > avg_volume * 1.3:
                    # Additional confirmation: price should be bouncing from zone
                    if len(close) >= 3 and close.iloc[-1] > close.iloc[-2]:
                        return "CE", df, False, "liquidity_zone"
        
        # Check bearish liquidity (resistance zone)
        for zone in bear_liq:
            if zone is not None and abs(last_close - zone) <= 5:
                if current_volume > avg_volume * 1.3:
                    # Additional confirmation: price should be rejecting from zone
                    if len(close) >= 3 and close.iloc[-1] < close.iloc[-2]:
                        return "PE", df, False, "liquidity_zone"
                        
    except Exception as e:
        print(f"Error in liquidity zone detection: {e}")
    
    return None

# --------- INSTITUTIONAL MOMENTUM CONFIRMATION 🚨
def institutional_momentum_confirmation(index, df, proposed_signal):
    """
    Final institutional confirmation layer with key level check
    """
    try:
        close = ensure_series(df['Close'])
        volume = ensure_series(df['Volume'])
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        
        if len(close) < 5:
            return False
            
        # Get current price and analyze key levels
        current_price = close.iloc[-1]
        key_levels = analyze_institutional_key_levels(index, current_price)
        
        # Price momentum confirmation
        if proposed_signal == "CE":
            # For CE: require upward momentum
            if not (close.iloc[-1] > close.iloc[-2] and close.iloc[-2] > close.iloc[-3]):
                return False
            # Check if near strong resistance (avoid false signals)
            if (key_levels['nearest_resistance'] is not None and 
                abs(current_price - key_levels['nearest_resistance']) / key_levels['nearest_resistance'] < 0.005):
                return False  # Too close to resistance
                
        elif proposed_signal == "PE":
            # For PE: require downward momentum
            if not (close.iloc[-1] < close.iloc[-2] and close.iloc[-2] < close.iloc[-3]):
                return False
            # Check if near strong support (avoid false signals)
            if (key_levels['nearest_support'] is not None and 
                abs(current_price - key_levels['nearest_support']) / key_levels['nearest_support'] < 0.005):
                return False  # Too close to support
                
        # Volume confirmation
        current_volume = volume.iloc[-1]
        avg_volume = volume.rolling(20).mean().iloc[-1] if len(volume) >= 20 else volume.mean()
        if current_volume < avg_volume * 1.2:
            return False  # Insufficient volume
            
        return True
        
    except Exception:
        return False

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

# --------- NEW: SIGNAL DEDUPLICATION AND COOLDOWN CHECK ---------
def can_send_signal(index, strike, option_type):
    """Check if we can send signal based on deduplication and cooldown rules"""
    global active_strikes, last_signal_time
    
    current_time = time.time()
    strike_key = f"{index}_{strike}_{option_type}"
    
    # Check if same strike is already active
    if strike_key in active_strikes:
        return False
        
    # Check cooldown for this index
    if index in last_signal_time:
        time_since_last = current_time - last_signal_time[index]
        if time_since_last < signal_cooldown:
            return False
    
    return True

def update_signal_tracking(index, strike, option_type, signal_id):
    """Update tracking for sent signals"""
    global active_strikes, last_signal_time
    
    strike_key = f"{index}_{strike}_{option_type}"
    active_strikes[strike_key] = {
        'signal_id': signal_id,
        'timestamp': time.time(),
        'targets_hit': 0
    }
    
    last_signal_time[index] = time.time()

def update_signal_progress(signal_id, targets_hit):
    """Update progress of active signal"""
    for strike_key, data in active_strikes.items():
        if data['signal_id'] == signal_id:
            active_strikes[strike_key]['targets_hit'] = targets_hit
            break

def clear_completed_signal(signal_id):
    """Clear signal from active tracking when completed"""
    global active_strikes
    active_strikes = {k: v for k, v in active_strikes.items() if v['signal_id'] != signal_id}

# --------- UPDATED STRATEGY CHECK WITH INSTITUTIONAL KEY LEVELS ---------
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

    # 🚨 HIGHEST PRIORITY: INSTITUTIONAL PRICE ACTION WITH KEY LEVELS 🚨
    institutional_signal = institutional_price_action_with_key_levels(index, df5)
    if institutional_signal:
        side, df, fakeout, strategy_key = institutional_signal
        if institutional_momentum_confirmation(index, df5, side):
            return side, df, fakeout, "institutional_retest"

    # 🚨 SECOND PRIORITY: OTE RETRACEMENT (KEEP AS REQUESTED) 🚨
    ote_signal = detect_ote_retracement(df5)
    if ote_signal:
        side, df, fakeout, strategy_key = ote_signal
        if institutional_momentum_confirmation(index, df5, side):
            return side, df, fakeout, "ote_retracement"

    # 🚨 THIRD PRIORITY: LIQUIDITY ZONE (KEEP AS REQUESTED) 🚨
    liquidity_signal = detect_liquidity_zone_signal(index, df5)
    if liquidity_signal:
        side, df, fakeout, strategy_key = liquidity_signal
        if institutional_momentum_confirmation(index, df5, side):
            return side, df, fakeout, "liquidity_zone"

    return None

# --------- FIXED: ENHANCED TRADE MONITORING AND TRACKING ---------
active_trades = {}

def calculate_pnl(entry, max_price, targets, targets_hit, sl):
    try:
        if targets is None or len(targets) == 0:
            diff = max_price - entry
            if diff > 0:
                return f"+{diff:.2f}"
            elif diff < 0:
                return f"-{abs(diff):.2f}"
            else:
                return "0"
        
        if not isinstance(targets_hit, (list, tuple)):
            targets_hit = list(targets_hit) if targets_hit is not None else [False]*len(targets)
        if len(targets_hit) < len(targets):
            targets_hit = list(targets_hit) + [False] * (len(targets) - len(targets_hit))
        
        achieved_prices = [target for i, target in enumerate(targets) if targets_hit[i]]
        if achieved_prices:
            exit_price = achieved_prices[-1]
            diff = exit_price - entry
            if diff > 0:
                return f"+{diff:.2f}"
            elif diff < 0:
                return f"-{abs(diff):.2f}"
            else:
                return "0"
        else:
            if max_price <= sl:
                diff = sl - entry
                if diff > 0:
                    return f"+{diff:.2f}"
                elif diff < 0:
                    return f"-{abs(diff):.2f}"
                else:
                    return "0"
            else:
                diff = max_price - entry
                if diff > 0:
                    return f"+{diff:.2f}"
                elif diff < 0:
                    return f"-{abs(diff):.2f}"
                else:
                    return "0"
    except Exception:
        return "0"

def monitor_price_live(symbol, entry, targets, sl, fakeout, thread_id, strategy_name, signal_data):
    def monitoring_thread():
        global daily_signals
        
        last_high = entry
        weakness_sent = False
        in_trade = False
        entry_price_achieved = False
        max_price_reached = entry
        targets_hit = [False] * len(targets)
        last_activity_time = time.time()
        signal_id = signal_data.get('signal_id')
        
        while True:
            current_time = time.time()
            
            # Check for inactivity (20 minutes)
            if not in_trade and (current_time - last_activity_time) > 1200:  # 20 minutes
                send_telegram(f"⏰ {symbol}: No activity for 20 minutes. Allowing new signals.", reply_to=thread_id)
                clear_completed_signal(signal_id)
                break
                
            if should_stop_trading():
                try:
                    final_pnl = calculate_pnl(entry, max_price_reached, targets, targets_hit, sl)
                except Exception:
                    final_pnl = "0"
                signal_data.update({
                    "entry_status": "NOT_ENTERED" if not entry_price_achieved else "ENTERED",
                    "targets_hit": sum(targets_hit),
                    "max_price_reached": max_price_reached,
                    "zero_targets": sum(targets_hit) == 0,
                    "no_new_highs": max_price_reached <= entry,
                    "final_pnl": final_pnl
                })
                daily_signals.append(signal_data)
                clear_completed_signal(signal_id)
                break
                
            price = fetch_option_price(symbol)
            if price:
                last_activity_time = current_time
                price = round(price)
                
                if price > max_price_reached:
                    max_price_reached = price
                
                if not in_trade:
                    if price >= entry:
                        send_telegram(f"✅ ENTRY TRIGGERED at {price}", reply_to=thread_id)
                        in_trade = True
                        entry_price_achieved = True
                        last_high = price
                        signal_data["entry_status"] = "ENTERED"
                else:
                    if price > last_high:
                        send_telegram(f"🚀 {symbol} making new high → {price}", reply_to=thread_id)
                        last_high = price
                    elif not weakness_sent and price < sl * 1.05:
                        send_telegram(f"⚡ {symbol} showing weakness near SL {sl}", reply_to=thread_id)
                        weakness_sent = True
                    
                    # Update signal progress
                    current_targets_hit = sum(targets_hit)
                    for i, target in enumerate(targets):
                        if price >= target and not targets_hit[i]:
                            send_telegram(f"🎯 {symbol}: Target {i+1} hit at ₹{target}", reply_to=thread_id)
                            targets_hit[i] = True
                            current_targets_hit = sum(targets_hit)
                            update_signal_progress(signal_id, current_targets_hit)
                    
                    # SL hit - allow immediate new signal
                    if price <= sl:
                        send_telegram(f"🔗 {symbol}: Stop Loss {sl} hit. Exit trade. ALLOWING NEW SIGNAL.", reply_to=thread_id)
                        try:
                            final_pnl = calculate_pnl(entry, max_price_reached, targets, targets_hit, sl)
                        except Exception:
                            final_pnl = "0"
                        signal_data.update({
                            "targets_hit": sum(targets_hit),
                            "max_price_reached": max_price_reached,
                            "zero_targets": sum(targets_hit) == 0,
                            "no_new_highs": max_price_reached <= entry,
                            "final_pnl": final_pnl
                        })
                        daily_signals.append(signal_data)
                        clear_completed_signal(signal_id)  # Clear for new signal
                        break
                        
                    # 2nd target hit - allow new signals but continue monitoring
                    if current_targets_hit >= 2:
                        update_signal_progress(signal_id, current_targets_hit)
                        # Continue monitoring but new signals allowed
                    
                    # All targets hit - complete trade
                    if all(targets_hit):
                        send_telegram(f"🏆 {symbol}: ALL TARGETS HIT! Trade completed successfully!", reply_to=thread_id)
                        try:
                            final_pnl = calculate_pnl(entry, max_price_reached, targets, targets_hit, sl)
                        except Exception:
                            final_pnl = "0"
                        signal_data.update({
                            "targets_hit": len(targets),
                            "max_price_reached": max_price_reached,
                            "zero_targets": False,
                            "no_new_highs": False,
                            "final_pnl": final_pnl
                        })
                        daily_signals.append(signal_data)
                        clear_completed_signal(signal_id)
                        break
            
            time.sleep(10)
    
    thread = threading.Thread(target=monitoring_thread)
    thread.daemon = True
    thread.start()

# --------- FIXED: WORKING EOD REPORT SYSTEM ---------
def send_individual_signal_reports():
    """Send each signal in separate detailed messages after market hours"""
    global daily_signals, all_generated_signals
    
    # 🚨 CRITICAL FIX: Combine both signal sources
    all_signals = daily_signals + all_generated_signals
    
    # Remove duplicates based on signal_id
    seen_ids = set()
    unique_signals = []
    for signal in all_signals:
        sid = signal.get('signal_id')
        if not sid:
            continue
        if sid not in seen_ids:
            seen_ids.add(sid)
            unique_signals.append(signal)
    
    if not unique_signals:
        send_telegram("📊 END OF DAY REPORT\nNo signals generated today.")
        return
    
    # Send header message
    send_telegram(f"🕒 END OF DAY SIGNAL REPORT - { (datetime.utcnow()+timedelta(hours=5,minutes=30)).strftime('%d-%b-%Y') }\n"
                  f"📈 Total Signals: {len(unique_signals)}\n"
                  f"─────────────────────────────")
    
    # Send each signal in separate message
    for i, signal in enumerate(unique_signals, 1):
        targets_hit_list = []
        if signal.get('targets_hit', 0) > 0:
            for j in range(signal.get('targets_hit', 0)):
                if j < len(signal.get('targets', [])):
                    targets_hit_list.append(str(signal['targets'][j]))
        
        targets_for_disp = signal.get('targets', [])
        while len(targets_for_disp) < 4:
            targets_for_disp.append('-')
        
        msg = (f"📊 SIGNAL #{i} - {signal.get('index','?')} {signal.get('strike','?')} {signal.get('option_type','?')}\n"
               f"─────────────────────────────\n"
               f"📅 Date: {(datetime.utcnow()+timedelta(hours=5,minutes=30)).strftime('%d-%b-%Y')}\n"
               f"🕒 Time: {signal.get('timestamp','?')}\n"
               f"📈 Index: {signal.get('index','?')}\n"
               f"🎯 Strike: {signal.get('strike','?')}\n"
               f"🔰 Type: {signal.get('option_type','?')}\n"
               f"🏷️ Strategy: {signal.get('strategy','?')}\n\n"
               
               f"💰 ENTRY: ₹{signal.get('entry_price','?')}\n"
               f"🎯 TARGETS: {targets_for_disp[0]} // {targets_for_disp[1]} // {targets_for_disp[2]} // {targets_for_disp[3]}\n"
               f"🛑 STOP LOSS: ₹{signal.get('sl','?')}\n\n"
               
               f"📊 PERFORMANCE:\n"
               f"• Entry Status: {signal.get('entry_status', 'PENDING')}\n"
               f"• Targets Hit: {signal.get('targets_hit', 0)}/4\n")
        
        if targets_hit_list:
            msg += f"• Targets Achieved: {', '.join(targets_hit_list)}\n"
        
        msg += (f"• Max Price Reached: ₹{signal.get('max_price_reached', signal.get('entry_price','?'))}\n"
                f"• Final P&L: {signal.get('final_pnl', '0')} points\n\n"
                
                f"⚡ Fakeout: {'YES' if signal.get('fakeout') else 'NO'}\n"
                f"📈 Index Price at Signal: {signal.get('index_price','?')}\n"
                f"🆔 Signal ID: {signal.get('signal_id','?')}\n"
                f"─────────────────────────────")
        
        send_telegram(msg)
        time.sleep(1)
    
    # Send summary
    total_pnl = 0.0
    successful_trades = 0
    for signal in unique_signals:
        pnl_str = signal.get("final_pnl", "0")
        try:
            if isinstance(pnl_str, str) and pnl_str.startswith("+"):
                total_pnl += float(pnl_str[1:])
                successful_trades += 1
            elif isinstance(pnl_str, str) and pnl_str.startswith("-"):
                total_pnl -= float(pnl_str[1:])
        except:
            pass
    
    summary_msg = (f"📈 DAY SUMMARY\n"
                   f"─────────────────────────────\n"
                   f"• Total Signals: {len(unique_signals)}\n"
                   f"• Successful Trades: {successful_trades}\n"
                   f"• Success Rate: {(successful_trades/len(unique_signals))*100:.1f}%\n"
                   f"• Total P&L: ₹{total_pnl:+.2f}\n"
                   f"─────────────────────────────")
    
    send_telegram(summary_msg)
    
    # 🚨 COMPULSORY CONFIRMATION
    send_telegram("✅ END OF DAY REPORTS COMPLETED! See you tomorrow at 9:15 AM! 🚀")

# 🚨 FIXED: UPDATED SIGNAL SENDING WITH STRICT EXPIRY VALIDATION 🚨
def send_signal(index, side, df, fakeout, strategy_key):
    global signal_counter, all_generated_signals
    
    # 🚨 CRITICAL FIX: Each index uses its OWN isolated strike calculation
    signal_detection_price = float(ensure_series(df["Close"]).iloc[-1])
    strike = round_strike(index, signal_detection_price)
    
    if strike is None:
        send_telegram(f"⚠️ {index}: could not determine strike (price missing). Signal skipped.")
        return
        
    # 🚨 CHECK DEDUPLICATION AND COOLDOWN
    if not can_send_signal(index, strike, side):
        return
        
    # 🚨 FIXED: STRICT EXPIRY ENFORCEMENT - Only use specified expiries
    symbol = get_option_symbol(index, EXPIRIES[index], strike, side)
    
    if symbol is None:
        # 🚨 SILENT REJECTION - No Telegram message for wrong expiry!
        print(f"❌ STRICT EXPIRY ENFORCEMENT: {index} {strike}{side} - Only {EXPIRIES[index]} allowed")
        return  # Just exit quietly without sending any message
    
    option_price = fetch_option_price(symbol)
    if not option_price: 
        return
    
    entry = round(option_price)
    
    high = ensure_series(df["High"])
    low = ensure_series(df["Low"])
    close = ensure_series(df["Close"])
    
    # 🚨 INSTITUTIONAL TARGETS WITH KEY LEVEL CONSIDERATION
    # First analyze key levels
    current_price = signal_detection_price
    key_levels = analyze_institutional_key_levels(index, current_price)
    
    # Calculate institutional-style targets
    if side == "CE":
        # For CE: Use bullish liquidity zones and key resistance levels
        if bull_liq:
            nearest_bull_zone = max([z for z in bull_liq if z is not None])
            price_gap = nearest_bull_zone - signal_detection_price
        elif key_levels['nearest_resistance'] is not None:
            # Use key resistance level as target reference
            price_gap = key_levels['nearest_resistance'] - signal_detection_price
        else:
            price_gap = signal_detection_price * 0.008  # 0.8% move
        
        # Institutional target multipliers
        base_move = max(price_gap * 0.3, 40)  # Minimum 40 points
        targets = [
            round(entry + base_move * 1.0),
            round(entry + base_move * 1.8),  # Bigger second target
            round(entry + base_move * 2.8),  # Bigger third target
            round(entry + base_move * 4.0)   # Bigger fourth target
        ]
        sl = round(entry - base_move * 0.8)
        
    else:  # PE
        # For PE: Use bearish liquidity zones and key support levels
        if bear_liq:
            nearest_bear_zone = min([z for z in bear_liq if z is not None])
            price_gap = signal_detection_price - nearest_bear_zone
        elif key_levels['nearest_support'] is not None:
            # Use key support level as target reference
            price_gap = signal_detection_price - key_levels['nearest_support']
        else:
            price_gap = signal_detection_price * 0.008  # 0.8% move
        
        # Institutional target multipliers
        base_move = max(price_gap * 0.3, 40)  # Minimum 40 points
        targets = [
            round(entry + base_move * 1.0),
            round(entry + base_move * 1.8),  # Bigger second target
            round(entry + base_move * 2.8),  # Bigger third target
            round(entry + base_move * 4.0)   # Bigger fourth target
        ]
        sl = round(entry - base_move * 0.8)
    
    # Get bull and bear liquidity for display
    bull_liq, bear_liq = institutional_liquidity_hunt(index, df)
    
    targets_str = "//".join(str(t) for t in targets) + "++"
    
    strategy_name = STRATEGY_NAMES.get(strategy_key, strategy_key.upper())
    
    signal_id = f"SIG{signal_counter:04d}"
    signal_counter += 1
    
    signal_data = {
        "signal_id": signal_id,
        "timestamp": (datetime.utcnow()+timedelta(hours=5,minutes=30)).strftime("%H:%M:%S"),
        "index": index,
        "strike": strike,
        "option_type": side,
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
        "final_pnl": "0",
        "key_level_analysis": {
            "nearest_support": key_levels.get('nearest_support'),
            "nearest_resistance": key_levels.get('nearest_resistance'),
            "support_retest": key_levels.get('support_retest', False),
            "resistance_retest": key_levels.get('resistance_retest', False),
            "hammer_at_resistance": key_levels.get('hammer_at_resistance', False)
        }
    }
    
    # 🚨 UPDATE SIGNAL TRACKING
    update_signal_tracking(index, strike, side, signal_id)
    
    # 🚨 FIX: Track signal immediately for EOD reports
    all_generated_signals.append(signal_data.copy())
    
    # Add key level info to telegram message
    key_level_info = ""
    if key_levels['nearest_support']:
        key_level_info += f"📉 Nearest Support: {key_levels['nearest_support']}\n"
    if key_levels['nearest_resistance']:
        key_level_info += f"📈 Nearest Resistance: {key_levels['nearest_resistance']}\n"
    if key_levels['support_retest']:
        key_level_info += f"🔄 Support Retest Detected\n"
    if key_levels['resistance_retest']:
        key_level_info += f"🔄 Resistance Retest Detected\n"
    if key_levels['hammer_at_resistance']:
        key_level_info += f"🔨 Green Hammer at Resistance (BEARISH)\n"
    
    msg = (f"🟢 {index} {strike} {side}\n"
           f"SYMBOL: {symbol}\n"
           f"ABOVE {entry}\n"
           f"TARGETS: {targets_str}\n"
           f"SL: {sl}\n"
           f"FAKEOUT: {'YES' if fakeout else 'NO'}\n"
           f"STRATEGY: {strategy_name}\n"
           f"{key_level_info}"
           f"SIGNAL ID: {signal_id}")
         
    thread_id = send_telegram(msg)
    
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
    
    monitor_price_live(symbol, entry, targets, sl, fakeout, thread_id, strategy_name, signal_data)

# --------- FIXED: UPDATED TRADE THREAD WITH ISOLATED INDICES ---------
def trade_thread(index):
    """Generate signals with completely isolated index processing"""
    result = analyze_index_signal(index)
    
    if not result:
        return
        
    if len(result) == 4:
        side, df, fakeout, strategy_key = result
    else:
        side, df, fakeout = result
        strategy_key = "unknown"
    
    # 🚨 CRITICAL FIX: Each index thread processes ONLY its own data
    # No cross-contamination between indices
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

# --------- FIXED: MAIN LOOP (KEPT INDICES ONLY) ---------
def run_algo_parallel():
    if not is_market_open(): 
        print("❌ Market closed - skipping iteration")
        return
        
    if should_stop_trading():
        global STOP_SENT, EOD_REPORT_SENT
        if not STOP_SENT:
            send_telegram("🛑 Market closed at 3:30 PM IST - Algorithm stopped")
            STOP_SENT = True
            
        # 🚨 FIX: GUARANTEED EOD REPORTS
        if not EOD_REPORT_SENT:
            time.sleep(15)  # Wait for all monitoring threads to complete
            send_telegram("📊 GENERATING COMPULSORY END-OF-DAY REPORT...")
            try:
                send_individual_signal_reports()
            except Exception as e:
                send_telegram(f"⚠️ EOD Report Error, retrying: {str(e)[:100]}")
                time.sleep(10)
                send_individual_signal_reports()  # Retry once
            EOD_REPORT_SENT = True
            send_telegram("✅ TRADING DAY COMPLETED! See you tomorrow at 9:15 AM! 🎯")
            
        return
        
    threads = []
    # 🚨 ONLY KEPT INDICES
    kept_indices = ["NIFTY", "BANKNIFTY", "SENSEX", "MIDCPNIFTY"]
    
    for index in kept_indices:
        t = threading.Thread(target=trade_thread, args=(index,))
        t.start()
        threads.append(t)
    
    for t in threads: 
        t.join()

# --------- FIXED: START WITH WORKING EOD SYSTEM ---------
STARTED_SENT = False
STOP_SENT = False
MARKET_CLOSED_SENT = False
EOD_REPORT_SENT = False

# Initialize strategy tracking
initialize_strategy_tracking()

while True:
    try:
        # Get current IST time
        utc_now = datetime.utcnow()
        ist_now = utc_now + timedelta(hours=5, minutes=30)
        current_time_ist = ist_now.time()
        current_datetime_ist = ist_now
        
        # Check if market is open
        market_open = is_market_open()
        
        # 🚨 MARKET CLOSED BEHAVIOR
        if not market_open:
            if not MARKET_CLOSED_SENT:
                send_telegram("🔴 Market is currently closed. Algorithm waiting for 9:15 AM...")
                MARKET_CLOSED_SENT = True
                STARTED_SENT = False
                STOP_SENT = False
                EOD_REPORT_SENT = False
            
            # 🚨 COMPULSORY EOD REPORT TRIGGER BETWEEN 3:30 PM - 4:00 PM
            if current_time_ist >= dtime(15,30) and current_time_ist <= dtime(16,0) and not EOD_REPORT_SENT:
                send_telegram("📊 GENERATING COMPULSORY END-OF-DAY REPORT...")
                time.sleep(10)
                send_individual_signal_reports()
                EOD_REPORT_SENT = True
                send_telegram("✅ EOD Report completed! Algorithm will resume tomorrow.")
            
            time.sleep(30)
            continue
        
        # 🚨 MARKET OPEN BEHAVIOR
        if not STARTED_SENT:
            send_telegram("🚀 GIT ULTIMATE MASTER ALGO STARTED - 4 Indices Running\n"
                         "✅ ONLY 2 STRATEGIES: LIQUIDITY ZONE & OTE RETRACEMENT\n"
                         "✅ INSTITUTIONAL KEY LEVEL ANALYSIS: 1D/5D/1M Support/Resistance\n"
                         "✅ RETEST PATTERN DETECTION at Key Levels\n"
                         "✅ BREAKOUT TRAP DETECTION (False Breakouts)\n"
                         "✅ GREEN HAMMER at Resistance = PE Signal\n"
                         "✅ Signal Deduplication & Cooldown\n"
                         "✅ Guaranteed EOD Reports at 3:30 PM\n"
                         "✅ 🚨 STRICT EXPIRY ENFORCEMENT 🚨")
            STARTED_SENT = True
            STOP_SENT = False
            MARKET_CLOSED_SENT = False
        
        # 🚨 MARKET CLOSE DETECTION WITH GUARANTEED EOD REPORT
        if should_stop_trading():
            if not STOP_SENT:
                send_telegram("🛑 Market closing time reached! Preparing EOD Report...")
                STOP_SENT = True
                STARTED_SENT = False
            
            # 🚨 GUARANTEED EOD REPORT - NO EXCEPTIONS
            if not EOD_REPORT_SENT:
                send_telegram("📊 FINALIZING TRADES...")
                time.sleep(20)  # Extra time for all threads to complete
                try:
                    send_individual_signal_reports()
                except Exception as e:
                    send_telegram(f"⚠️ EOD Report Error, retrying: {str(e)[:100]}")
                    time.sleep(10)
                    send_individual_signal_reports()  # Retry once
                EOD_REPORT_SENT = True
                send_telegram("✅ TRADING DAY COMPLETED! See you tomorrow at 9:15 AM! 🎯")
            
            time.sleep(60)
            continue
            
        # 🚨 RUN MAIN ALGORITHM DURING MARKET HOURS
        run_algo_parallel()
        time.sleep(30)
        
    except Exception as e:
        error_msg = f"⚠️ Main loop error: {str(e)[:100]}"
        send_telegram(error_msg)
        time.sleep(60)

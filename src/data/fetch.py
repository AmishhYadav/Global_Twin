"""
Global Twin — Real-World Data Fetcher

Pulls economic data from FRED and Yahoo Finance APIs,
normalizes to daily frequency, and saves as clean CSVs.
"""

import os
import io
import pandas as pd
import numpy as np
import requests
from datetime import datetime


# FRED public CSV endpoint (no API key)
FRED_BASE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def fetch_fred_series(series_id, start_date="2018-01-01", end_date=None):
    """
    Fetch a single FRED time series via their public CSV endpoint (no API key needed).
    
    Args:
        series_id: FRED series identifier (e.g., 'UNRATE')
        start_date: Start date string 'YYYY-MM-DD'
        end_date: End date string (defaults to today)
    
    Returns:
        pd.Series with DatetimeIndex, or None on failure.
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    url = f"{FRED_BASE_URL}?id={series_id}&cosd={start_date}&coed={end_date}"
    
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        
        content = resp.text
        
        # Guard: FRED sometimes returns HTML error pages
        if '<html' in content.lower() or '<title>' in content.lower():
            print(f"  ✗ FRED [{series_id}]: returned HTML instead of CSV (possible rate limit)")
            return None
        
        # Parse CSV — auto-detect the date column name
        df = pd.read_csv(io.StringIO(content))
        
        if df.empty or len(df.columns) < 2:
            print(f"  ✗ FRED [{series_id}]: empty or malformed CSV")
            return None
        
        # First column is the date, second is the value
        date_col = df.columns[0]
        val_col = df.columns[1]
        
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col])
        df = df.set_index(date_col)
        df.index.name = 'DATE'
        
        series = pd.to_numeric(df[val_col], errors='coerce')
        series = series.dropna()
        series.name = series_id
        
        if len(series) == 0:
            print(f"  ✗ FRED [{series_id}]: no valid numeric data")
            return None
        
        print(f"  ✓ FRED [{series_id}]: {len(series)} observations ({series.index.min().date()} to {series.index.max().date()})")
        return series
        
    except requests.exceptions.RequestException as e:
        print(f"  ✗ FRED [{series_id}]: network error — {e}")
        return None
    except Exception as e:
        print(f"  ✗ FRED [{series_id}] failed: {e}")
        return None


def fetch_yahoo_ticker(ticker, indicator_name, start_date="2018-01-01", end_date=None):
    """
    Fetch a single Yahoo Finance ticker's close price.
    
    Args:
        ticker: Yahoo Finance ticker (e.g., 'CL=F')
        indicator_name: Human-readable name for the indicator
        start_date: Start date string
        end_date: End date string (defaults to today)
    
    Returns:
        pd.Series with DatetimeIndex, or None on failure.
    """
    try:
        import yfinance as yf
    except ImportError:
        print(f"  ✗ yfinance not installed. Run: pip install yfinance")
        return None
    
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    try:
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True,
            timeout=30,
        )
        
        if data is None or data.empty:
            print(f"  ✗ Yahoo [{indicator_name}] ({ticker}): No data returned")
            return None
        
        # Handle multi-level columns from newer yfinance
        if isinstance(data.columns, pd.MultiIndex):
            # Try to flatten
            data.columns = data.columns.get_level_values(0)
        
        if 'Close' in data.columns:
            series = data['Close']
        elif len(data.columns) == 1:
            series = data.iloc[:, 0]
        else:
            print(f"  ✗ Yahoo [{indicator_name}] ({ticker}): unexpected columns {list(data.columns)}")
            return None
        
        series = series.dropna()
        series.name = indicator_name
        series.index.name = 'DATE'
        
        if len(series) == 0:
            print(f"  ✗ Yahoo [{indicator_name}] ({ticker}): no valid data after cleaning")
            return None
        
        print(f"  ✓ Yahoo [{indicator_name}]: {len(series)} observations ({series.index.min().date()} to {series.index.max().date()})")
        return series
        
    except Exception as e:
        print(f"  ✗ Yahoo [{indicator_name}] ({ticker}): {e}")
        return None


def fetch_all_indicators(start_date="2018-01-01", save_dir="data/raw"):
    """
    Fetch ALL indicators from the registry and save to CSV files.
    
    Produces two files:
      - data/raw/fred_indicators.csv
      - data/raw/yahoo_indicators.csv
    
    Returns:
        dict with 'fred' and 'yahoo' DataFrames.
    """
    from src.data.indicators import FRED_INDICATORS, YAHOO_INDICATORS
    
    os.makedirs(save_dir, exist_ok=True)
    
    # ── FRED ──
    print("\n" + "=" * 50)
    print("Fetching FRED Economic Indicators...")
    print("=" * 50)
    fred_series_list = []
    for name, meta in FRED_INDICATORS.items():
        s = fetch_fred_series(meta['series'], start_date=start_date)
        if s is not None:
            s.name = name
            fred_series_list.append(s)
    
    fred_df = pd.DataFrame()
    if fred_series_list:
        fred_df = pd.concat(fred_series_list, axis=1)
        fred_df = fred_df.asfreq('D')
        fred_df = fred_df.ffill().bfill()
        fred_df.to_csv(os.path.join(save_dir, "fred_indicators.csv"))
        print(f"\n→ Saved FRED data: {fred_df.shape[0]} rows × {fred_df.shape[1]} columns")
    else:
        print("\n⚠ No FRED data fetched. Check network connectivity.")
    
    # ── Yahoo Finance ──
    print("\n" + "=" * 50)
    print("Fetching Yahoo Finance Market Data...")
    print("=" * 50)
    yahoo_series_list = []
    for name, meta in YAHOO_INDICATORS.items():
        s = fetch_yahoo_ticker(meta['ticker'], name, start_date=start_date)
        if s is not None:
            yahoo_series_list.append(s)
    
    yahoo_df = pd.DataFrame()
    if yahoo_series_list:
        yahoo_df = pd.concat(yahoo_series_list, axis=1)
        yahoo_df = yahoo_df.asfreq('D')
        yahoo_df = yahoo_df.ffill().bfill()
        yahoo_df.to_csv(os.path.join(save_dir, "yahoo_indicators.csv"))
        print(f"\n→ Saved Yahoo data: {yahoo_df.shape[0]} rows × {yahoo_df.shape[1]} columns")
    else:
        print("\n⚠ No Yahoo data fetched. Try: pip install --upgrade yfinance")
    
    return {"fred": fred_df, "yahoo": yahoo_df}


def load_all_indicators(data_dir="data/raw"):
    """
    Load previously fetched indicator CSVs and merge into a unified DataFrame.
    
    Returns:
        pd.DataFrame with DatetimeIndex and all indicators as columns.
    """
    fred_path = os.path.join(data_dir, "fred_indicators.csv")
    yahoo_path = os.path.join(data_dir, "yahoo_indicators.csv")
    
    frames = []
    
    if os.path.exists(fred_path):
        fred = pd.read_csv(fred_path, parse_dates=['DATE'], index_col='DATE')
        frames.append(fred)
        print(f"  Loaded FRED: {fred.shape}")
    
    if os.path.exists(yahoo_path):
        yahoo = pd.read_csv(yahoo_path, parse_dates=['DATE'], index_col='DATE')
        frames.append(yahoo)
        print(f"  Loaded Yahoo: {yahoo.shape}")
    
    if not frames:
        raise FileNotFoundError(f"No data files found in {data_dir}/. Run fetch_all_indicators() first.")
    
    merged = pd.concat(frames, axis=1)
    merged = merged.ffill().bfill()
    merged.dropna(axis=1, how='all', inplace=True)
    
    print(f"  Merged: {merged.shape[0]} rows × {merged.shape[1]} indicators")
    return merged

"""
Global Twin — Multi-Country Data Architecture

Provides per-country data slicing, cross-country reference tables,
and a unified loader that queries by country + indicator.
"""

import os
import pandas as pd
import numpy as np
from src.data.indicators import (
    FRED_INDICATORS, YAHOO_INDICATORS,
    COUNTRY_INDICATORS, GLOBAL_INDICATORS, COUNTRIES
)


class CountryDataManager:
    """
    Manages economic data organized by country.
    
    Provides:
        - Per-country DataFrames with country-specific + global indicators
        - Cross-country reference table linking equivalent indicators
        - Query API: get data by country, by indicator, or by both
    """
    
    # Cross-country equivalent indicator mapping
    # Maps a canonical concept → per-country indicator name
    CROSS_COUNTRY_REF = {
        "GDP_GROWTH": {
            "US": "US_GDP_GROWTH",
            "EU": "EU_GDP_GROWTH",
            "CN": "CN_GDP_GROWTH",
            "IN": "IN_GDP_GROWTH",
            "JP": "JP_GDP_GROWTH",
        },
        "CPI_INFLATION": {
            "US": "US_CPI_INFLATION",
            "EU": "EU_CPI_INFLATION",
            "CN": "CN_CPI_INFLATION",
            "IN": "IN_CPI_INFLATION",
            "JP": "JP_CPI_INFLATION",
        },
        "UNEMPLOYMENT": {
            "US": "US_UNEMPLOYMENT",
            "EU": "EU_UNEMPLOYMENT",
            "JP": "JP_UNEMPLOYMENT",
        },
        "CENTRAL_BANK_RATE": {
            "US": "US_FED_RATE",
            "EU": "EU_ECB_RATE",
            "JP": "JP_BOJ_RATE",
        },
    }
    
    def __init__(self, data_dir="data/raw"):
        """
        Initialize the CountryDataManager.
        
        Args:
            data_dir: Directory containing fred_indicators.csv and yahoo_indicators.csv
        """
        self.data_dir = data_dir
        self._unified_df = None
        self._country_cache = {}
        
    def load(self):
        """Load raw CSVs and build the unified DataFrame."""
        fred_path = os.path.join(self.data_dir, "fred_indicators.csv")
        yahoo_path = os.path.join(self.data_dir, "yahoo_indicators.csv")
        
        frames = []
        
        if os.path.exists(fred_path):
            fred = pd.read_csv(fred_path, parse_dates=['DATE'], index_col='DATE')
            frames.append(fred)
        
        if os.path.exists(yahoo_path):
            yahoo = pd.read_csv(yahoo_path, parse_dates=['DATE'], index_col='DATE')
            frames.append(yahoo)
        
        if not frames:
            raise FileNotFoundError(
                f"No data files in {self.data_dir}/. Run: python scripts/fetch_data.py"
            )
        
        self._unified_df = pd.concat(frames, axis=1)
        self._unified_df = self._unified_df.ffill().bfill()
        self._unified_df.dropna(axis=1, how='all', inplace=True)
        
        # Build per-country caches
        self._country_cache = {}
        for code in COUNTRIES:
            self._country_cache[code] = self._build_country_df(code)
        
        return self
    
    def load_synthetic(self):
        """
        Load synthetic fallback data when real API data isn't available.
        Useful for development and testing.
        """
        np.random.seed(42)
        dates = pd.date_range(start='2018-01-01', end='2023-12-31', freq='D')
        n = len(dates)
        
        data = {}
        
        # Generate correlated synthetic data per indicator
        base_oil = np.cumsum(np.random.normal(0, 0.5, n)) + 60
        base_gas = np.cumsum(np.random.normal(0, 0.1, n)) + 3
        base_gold = np.cumsum(np.random.normal(0, 2, n)) + 1800
        
        # Global commodities & markets
        data['CRUDE_OIL'] = base_oil + np.random.normal(0, 2, n)
        data['NATURAL_GAS'] = base_gas + np.random.normal(0, 0.2, n)
        data['GOLD'] = base_gold + np.random.normal(0, 10, n)
        data['COPPER'] = np.cumsum(np.random.normal(0, 0.02, n)) + 4
        data['WHEAT'] = np.cumsum(np.random.normal(0, 0.5, n)) + 500
        data['SP500'] = np.cumsum(np.random.normal(0.02, 1, n)) + 3000
        data['VIX'] = np.abs(np.cumsum(np.random.normal(0, 0.3, n)) + 20)
        data['BALTIC_DRY_INDEX'] = np.abs(np.cumsum(np.random.normal(0, 5, n)) + 1500)
        data['SEMICONDUCTOR_IDX'] = np.cumsum(np.random.normal(0, 1, n)) + 2500
        
        # Exchange rates
        data['EUR_USD'] = np.cumsum(np.random.normal(0, 0.001, n)) + 1.10
        data['CNY_USD'] = np.cumsum(np.random.normal(0, 0.002, n)) + 7.0
        data['INR_USD'] = np.cumsum(np.random.normal(0, 0.01, n)) + 75
        data['JPY_USD'] = np.cumsum(np.random.normal(0, 0.05, n)) + 110
        
        # US indicators
        data['US_GDP_GROWTH'] = np.random.normal(2.5, 0.5, n)
        data['US_UNEMPLOYMENT'] = np.clip(np.cumsum(np.random.normal(0, 0.01, n)) + 4.0, 2, 15)
        data['US_CPI_INFLATION'] = np.cumsum(np.random.normal(0.005, 0.02, n)) + 250
        data['US_FED_RATE'] = np.clip(np.cumsum(np.random.normal(0, 0.005, n)) + 2.0, 0, 6)
        data['US_TRADE_BALANCE'] = np.random.normal(-60000, 5000, n)
        data['US_CONSUMER_CONFIDENCE'] = np.clip(np.cumsum(np.random.normal(0, 0.2, n)) + 95, 50, 120)
        data['US_MANUFACTURING_PMI'] = np.random.normal(12800, 200, n)
        
        # EU indicators
        data['EU_GDP_GROWTH'] = np.cumsum(np.random.normal(0.001, 0.005, n)) + 2800000
        data['EU_UNEMPLOYMENT'] = np.clip(np.cumsum(np.random.normal(0, 0.005, n)) + 7.0, 4, 12)
        data['EU_CPI_INFLATION'] = np.cumsum(np.random.normal(0.003, 0.015, n)) + 105
        data['EU_ECB_RATE'] = np.clip(np.cumsum(np.random.normal(0, 0.003, n)) + 0.5, -1, 5)
        
        # China indicators  
        data['CN_GDP_GROWTH'] = np.cumsum(np.random.normal(0.02, 0.01, n)) + 14000000000000
        data['CN_CPI_INFLATION'] = np.random.normal(2.5, 0.8, n)
        
        # India indicators
        data['IN_GDP_GROWTH'] = np.cumsum(np.random.normal(0.01, 0.008, n)) + 2700000000000
        data['IN_CPI_INFLATION'] = np.random.normal(5.0, 1.0, n)
        
        # Japan indicators
        data['JP_GDP_GROWTH'] = np.cumsum(np.random.normal(0.005, 0.005, n)) + 5000000000000
        data['JP_UNEMPLOYMENT'] = np.clip(np.cumsum(np.random.normal(0, 0.003, n)) + 2.5, 1, 6)
        data['JP_CPI_INFLATION'] = np.random.normal(1.0, 0.5, n)
        data['JP_BOJ_RATE'] = np.clip(np.cumsum(np.random.normal(0, 0.001, n)) - 0.1, -0.5, 1)
        
        self._unified_df = pd.DataFrame(data, index=dates)
        self._unified_df.index.name = 'DATE'
        
        # Build country caches
        self._country_cache = {}
        for code in COUNTRIES:
            self._country_cache[code] = self._build_country_df(code)
        
        return self
    
    def _build_country_df(self, country_code):
        """Build a DataFrame for a single country: its own indicators + global ones."""
        country_cols = COUNTRY_INDICATORS.get(country_code, [])
        global_cols = GLOBAL_INDICATORS
        
        all_cols = []
        for col in country_cols + global_cols:
            if col in self._unified_df.columns:
                all_cols.append(col)
        
        if not all_cols:
            return pd.DataFrame()
        
        return self._unified_df[all_cols].copy()
    
    def get_country_data(self, country_code):
        """
        Get the full DataFrame for a specific country.
        Includes country-specific indicators + global indicators.
        
        Args:
            country_code: 'US', 'EU', 'CN', 'IN', or 'JP'
        
        Returns:
            pd.DataFrame with DatetimeIndex
        """
        if country_code not in self._country_cache:
            raise ValueError(f"Unknown country: {country_code}. Valid: {list(COUNTRIES.keys())}")
        return self._country_cache[country_code]
    
    def get_indicator(self, indicator_name):
        """
        Get a single indicator series across all available dates.
        
        Args:
            indicator_name: e.g., 'CRUDE_OIL', 'US_UNEMPLOYMENT'
        
        Returns:
            pd.Series with DatetimeIndex
        """
        if self._unified_df is None:
            raise RuntimeError("Data not loaded. Call load() or load_synthetic() first.")
        if indicator_name not in self._unified_df.columns:
            raise ValueError(f"Unknown indicator: {indicator_name}")
        return self._unified_df[indicator_name]
    
    def get_cross_country(self, concept):
        """
        Get equivalent indicators across all countries for a concept.
        
        Args:
            concept: e.g., 'GDP_GROWTH', 'CPI_INFLATION', 'UNEMPLOYMENT'
        
        Returns:
            pd.DataFrame where each column is a country's version of that indicator
        """
        if concept not in self.CROSS_COUNTRY_REF:
            raise ValueError(
                f"Unknown concept: {concept}. "
                f"Valid: {list(self.CROSS_COUNTRY_REF.keys())}"
            )
        
        mapping = self.CROSS_COUNTRY_REF[concept]
        cols = {}
        for country, indicator in mapping.items():
            if indicator in self._unified_df.columns:
                cols[country] = self._unified_df[indicator]
        
        return pd.DataFrame(cols)
    
    def get_all_data(self):
        """Return the full unified DataFrame with all indicators."""
        if self._unified_df is None:
            raise RuntimeError("Data not loaded. Call load() or load_synthetic() first.")
        return self._unified_df
    
    def list_indicators(self, country_code=None):
        """
        List available indicators, optionally filtered by country.
        
        Args:
            country_code: If provided, shows only that country's indicators + global.
        
        Returns:
            list of indicator name strings
        """
        if country_code:
            df = self.get_country_data(country_code)
            return list(df.columns)
        return list(self._unified_df.columns) if self._unified_df is not None else []
    
    def summary(self):
        """Print a summary of loaded data."""
        if self._unified_df is None:
            print("No data loaded.")
            return
        
        print(f"Unified DataFrame: {self._unified_df.shape[0]} days × {self._unified_df.shape[1]} indicators")
        print(f"Date range: {self._unified_df.index.min().date()} to {self._unified_df.index.max().date()}")
        print()
        for code, name in COUNTRIES.items():
            df = self._country_cache.get(code, pd.DataFrame())
            print(f"  {name} ({code}): {df.shape[1]} indicators")

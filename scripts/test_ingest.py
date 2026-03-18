import os
import pandas as pd
import numpy as np
import sys

# Ensure src module is visible
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.ingest import load_and_clean_data

def generate_mock_data(filepath):
    dates = pd.date_range(start='2020-01-01', end='2020-01-31', freq='D')
    # drop some dates to test asfreq('D') gap filling
    dates = dates.drop([pd.to_datetime('2020-01-10'), pd.to_datetime('2020-01-15')])
    
    data = {
        'Date': dates,
        'Oil_Price': np.random.normal(60, 5, len(dates)),
        'Shipping_Cost': np.random.normal(1500, 200, len(dates))
    }
    df = pd.DataFrame(data)
    
    # Introduce explicit NaNs
    df.loc[5, 'Oil_Price'] = np.nan
    df.loc[6, 'Oil_Price'] = np.nan
    
    df.to_csv(filepath, index=False)
    print(f"Created mock data at {filepath}")

if __name__ == "__main__":
    test_file = "test_mock_data.csv"
    try:
        generate_mock_data(test_file)
        
        print("\nTesting load_and_clean_data...")
        clean_df = load_and_clean_data(test_file)
        
        if clean_df is not None:
            print(f"\nCleaned DataFrame Shape: {clean_df.shape}")
            print("\nFirst 5 rows:")
            print(clean_df.head(7))
            
            # Validate output is daily and has No NaNs
            assert clean_df.index.freq == 'D', "Frequency is not Daily (D)"
            assert clean_df.isna().sum().sum() == 0, "DataFrame contains NaNs after cleaning"
            print("\nSUCCESS: Data ingestion and cleaning passed basic validation.")
            sys.exit(0)
        else:
            print("\nFAILURE: Returned DataFrame is None")
            sys.exit(1)
            
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)

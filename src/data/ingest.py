import pandas as pd
import numpy as np
import warnings

def load_and_clean_data(filepath, threshold=10):
    """
    Load data from a CSV or Excel file, normalize dates, and handle missing values.
    
    Args:
        filepath (str): Path to the data file.
        threshold (int): Maximum allowed consecutive missing values before warning.
        
    Returns:
        pd.DataFrame: Cleaned data ready for ML training.
    """
    try:
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith('.xlsx') or filepath.endswith('.xls'):
            df = pd.read_excel(filepath)
        else:
            raise ValueError("Unsupported file format. Please use CSV or Excel.")
            
        # Standardize dates
        # Assume there's a 'date' or 'Date' column
        date_cols = [col for col in df.columns if col.lower() == 'date']
        if date_cols:
            date_col = date_cols[0]
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
            # Reindex to Daily frequency which introduces NaNs for missing days
            # Then we use ffill
            df = df.asfreq('D')
            
            # Check for large gaps before forward filling
            missing_stats = df.isna().sum()
            for col, missing in missing_stats.items():
                if missing > threshold:
                    print(f"WARNING: Feature '{col}' has {missing} missing values. Consider checking data source.")
            
            # Forward fill missing values
            df = df.ffill()
            # If any missing at the beginning, backward fill
            df = df.bfill()
        else:
            print("WARNING: No date column found. Returning dataframe as-is (with basic ffill).")
            df = df.ffill().bfill()
            
        return df

    except Exception as e:
        print(f"Error loading and cleaning data: {e}")
        return None

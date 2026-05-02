import pandas as pd
from datetime import time, datetime, timedelta
import numpy as np

def load_tick_data(file_path_or_config: str = None) -> pd.DataFrame:
    """
    Loads tick data from a file or generates sample data.

    Args:
        file_path_or_config: Path to the data file or a configuration string.
                             If None or "simulate", generates sample data.

    Returns:
        A pandas DataFrame with tick data.
    """
    if file_path_or_config is None or file_path_or_config == "simulate":
        # Generate sample data for a few days
        dates = ["2022-06-13", "2022-06-14"]
        all_ticks = []
        base_price = 100.0

        for date_str in dates:
            start_timestamp = datetime.strptime(f"{date_str} 09:00:00", "%Y-%m-%d %H:%M:%S")
            end_timestamp = datetime.strptime(f"{date_str} 14:00:00", "%Y-%m-%d %H:%M:%S")
            current_timestamp = start_timestamp

            while current_timestamp <= end_timestamp:
                # Simulate price fluctuations
                bid_price = base_price + np.random.uniform(-0.5, 0.5)
                ask_price = bid_price + np.random.uniform(0.01, 0.1) # Ensure ask > bid
                bid_qty = np.random.randint(1, 100)
                ask_qty = np.random.randint(1, 100)

                all_ticks.append({
                    "timestamp": current_timestamp,
                    "best_bid_price": bid_price,
                    "best_ask_price": ask_price,
                    "best_bid_qty": bid_qty,
                    "best_ask_qty": ask_qty,
                })
                # Variable time between ticks, ensuring some granularity
                current_timestamp += timedelta(seconds=np.random.randint(1, 6), milliseconds=np.random.randint(0,1000)) 
            base_price += np.random.uniform(-5, 5) # Change base price for the next day

        df = pd.DataFrame(all_ticks)
        if df.empty: # Should not happen with simulation settings
             df = pd.DataFrame(columns=['timestamp', 'best_bid_price', 'best_ask_price', 'best_bid_qty', 'best_ask_qty'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    else:
        # Placeholder for loading data from a file
        # This part will be implemented based on the actual data format
        # For now, assume it's a CSV, can be adapted later
        try:
            df = pd.read_csv(file_path_or_config)
            # Basic validation for required columns
            required_cols = ['timestamp', 'best_bid_price', 'best_ask_price', 'best_bid_qty', 'best_ask_qty']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"CSV file must contain columns: {', '.join(required_cols)}")
            df['timestamp'] = pd.to_datetime(df['timestamp']) # Ensure timestamp is datetime object
            return df
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {file_path_or_config}")
        except Exception as e:
            raise Exception(f"Error loading data from {file_path_or_config}: {e}")


def filter_by_time(df: pd.DataFrame, start_time_str: str = "09:30:00", end_time_str: str = "13:00:00") -> pd.DataFrame:
    """
    Filters the DataFrame to include only rows where the time part of 'timestamp'
    is between start_time_str and end_time_str (inclusive).
    """
    if df.empty:
        return df.copy() # Return an empty copy with same columns

    start_t = time.fromisoformat(start_time_str)
    end_t = time.fromisoformat(end_time_str)
    
    # Ensure timestamp column is datetime type
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    df_time = df['timestamp'].dt.time
    return df[(df_time >= start_t) & (df_time <= end_t)].copy()


def normalize_prices(df: pd.DataFrame, price_dev: float = 64.0) -> pd.DataFrame:
    """
    Normalizes price columns based on the opening price of each day.
    The opening price is defined as the first available mid-price (avg of best bid and ask)
    at or after 09:30:00 for that day.
    """
    if df.empty:
        df_copy = df.copy()
        df_copy['norm_best_bid_price'] = np.nan
        df_copy['norm_best_ask_price'] = np.nan
        return df_copy

    df_normalized = df.copy()
    
    # Ensure timestamp column is datetime type
    if not pd.api.types.is_datetime64_any_dtype(df_normalized['timestamp']):
        df_normalized['timestamp'] = pd.to_datetime(df_normalized['timestamp'])
        
    df_normalized['date'] = df_normalized['timestamp'].dt.date

    opening_prices = {}
    # Group by date and find the opening price for each day
    for date_val, group in df_normalized.groupby('date'):
        # Filter for market hours (e.g., 09:30 onwards for opening price calculation)
        market_open_filter_time = time.fromisoformat("09:30:00")
        
        # Sort by timestamp to ensure the first price is correctly identified
        group_sorted = group.sort_values(by='timestamp')
        
        market_hours_data = group_sorted[group_sorted['timestamp'].dt.time >= market_open_filter_time]
        
        if not market_hours_data.empty:
            # P_open is the mid-price of the first tick at or after 09:30
            p_open = (market_hours_data['best_bid_price'].iloc[0] + market_hours_data['best_ask_price'].iloc[0]) / 2.0
            opening_prices[date_val] = p_open
        else:
            # If no data at or after 09:30 for this day in the input df,
            # then we cannot determine an opening price for normalization.
            # These rows will have NaN for normalized prices.
            opening_prices[date_val] = np.nan

    # Apply normalization
    df_normalized['norm_best_bid_price'] = np.nan
    df_normalized['norm_best_ask_price'] = np.nan
    df_normalized['P_open'] = np.nan
    df_normalized['P_dev'] = np.nan


    for date_val_loop, p_open_val in opening_prices.items():
        if pd.notna(p_open_val): # Proceed only if P_open is not NaN
            day_mask = (df_normalized['date'] == date_val_loop)
            df_normalized.loc[day_mask, 'norm_best_bid_price'] = \
                (df_normalized.loc[day_mask, 'best_bid_price'] - p_open_val) / price_dev
            df_normalized.loc[day_mask, 'norm_best_ask_price'] = \
                (df_normalized.loc[day_mask, 'best_ask_price'] - p_open_val) / price_dev
            df_normalized.loc[day_mask, 'P_open'] = p_open_val
            df_normalized.loc[day_mask, 'P_dev'] = price_dev
                
    return df_normalized.drop(columns=['date'])


if __name__ == '__main__':
    # Example Usage
    print("Simulating data...")
    raw_data = load_tick_data("simulate")
    print(f"Generated {len(raw_data)} ticks.")
    print("Raw data sample (first 5 rows):")
    print(raw_data.head())
    print(f"Raw data dtypes:\n{raw_data.dtypes}")


    print("\nFiltering data between 09:30:00 and 13:00:00...")
    # Test filter_by_time with default times
    filtered_data_default_times = filter_by_time(raw_data.copy())
    print(f"Filtered data (default times) has {len(filtered_data_default_times)} ticks.")
    if not filtered_data_default_times.empty:
        print("Filtered data (default times) sample:")
        print(filtered_data_default_times.head())
        # Verify time boundaries
        print(f"Min time in filtered data: {filtered_data_default_times['timestamp'].dt.time.min()}")
        print(f"Max time in filtered data: {filtered_data_default_times['timestamp'].dt.time.max()}")
    else:
        print("Filtered data (default times) is empty.")

    
    print("\nNormalizing prices for the filtered data (default times)...")
    if not filtered_data_default_times.empty:
        normalized_data_default_times = normalize_prices(filtered_data_default_times.copy())
        print("Normalized data (default times) sample:")
        print(normalized_data_default_times[['timestamp', 'best_bid_price', 'best_ask_price', 'norm_best_bid_price', 'norm_best_ask_price', 'P_open', 'P_dev']].head())
        print("\nChecking for NaNs in normalized prices (default times data, first 5 rows):")
        print(normalized_data_default_times[['norm_best_bid_price', 'norm_best_ask_price', 'P_open', 'P_dev']].head().isnull().sum())
        print("\nNormalized Bid Price (default times) distribution:")
        print(normalized_data_default_times['norm_best_bid_price'].describe())
        print("\nNormalized Ask Price (default times) distribution:")
        print(normalized_data_default_times['norm_best_ask_price'].describe())

        # Check if any day's data was entirely removed by filtering, leading to NaNs after normalization attempt
        # This is more about how filter + normalize interact.
        # If filter_by_time results in a day having NO data within 09:30-13:00,
        # but that day HAD data outside it, normalize_prices (if called on raw_data)
        # might behave differently than if called on filtered_data.
        # The current design is to call normalize_prices on *already filtered* data.
        
        # Verify that all normalized prices are based on an opening price at or after 09:30
        # For each day in normalized_data_default_times, find P_open used.
        # This requires re-calculating P_open as it's done inside normalize_prices.
        temp_check_df = filtered_data_default_times.copy()
        temp_check_df['date'] = temp_check_df['timestamp'].dt.date
        for date_val, group in temp_check_df.groupby('date'):
            market_open_filter_time = time.fromisoformat("09:30:00")
            group_sorted = group.sort_values(by='timestamp')
            market_hours_data = group_sorted[group_sorted['timestamp'].dt.time >= market_open_filter_time]
            if not market_hours_data.empty:
                p_open_check = (market_hours_data['best_bid_price'].iloc[0] + market_hours_data['best_ask_price'].iloc[0]) / 2.0
                first_timestamp_for_open = market_hours_data['timestamp'].iloc[0]
                print(f"Day {date_val}: P_open for normalization was {p_open_check:.2f} (from tick at {first_timestamp_for_open.time()})")
                # Check a sample normalized value
                original_bid = normalized_data_default_times[normalized_data_default_times['timestamp'] == first_timestamp_for_open]['best_bid_price'].iloc[0]
                norm_bid = normalized_data_default_times[normalized_data_default_times['timestamp'] == first_timestamp_for_open]['norm_best_bid_price'].iloc[0]
                expected_norm_bid = (original_bid - p_open_check) / 64.0
                # print(f"  Sample: orig_bid={original_bid:.2f}, norm_bid={norm_bid:.4f}, expected_norm_bid={expected_norm_bid:.4f}")
                assert abs(norm_bid - expected_norm_bid) < 1e-9, "Mismatch in normalization check"

            else:
                print(f"Day {date_val}: No data at or after 09:30 in the filtered set. Normalized prices should be NaN.")
                assert normalized_data_default_times[normalized_data_default_times['timestamp'].dt.date == date_val]['norm_best_bid_price'].isnull().all()


    else:
        print("Filtered data (default times) is empty, skipping normalization.")


    # Test case: Data for a day entirely outside filtering window
    print("\nTesting with data entirely outside 09:30-13:00 window (e.g., 08:00-09:00)...")
    early_data_list = []
    early_date_dt = datetime(2022, 6, 16)
    base_price_early = 120.0
    start_ts_early = datetime.combine(early_date_dt, time(8,0,0))
    end_ts_early = datetime.combine(early_date_dt, time(9,0,0))
    curr_ts_early = start_ts_early
    while curr_ts_early <= end_ts_early:
        bid = base_price_early + np.random.uniform(-0.1, 0.1)
        ask = bid + np.random.uniform(0.01, 0.03)
        early_data_list.append({
            "timestamp": curr_ts_early, "best_bid_price": bid, "best_ask_price": ask,
            "best_bid_qty": 30, "best_ask_qty": 30
        })
        curr_ts_early += timedelta(seconds=np.random.randint(10, 30))
    
    early_df = pd.DataFrame(early_data_list)
    early_df['timestamp'] = pd.to_datetime(early_df['timestamp']) # Ensure dtype
    print(f"Generated {len(early_df)} early ticks for {early_date_dt.date()}.")
    
    filtered_early_data = filter_by_time(early_df.copy()) # Default 09:30-13:00 filter
    print(f"Filtered early data has {len(filtered_early_data)} ticks (should be 0).")
    assert filtered_early_data.empty

    if not filtered_early_data.empty: # This block should ideally not run
        normalized_early_data = normalize_prices(filtered_early_data.copy())
        print("Normalized early data sample (should not appear if logic is correct):")
        print(normalized_early_data.head())
    else:
        print("Filtered early data is empty. Normalizing an empty DF...")
        normalized_empty_from_early = normalize_prices(filtered_early_data.copy())
        assert normalized_empty_from_early.empty
        assert 'norm_best_bid_price' in normalized_empty_from_early.columns
        print("Normalization of empty DF from early data is also empty, with norm columns.")


    # Test with an empty dataframe from the start
    print("\nTesting with initially empty DataFrame...")
    empty_df = pd.DataFrame(columns=['timestamp', 'best_bid_price', 'best_ask_price', 'best_bid_qty', 'best_ask_qty'])
    empty_df['timestamp'] = pd.to_datetime(empty_df['timestamp']) # Ensure dtype for dt accessor

    filtered_empty = filter_by_time(empty_df.copy())
    print(f"Filtered empty DataFrame has {len(filtered_empty)} rows.")
    assert filtered_empty.empty
    
    normalized_empty = normalize_prices(filtered_empty.copy())
    print(f"Normalized empty DataFrame has {len(normalized_empty)} rows.")
    assert normalized_empty.empty
    assert 'norm_best_bid_price' in normalized_empty.columns
    assert 'norm_best_ask_price' in normalized_empty.columns
    assert 'P_open' in normalized_empty.columns
    assert 'P_dev' in normalized_empty.columns
    print(normalized_empty.head())
    
    print("\nAll example usage tests in data_loader.py complete.")

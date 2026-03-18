import pandas as pd
import numpy as np
import os

def process_data(input_path='../data/raw/acled_manipur_synthetic.csv', output_path='../data/processed/manipur_processed.csv'):
    """Reads raw ACLED data, cleans columns, and engineers features."""
    print("Loading data...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"File not found: {input_path}")
        return None

    # Clean columns: select relevant ones
    cols_to_keep = ['event_date', 'year', 'admin2', 'event_type', 
                    'actor1', 'actor2', 'fatalities', 'latitude', 'longitude']
    df = df[cols_to_keep].copy()
    
    # Rename admin2 to district for clarity
    df.rename(columns={'admin2': 'district'}, inplace=True)
    
    # Feature Engineering: month, year, season
    df['event_date'] = pd.to_datetime(df['event_date'])
    df['month'] = df['event_date'].dt.month
    df['year_month'] = df['event_date'].dt.to_period('M').astype(str)
    
    def get_season(month):
        if month in [12, 1, 2]: return 'Winter'
        elif month in [3, 4, 5]: return 'Pre-Monsoon'
        elif month in [6, 7, 8, 9]: return 'Monsoon'
        else: return 'Post-Monsoon'
        
    df['season'] = df['month'].apply(get_season)
    
    # Rolling Event Count 30d per district
    # To do this, sort by district and date
    df = df.sort_values(by=['district', 'event_date']).reset_index(drop=True)
    # create a dummy column to count
    df['event_count'] = 1
    
    # Compute 30-day rolling count per district
    df.set_index('event_date', inplace=True)
    rolling_counts = df.groupby('district')['event_count'].rolling('30D').count().reset_index()
    rolling_counts.rename(columns={'event_count': 'rolling_event_count_30d'}, inplace=True)
    
    df = df.reset_index()
    # Merge back
    df = pd.merge(df, rolling_counts, on=['district', 'event_date'], how='left')
    
    # Fill empty
    df['rolling_event_count_30d'].fillna(0, inplace=True)
    
    # Save processed data
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")
    print(f"Total shape: {df.shape}")
    
    return df

if __name__ == "__main__":
    
    os.makedirs('../data/processed', exist_ok=True)
    # Ensure run from src folder
    if os.path.exists('data/raw/acled_manipur_synthetic.csv'):
        # we are in root dir
        process_data('data/raw/acled_manipur_synthetic.csv', 'data/processed/manipur_processed.csv')
    else:
        # we are in src dir
        process_data()

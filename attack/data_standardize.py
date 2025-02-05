import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def split_and_standardize_data(input_file='dataset/ETTh1_original.csv', 
                             train_output='attack/data/ETTh1_train.csv',
                             test_output='attack/data/ETTh1_test.csv',
                             params_output='attack/data/standardization_params.csv'):
    """
    Split ETTh1 dataset into train (80%) and test (20%) sets
    """
    # Read data
    print(f"Reading data from {input_file}")
    df = pd.read_csv(input_file)
    
    # Rename columns to match our format
    df = df.rename(columns={'date': 'ds', 'OT': 'y'})
    
    # Add unique_id column required by models
    df['unique_id'] = 1
    
    # Convert date to datetime
    df['ds'] = pd.to_datetime(df['ds'])
    
    # Calculate split point
    total_length = len(df)
    train_size = int(total_length * 0.8)  # 80% for training
    
    # Split data
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()
    
    print(f"Data split sizes:")
    print(f"Train: {len(train_df)} ({len(train_df)/total_length*100:.1f}%)")
    print(f"Test: {len(test_df)} ({len(test_df)/total_length*100:.1f}%)")
    
    # Calculate standardization parameters from training data
    mean = train_df['y'].mean()
    std = train_df['y'].std()
    print(f"\nStandardization parameters:")
    print(f"Mean: {mean:.4f}")
    print(f"Std: {std:.4f}")
    print(f"Attack scale (2% of mean): {mean * 0.02:.4f}")
    
    # Save standardization parameters
    params_df = pd.DataFrame({
        'parameter': ['mean', 'std', 'attack_scale'],
        'value': [mean, std, mean * 0.02]
    })
    params_df.to_csv(params_output, index=False)
    print(f"\nStandardization parameters saved to: {params_output}")
    
    # Save train and test sets
    train_df.to_csv(train_output, index=False)
    test_df.to_csv(test_output, index=False)
    print(f"Train data saved to: {train_output}")
    print(f"Test data saved to: {test_output}")
    
    return train_df, test_df, mean, std

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    import os
    os.makedirs('attack/data', exist_ok=True)
    
    # Split and save data
    train_df, test_df, mean, std = split_and_standardize_data()
    
    # Print some statistics
    print("\nData statistics:")
    print("Training set:")
    print(f"  Mean: {train_df['y'].mean():.4f}")
    print(f"  Std: {train_df['y'].std():.4f}")
    print(f"  Min: {train_df['y'].min():.4f}")
    print(f"  Max: {train_df['y'].max():.4f}")
    print("\nTest set:")
    print(f"  Mean: {test_df['y'].mean():.4f}")
    print(f"  Std: {test_df['y'].std():.4f}")
    print(f"  Min: {test_df['y'].min():.4f}")
    print(f"  Max: {test_df['y'].max():.4f}")
    
    print("\nTime range:")
    print(f"Train: from {train_df['ds'].min()} to {train_df['ds'].max()}")
    print(f"Test: from {test_df['ds'].min()} to {test_df['ds'].max()}") 
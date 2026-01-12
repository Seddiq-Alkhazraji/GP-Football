import pandas as pd
import numpy as np

# --- CONFIGURATION ---
INPUT_FILE = "scout_data.csv"
OUTPUT_FILE = "scout_data_weighted.csv"
DECAY_FACTOR = 0.75  # 1.0 = Latest season, 0.75 = Last season is worth 75% of this one

METADATA_COLS = [
    'Player', 'Team', 'Comp', 'Nation', 'Age', 'Born', 'Url', 
    'Season_End_Year', 'main_pos', 'secondary_pos_1', 'secondary_pos_2', 
    'role_1', 'role_2', 'role_3', 'role_4'
]

def calculate_weighted_stats(df):
    print("Loading data...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found.")
        return

    # 1. Identify Numeric Columns (The Stats)
    # We exclude metadata columns from the math
    stat_cols = [col for col in df.columns if col not in METADATA_COLS and df[col].dtype in ['float64', 'int64']]
    
    print(f"Processing {len(df['Player'].unique())} players...")
    
    # 2. Define the Weighted Average Function
    def weighted_avg(group):
        # Sort by year descending (Latest first)
        group = group.sort_values('Season_End_Year', ascending=False)
        
        # If player only has 1 season, return it as is
        if len(group) == 1:
            return group.iloc[0]
        
        # Calculate weights: [1.0, 0.75, 0.56, ...]
        years_ago = group['Season_End_Year'].max() - group['Season_End_Year']
        weights = DECAY_FACTOR ** years_ago
        
        # Create a container for the result
        result = group.iloc[0].copy() 
        
        # Calculate weighted average for stats
        for col in stat_cols:
            if col in group.columns and group[col].notna().any():
                # Weighted Mean Formula: Sum(Value * Weight) / Sum(Weights)
                valid_indices = group[col].notna()
                if valid_indices.sum() == 0:
                    result[col] = 0
                else:
                    vals = group.loc[valid_indices, col]
                    w = weights.loc[valid_indices]
                    result[col] = np.average(vals, weights=w)
        
        return result

    # 3. Apply to every player
    df_weighted = df.groupby('Player', group_keys=False).apply(weighted_avg)
    
    # 4. Save
    df_weighted.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Success! Weighted data saved to {OUTPUT_FILE}")
    print(f"Merged {len(df)} seasonal records into {len(df_weighted)} unique player profiles.")

if __name__ == "__main__":
    calculate_weighted_stats(None)
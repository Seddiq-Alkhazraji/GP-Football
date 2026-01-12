import pandas as pd
import numpy as np
from scipy import stats

# The 10 Key Metrics used in model
FEATURES = [
    'Gls_Standard_Per90', 'Ast_Standard_Per90', 'npxG_Per', 'xAG_Per',
    'PrgP_Per90', 'PrgC_Carries_Per90', 'TklW_Tackles_Per90', 
    'Int_Def_Per90', 'Won_percent_Aerial', 'SCA90_SCA'
]

def get_team_profile(df, team_name):
    """Calculates the average stats for a specific team."""
    team_df = df[df['Team'] == team_name]
    if team_df.empty:
        return None
    # Calculate mean of numeric columns only
    profile = team_df.select_dtypes(include=[np.number]).mean()
    return profile

def identify_team_needs(df, team_name, percentile_threshold=40):
    """
    Identifies metrics where the team is below the league average.
    Returns: {Metric: {'score': percentile_rank, 'severity': 'Critical'/'Moderate'}}
    """
    team_profile = get_team_profile(df, team_name)
    if team_profile is None: 
        return {}

    needs = {}
    
    for feature in FEATURES:
        if feature not in df.columns: continue
        
        # 1. Get all league values for this metric
        league_values = df[feature].dropna()
        team_val = team_profile.get(feature, 0)
        
        # 2. Calculate Team's Percentile Rank (0-100)
        rank = stats.percentileofscore(league_values, team_val, kind='weak')
        
        # 3. Identify Weaknesses (Bottom 40%)
        if rank < percentile_threshold:
            severity = "Critical" if rank < 20 else "Moderate"
            needs[feature] = {'score': rank, 'severity': severity}
            
    return needs

def calculate_fit_score(candidate_row, team_needs, df_context):
    """
    Calculates a 'Fit Score' (0-100) based on how well a candidate
    fixes the team's specific weaknesses.
    
    df_context: The full dataframe (needed to calculate percentiles)
    """
    if not team_needs:
        return 50.0 # Neutral fit if team has no major weaknesses
    
    weighted_score_sum = 0
    total_weights = 0
    
    for metric, info in team_needs.items():
        if metric not in candidate_row: continue
        
        # 1. Determine Weight 
        weight = 3.0 if info['severity'] == "Critical" else 1.0
        
        # 2. Get Player's Percentile in this metric
        player_val = candidate_row[metric]
        league_values = df_context[metric].dropna()
        player_percentile = stats.percentileofscore(league_values, player_val, kind='weak')
        
        # 3. Add to Score
        weighted_score_sum += (player_percentile * weight)
        total_weights += weight
        
    if total_weights == 0: return 50.0
    
    # Final Score 0-100
    return weighted_score_sum / total_weights
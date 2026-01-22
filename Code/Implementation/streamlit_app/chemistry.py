import pandas as pd
import numpy as np
import joblib
import os
from scipy import stats

# --- CONFIGURATION ---
MODEL_PATH = "chemistry_rf_model.pkl"
ENCODER_PATH = "role_encoder.pkl"

# Global Variables
CHEM_MODEL = None
ROLE_ENCODER = None

def load_chemistry_engine():
    """Loads the ML models."""
    global CHEM_MODEL, ROLE_ENCODER
    
    if os.path.exists(MODEL_PATH):
        if CHEM_MODEL is None:
            try:
                CHEM_MODEL = joblib.load(MODEL_PATH)
            except:
                print("❌ Failed to load Chemistry Model.")
    
    if os.path.exists(ENCODER_PATH):
        if ROLE_ENCODER is None:
            try:
                ROLE_ENCODER = joblib.load(ENCODER_PATH)
            except:
                print("❌ Failed to load Role Encoder.")
            
    return CHEM_MODEL is not None

def load_ratings_data(filepath):
    """Loads ratings and cleans columns."""
    if not os.path.exists(filepath): return pd.DataFrame()

    try:
        df = pd.read_csv(filepath, encoding='utf-8-sig')
    except:
        return pd.DataFrame()

    # Clean Columns
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    # Rename map
    rename_map = {
        'role': 'main_role', 'best_role': 'main_role', 'position': 'main_role',
        'off': 'off_rating', 'offensive': 'off_rating', 'attack': 'off_rating',
        'def': 'def_rating', 'defensive': 'def_rating', 'defense': 'def_rating',
        'name': 'player_name', 'player': 'player_name', 'full_name': 'player_name',
        'team': 'team_name', 'club': 'team_name', 'squad': 'team_name'
    }
    df = df.rename(columns=rename_map)
    
    if 'player_name' not in df.columns: return pd.DataFrame()

    # Numeric conversion
    for col in ['off_rating', 'def_rating']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(50)

    return df

# --- CALCULATE RATINGS FROM STATS ---
def estimate_ratings_from_stats(row):
    """
    If off_rating/def_rating are missing (0), estimate them from stats.
    This prevents the model from seeing '0 skill' players.
    """
    off = 50.0 # Default Average
    def_ = 50.0
    
    # 1. Estimate Offensive Rating
    if row.get('off_rating', 0) > 0:
        off = float(row['off_rating'])
    else:
        # Heuristic based on goals/assists/creation
        gls = float(row.get('Gls_Standard_Per90', 0))
        ast = float(row.get('Ast_Standard_Per90', 0))
        sca = float(row.get('SCA90_SCA', 0))
        # Simple weighted sum scaled to ~50-90 range
        raw_off = (gls * 25) + (ast * 20) + (sca * 5) + 50
        off = min(99, max(40, raw_off))

    # 2. Estimate Defensive Rating
    if row.get('def_rating', 0) > 0:
        def_ = float(row['def_rating'])
    else:
        tkl = float(row.get('TklW_Tackles_Per90', 0))
        inte = float(row.get('Int_Def_Per90', 0))
        rec = float(row.get('Recov_Per90', 0))
        raw_def = (tkl * 10) + (inte * 10) + (rec * 2) + 40
        def_ = min(99, max(40, raw_def))
        
    return off, def_

# --- PREDICTIVE CHEMISTRY ENGINE ---

def predict_pair_chemistry(p1_row, p2_row):
    """
    Predicts and SCALES the chemistry score to 0-100.
    """
    load_chemistry_engine()
    
    if CHEM_MODEL is None: return 50.0 

    # 1. Encode Roles
    try:
        r1 = str(p1_row.get('main_role', 'Unknown'))
        r2 = str(p2_row.get('main_role', 'Unknown'))
        
        if ROLE_ENCODER:
            r1_enc = ROLE_ENCODER.transform([r1])[0] if r1 in ROLE_ENCODER.classes_ else -1
            r2_enc = ROLE_ENCODER.transform([r2])[0] if r2 in ROLE_ENCODER.classes_ else -1
        else:
            r1_enc, r2_enc = 0, 0
    except:
        r1_enc, r2_enc = 0, 0

    # 2. Get/Estimate Ratings 
    off1, def1 = estimate_ratings_from_stats(p1_row)
    off2, def2 = estimate_ratings_from_stats(p2_row)

    # 3. Prepare Vector
    vec = np.array([[r1_enc, off1, def1, r2_enc, off2, def2]])

    # 4. Predict & SCALE
    try:
        raw_impact = CHEM_MODEL.predict(vec)[0]
        
        # --- SCALING LOGIC  ---
        # Raw impact is usually small (e.g., -0.05 to +0.05).
        # We map it to 0-100.
        # 0.0 impact -> 50 score (Average)
        # +0.05 impact -> 75 score (Good)
        # +0.10 impact -> 100 score (Perfect)
        
        scaled_score = 50 + (raw_impact * 500) 
        
        # Clamp between 1 and 99
        final_score = max(1.0, min(99.0, scaled_score))
        
        return final_score
        
    except Exception as e:
        print(f"Prediction Error: {e}")
        return 50.0

def get_squad_chemistry_map(target_player_row, squad_df):
    results = []
    
    # Identify Player ID or Name
    t_name = target_player_row.get('Player', target_player_row.get('player_name', 'Target'))
    
    for _, teammate in squad_df.iterrows():
        c_name = teammate.get('Player', teammate.get('player_name', 'Teammate'))
        if c_name == t_name: continue
        
        score = predict_pair_chemistry(target_player_row, teammate)
        
        results.append({
            'Teammate': c_name,
            'Role': teammate.get('main_role', 'Unknown'),
            'Chemistry_Score': score
        })
        
    if not results: return pd.DataFrame()
    return pd.DataFrame(results).sort_values('Chemistry_Score', ascending=False)

# --- UTILS FOR NEEDS ---
FEATURES = ['Gls_Standard_Per90', 'Ast_Standard_Per90', 'npxG_Per', 'xAG_Per', 'PrgP_Per90', 'PrgC_Carries_Per90', 'TklW_Tackles_Per90', 'Int_Def_Per90', 'Won_percent_Aerial', 'SCA90_SCA']

def get_team_profile(df, team_name):
    if 'Team' not in df.columns: return None
    team_df = df[df['Team'] == team_name]
    if team_df.empty: return None
    return team_df.select_dtypes(include=[np.number]).mean()

def identify_team_needs(df, team_name, percentile_threshold=40):
    team_profile = get_team_profile(df, team_name)
    if team_profile is None: return {}
    needs = {}
    for feature in FEATURES:
        if feature not in df.columns: continue
        league_values = df[feature].dropna()
        team_val = team_profile.get(feature, 0)
        rank = stats.percentileofscore(league_values, team_val, kind='weak')
        if rank < percentile_threshold:
            severity = "Critical" if rank < 20 else "Moderate"
            needs[feature] = {'score': rank, 'severity': severity}
    return needs

def calculate_fit_score(candidate_row, team_needs, df_context):
    if not team_needs: return 50.0
    weighted_score_sum = 0
    total_weights = 0
    for metric, info in team_needs.items():
        if metric not in candidate_row: continue
        weight = 3.0 if info['severity'] == "Critical" else 1.0
        player_val = candidate_row[metric]
        league_values = df_context[metric].dropna()
        player_percentile = stats.percentileofscore(league_values, player_val, kind='weak')
        weighted_score_sum += (player_percentile * weight)
        total_weights += weight
    if total_weights == 0: return 50.0
    return weighted_score_sum / total_weights
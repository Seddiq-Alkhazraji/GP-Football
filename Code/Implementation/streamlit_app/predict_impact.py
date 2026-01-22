import pandas as pd
import xgboost as xgb
import numpy as np
import json
import os

# --- CONFIGURATION ---
MODEL_FILE = "scout_elo_model_best.json"       
FEATURE_MAP_FILE = "model_features.json"       
PLAYER_DATA_FILE = "scout_outfield.csv"        
ELO_FILE = "elo_new.csv"                       


# --- GLOBAL LOADERS ---
_model = None
_model_cols = None
_df_players = None
_df_elo = None

def load_resources():
    """Loads model and data once to improve performance."""
    global _model, _model_cols, _df_players, _df_elo
    
    if _model is None:
        print("📂 Loading Engine Resources...")
        # 1. Load Model
        _model = xgb.XGBRegressor()
        _model.load_model(MODEL_FILE)
        
        # 2. Load Feature Map
        with open(FEATURE_MAP_FILE, "r") as f:
            _model_cols = json.load(f)
            
        # 3. Load Data
        _df_players = pd.read_csv(PLAYER_DATA_FILE)
        _df_players['search_key'] = _df_players['Player'].astype(str).str.lower().str.strip()
        
        _df_elo = pd.read_csv(ELO_FILE)
        print("✅ Resources Loaded.")

def clean_txt(text):
    if not isinstance(text, str): return ""
    return text.lower().strip()

def get_elo_for_team(team_name, df_elo):
    """Finds the Elo for a team (fuzzy match)."""
    search_key = clean_txt(team_name)
    
    # Identify relevant columns dynamically
    col_team = next((c for c in df_elo.columns if 'team' in c.lower() or 'club' in c.lower()), None)
    col_elo = next((c for c in df_elo.columns if 'elo' in c.lower()), None)
    
    if not col_team or not col_elo:
        print("⚠️  Elo file structure unknown. Defaulting to 1500.")
        return 1500

    # Search
    for idx, row in df_elo.iterrows():
        db_name = clean_txt(str(row[col_team]))
        if search_key in db_name or db_name in search_key:
            return row[col_elo]
            
    # Default if not found
    print(f"⚠️  Team '{team_name}' not found. Using Average (1500).")
    return 1500

def predict_transfer(player_name, target_team_name):
    """
    The main function to be called by App or CLI.
    Returns: dictionary with results.
    """
    # Ensure resources are loaded
    load_resources()
    
    # 1. Find Player
    search_key = clean_txt(player_name)
    player_row = _df_players[_df_players['search_key'] == search_key]
    
    if player_row.empty:
        player_row = _df_players[_df_players['search_key'].str.contains(search_key)]
        
    if player_row.empty:
        return {"error": f"Player '{player_name}' not found."}
    
    # Take the first match
    player_data = player_row.iloc[0].to_dict()
    real_name = player_data['Player']
    
    # 2. Get Team Elo
    current_elo = get_elo_for_team(target_team_name, _df_elo)
    
    # 3. Prepare Features
    player_data['Team_Start_Elo'] = current_elo
    
    # Create DataFrame and align columns
    input_df = pd.DataFrame([player_data])
    
    # Reindex aligns columns exactly as the model expects
    input_df = input_df.reindex(columns=_model_cols, fill_value=0)
    
    # 4. Predict
    pred_change = _model.predict(input_df)[0]
    final_elo = current_elo + pred_change
    
    # 5. Formulate Verdict
    verdict = "😐 NEUTRAL"
    if pred_change > 20: verdict = "🚀 GAME CHANGER"
    elif pred_change > 5: verdict = "✅ POSITIVE"
    elif pred_change < -5: verdict = "📉 NEGATIVE"
    elif pred_change < -20: verdict = "❌ FLOP RISK"
    
    return {
        "player": real_name,
        "team": target_team_name,
        "current_elo": round(current_elo),
        "predicted_impact": round(pred_change, 2),
        "forecasted_elo": round(final_elo),
        "verdict": verdict
    }

# --- CLI INTERFACE  ---
if __name__ == "__main__":
    print("\n🔮 --- SCOUT AI SIMULATOR (v2.0) --- 🔮")
    load_resources()
    
    while True:
        print("\n" + "="*40)
        p_in = input("⚽ Player Name (or 'q'): ").strip()
        if p_in.lower() == 'q': break
        t_in = input("🛡️  Target Team: ").strip()
        
        result = predict_transfer(p_in, t_in)
        
        if "error" in result:
            print(f"❌ {result['error']}")
        else:
            print(f"\n📊 RESULTS for {result['player']} ➡️  {result['team']}")
            print(f"   Current Elo:   {result['current_elo']}")
            print(f"   Impact:        {result['predicted_impact']:+}")
            print(f"   Forecast:      {result['forecasted_elo']}")
            print(f"   AI Verdict:    {result['verdict']}")
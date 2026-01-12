import pandas as pd
import xgboost as xgb
import numpy as np
import os

# --- CONFIGURATION ---
MODEL_FILE = "impact_model_hybrid.json"
STATS_FILE = "scout_data_weighted.csv"
CURRENT_ELO_FILE = "elo_end_24.csv"  


FEATURES = [
    'Gls_Standard_Per90', 'Ast_Standard_Per90', 'npxG_Per', 'xAG_Per',
    'PrgP_Per90', 'PrgC_Carries_Per90', 'TklW_Tackles_Per90', 
    'Int_Def_Per90', 'Won_percent_Aerial', 'SCA90_SCA'
]

def clean_name(name):
    if not isinstance(name, str): return ""
    return name.lower().strip()

def get_team_elo(team_name, df_elo):
    """Finds the Elo for a team (fuzzy match)."""
    search_key = clean_name(team_name)
    
    # 1. Try finding column name
    col_team = None
    for c in df_elo.columns:
        if 'team' in c.lower() or 'club' in c.lower():
            col_team = c
            break
            
    col_elo = None
    for c in df_elo.columns:
        if c.lower() == 'elo':
            col_elo = c
            break
            
    if not col_team or not col_elo:
        return 1500  

    # 2. Search for team
    for idx, row in df_elo.iterrows():
        db_name = clean_name(row[col_team])
        if search_key in db_name or db_name in search_key:
            return row[col_elo]
            
    return None

def predict_transfer():
    print("\n🔮 --- AI TRANSFER PREDICTOR --- 🔮")
    
    # 1. Load Model
    if not os.path.exists(MODEL_FILE):
        print("❌ Model not found. Run train_model.py first.")
        return
    
    model = xgb.XGBRegressor()
    model.load_model(MODEL_FILE)
    
    # 2. Load Data
    print("📂 Loading Player Stats & Team Data...")
    df_stats = pd.read_csv(STATS_FILE)
    df_elo = pd.read_csv(CURRENT_ELO_FILE)
    
    # Prepare lookup columns
    df_stats['search_name'] = df_stats['Player'].astype(str).str.lower()

    while True:
        print("\n" + "="*30)
        player_in = input("⚽ Enter Player Name (or 'q' to quit): ").strip()
        if player_in.lower() == 'q': break
        
        team_in = input("🛡️  Enter New Club: ").strip()
        
        # --- FIND PLAYER ---
        found_players = df_stats[df_stats['search_name'].str.contains(clean_name(player_in))]
        
        if found_players.empty:
            print(f"❌ Player '{player_in}' not found in scout database.")
            continue
        
        # Pick the first match (or most recent season if duplicates exist)
        player_row = found_players.iloc[0]
        real_name = player_row['Player']
        
        # --- FIND TEAM ELO ---
        current_elo = get_team_elo(team_in, df_elo)
        if current_elo is None:
            print(f"⚠️  Team '{team_in}' not found. Using default Elo 1500.")
            current_elo = 1500
        
        # --- PREDICT ---
        # Construct Input: [Team_Start_Elo, Feature1, Feature2, ...]
        stats_vector = player_row[FEATURES].fillna(0).values
        input_vector = np.concatenate([[current_elo], stats_vector])
        
        input_2d = np.array([input_vector])
        
        prediction = model.predict(input_2d)[0]
        
        # --- RESULT ---
        new_elo = current_elo + prediction
        sign = "+" if prediction > 0 else ""
        
        print(f"\n📊 PREDICTION RESULTS for {real_name} -> {team_in}")
        print(f"   ------------------------------------------")
        print(f"   🔹 Current Team Elo: {current_elo:.0f}")
        print(f"   🔹 Player Impact:    {sign}{prediction:.2f}")
        print(f"   🔹 Predicted Elo:    {new_elo:.0f}")
        
        if prediction > 5:
            print("   🚀 VERDICT: GAME CHANGER! Sign him immediately.")
        elif prediction > 0:
            print("   ✅ VERDICT: Good signing. Will improve the squad.")
        elif prediction > -5:
            print("   ⚠️ VERDICT: Risky. Might struggle to adapt.")
        else:
            print("   ❌ VERDICT: FLOP WARNING. Do not sign.")

if __name__ == "__main__":
    predict_transfer()
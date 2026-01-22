import pandas as pd
import numpy as np
import pickle
import json
import os
import warnings
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")

# --- IMPORT HELPERS ---
try:
    from predict_impact import predict_transfer, clean_txt
    from chemistry import get_squad_chemistry_map, load_chemistry_engine, get_target_squad
except ImportError:
    print("⚠️  Helpers not found. Make sure predict_impact.py and chemistry.py are in the folder.")

# --- CONFIGURATION ---
KNN_MODEL = "knn_similarity.pkl"
KNN_SCALER = "knn_scaler.pkl"
SCOUT_DATA = "scout_outfield.csv"

# --- 🧬 THE 24 DNA FEATURES ---
# These are the stats that define "Playstyle"
DNA_FEATURES = [
    'Gls_Standard_Per90', 'Ast_Standard_Per90', 'npxG_Per', 'xAG_Per', 
    'Sh_per_90_Standard', 'SoT_percent_Standard', 'G_per_Sh_Standard',
    'PrgP_Per90', 'PrgC_Carries_Per90', 'Att_Take_Per90', 'Succ_Take_Per90',
    'TklW_Tackles_Per90', 'Int_Def_Per90', 'Blocks_Blocks_Per90', 'Clr_Per90',
    'Won_percent_Aerial', 'SCA90_SCA', 'GCA90_GCA', 'Cmp_percent_Total',
    'Cmp_Short_Per90', 'Cmp_Medium_Per90', 'Cmp_Long_Per90', 
    'Att Pen_Touches_Per90', 'Recov_Per90'
]

# --- GLOBAL RESOURCES ---
_knn = None
_scaler = None
_df_scout = None

def train_new_knn_model():
    """Forces a retrain of the KNN model to fix feature mismatch errors."""
    print("\n🔄 RE-TRAINING KNN MODEL (Fixing Feature Mismatch)...")
    
    if not os.path.exists(SCOUT_DATA):
        print(f"❌ Error: {SCOUT_DATA} not found.")
        return

    df = pd.read_csv(SCOUT_DATA)
    
    # 1. Select only valid DNA columns that exist in the Data
    valid_features = [c for c in DNA_FEATURES if c in df.columns]
    
    if len(valid_features) < 10:
        print("❌ Error: Not enough DNA columns found in CSV.")
        return

    print(f"   Training on {len(valid_features)} DNA features...")
    
    # 2. Prepare Data
    X = df[valid_features].fillna(0)
    
    # 3. Scale
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 4. Train KNN
    knn = NearestNeighbors(n_neighbors=6, metric='cosine', algorithm='brute')
    knn.fit(X_scaled)
    
    # 5. Save Fresh Files
    with open(KNN_MODEL, 'wb') as f: pickle.dump(knn, f)
    with open(KNN_SCALER, 'wb') as f: pickle.dump(scaler, f)
    
    # Save the feature list
    with open("dna_features.json", "w") as f: json.dump(valid_features, f)
    
    print("✅ New KNN Model & Scaler Saved Successfully!")

def load_knn_resources():
    global _knn, _scaler, _df_scout
    
    # If files don't exist, train them first
    if not os.path.exists(KNN_MODEL) or not os.path.exists(KNN_SCALER):
        train_new_knn_model()

    if _knn is None:
        with open(KNN_MODEL, 'rb') as f: _knn = pickle.load(f)
        with open(KNN_SCALER, 'rb') as f: _scaler = pickle.load(f)
        _df_scout = pd.read_csv(SCOUT_DATA)
        _df_scout['search_key'] = _df_scout['Player'].apply(lambda x: str(x).lower().strip())

def get_similar_players(player_name, n=4):
    """Uses the KNN model to find players with similar stats."""
    load_knn_resources()
    
    # 1. Load the feature list
    if os.path.exists("dna_features.json"):
        with open("dna_features.json", "r") as f:
            model_features = json.load(f)
    else:
        # Fallback if json missing 
        model_features = [c for c in DNA_FEATURES if c in _df_scout.columns]

    # 2. Find Target Player
    s_key = clean_txt(player_name)
    target = _df_scout[_df_scout['search_key'] == s_key]
    
    if target.empty:
        return []

    # 3. Create Vector 
    target_vec = target.iloc[0][model_features].fillna(0).values.reshape(1, -1)
    
    # 4. Scale & Query
    target_vec_scaled = _scaler.transform(target_vec)
    distances, indices = _knn.kneighbors(target_vec_scaled, n_neighbors=n+1)
    
    results = []
    for i in range(1, len(indices[0])):
        idx = indices[0][i]
        sim_player = _df_scout.iloc[idx]
        
        # Convert cosine distance to similarity %
        sim_score = (1 - distances[0][i]) * 100
        
        results.append({
            "name": sim_player['Player'],
            "team": sim_player['Team'],
            "similarity": f"{sim_score:.1f}%"
        })
        
    return results

def generate_scout_report(player_name, target_team):
    print(f"\n📋 GENERATING FULL SCOUT REPORT: {player_name} -> {target_team}")
    print("="*60)
    
    # 1. IMPACT PREDICTION
    try:
        impact = predict_transfer(player_name, target_team)
        if "error" in impact:
            print(f"❌ {impact['error']}")
        else:
            print(f"\n🔮 IMPACT PROJECTION")
            print(f"   Model Verdict:    {impact['verdict']}")
            print(f"   Elo Change:       {impact['predicted_impact']:+.2f} points")
            print(f"   Forecasted Elo:   {impact['forecasted_elo']}")
    except Exception as e:
        print(f"❌ Impact Model Error: {e}")

    # 2. CHEMISTRY CHECK
    print(f"\n⚗️  CHEMISTRY CHECK")
    try:
        squad_df = get_target_squad(target_team)
        if squad_df.empty:
            print("   ⚠️  Target Squad not found.")
        else:
            load_knn_resources()
            p_row_list = _df_scout[_df_scout['search_key'] == clean_txt(player_name)]
            
            if not p_row_list.empty:
                p_row = p_row_list.iloc[0]
                chem_df = get_squad_chemistry_map(p_row, squad_df)
                
                if not chem_df.empty:
                    avg_chem = chem_df['Chemistry_Score'].mean()
                    best_link = chem_df.iloc[0]
                    print(f"   Avg Squad Fit:    {avg_chem:.1f}/100")
                    print(f"   Best Partner:     {best_link['Teammate']} (Score: {best_link['Chemistry_Score']:.1f})")
                else:
                    print("   ⚠️  No chemistry data available.")
            else:
                print("   ⚠️  Player stats not found for chemistry.")
    except Exception as e:
        print(f"⚠️  Chemistry Error: {e}")

    # 3. SIMILARITY ENGINE (KNN)
    print(f"\n🧬 PLAYER DNA (Similar Profiles)")
    try:
        similar_players = get_similar_players(player_name)
        if similar_players:
            for p in similar_players:
                print(f"   • {p['name']} ({p['team']}) - {p['similarity']} Match")
        else:
            print("   No similar players found.")
    except Exception as e:
        print(f"❌ KNN Error: {e}")

if __name__ == "__main__":
    if not os.path.exists("dna_features.json"):
        train_new_knn_model()
        
    # Test
    generate_scout_report("Cole Palmer", "Manchester United")
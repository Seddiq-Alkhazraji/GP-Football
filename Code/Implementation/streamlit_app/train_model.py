import pandas as pd
import xgboost as xgb
import numpy as np
import os
import pickle
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

# --- VERSION CHECK ---
print("\n✅ RUNNING FINAL SCRIPT V5 (Added Player DNA Engine)\n")

# --- CONFIGURATION ---
TRANSFERS_FILE = "transfers.csv"       
MARKET_FILE = "market_values.csv"      
STATS_FILE = "scout_data_weighted.csv" 
WAGES_FILE = "wages.csv"               
PLAYSTYLE_FILE = "team_playstyles.csv" 

# OUTPUT MODELS
REGRESSION_MODEL = "impact_model_hybrid.json"
CLASSIFIER_MODEL = "success_probability_model.pkl"
DNA_MODEL = "player_dna_model.pkl"         # <--- NEW
DNA_ENCODER = "dna_label_encoder.pkl"      # <--- NEW
STYLE_COLUMNS_FILE = "style_columns.pkl" 

# FEATURES
BASE_FEATURES = [
    'Gls_Standard_Per90', 'Ast_Standard_Per90', 'npxG_Per', 'xAG_Per',
    'PrgP_Per90', 'PrgC_Carries_Per90', 'TklW_Tackles_Per90', 
    'Int_Def_Per90', 'Won_percent_Aerial', 'SCA90_SCA'
]

# --- HELPER FUNCTIONS ---
def clean_team_name(name):
    if not isinstance(name, str): return ""
    name = name.lower()
    replacements = [" fc", "cf ", " ac", "as ", "borussia ", "bayer ", "sporting ", "jk", "fk"]
    for r in replacements: name = name.replace(r, "")
    return name.strip()

def clean_money(val):
    if pd.isna(val): return 0
    if isinstance(val, (int, float)): return val
    clean = re.sub(r'[^\d.]', '', str(val))
    try: return float(clean)
    except: return 0

def simplify_position(pos):
    """Maps specific positions to general DNA classes."""
    if not isinstance(pos, str): return "Unknown"
    pos = pos.lower()
    if "goalkeeper" in pos: return "GK"
    if "back" in pos or "defender" in pos: return "DEF"
    if "midfield" in pos: return "MID"
    if "winger" in pos or "forward" in pos or "striker" in pos: return "ATT"
    return "Unknown"

# --- TRAINING ENGINES ---

def train_dna_engine(df_stats, df_market):
    """
    Trains a classifier to predict a player's 'True Position' based on stats.
    Useful for finding players playing out of position (e.g. inverted fullbacks).
    """
    print("🧬 Training Player DNA Model (Position Classifier)...")
    
    # 1. Merge Stats with Position Labels
    # We use the Market Value file because it has the 'player_position' column
    df_stats['join_key'] = df_stats['Player'].str.lower().str.strip()
    df_market['join_key'] = df_market['player_name'].str.lower().str.strip()
    
    # Inner join - we need both stats and a label
    merged = pd.merge(df_stats, df_market[['join_key', 'player_position']], on='join_key', how='inner')
    
    if merged.empty:
        print("   ⚠️ No matching players found for DNA training. Skipping.")
        return

    # 2. Process Labels
    merged['simple_pos'] = merged['player_position'].apply(simplify_position)
    valid_data = merged[merged['simple_pos'] != "Unknown"].copy()
    
    X = valid_data[BASE_FEATURES].fillna(0).values
    y = valid_data['simple_pos'].values
    
    # 3. Train
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    dna_model = LogisticRegression(multi_class='multinomial', max_iter=2000)
    dna_model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, dna_model.predict(X_test))
    print(f"   🎯 DNA Accuracy: {acc*100:.1f}% (Can identify {le.classes_})")
    
    # 4. Save
    with open(DNA_MODEL, 'wb') as f: pickle.dump(dna_model, f)
    with open(DNA_ENCODER, 'wb') as f: pickle.dump(le, f)

def process_dataset(transfers, elo_start_file, elo_end_file, df_stats, df_wages, df_playstyle, season_label):
    print(f"   -> Processing {season_label} data...")
    
    # Merge Wages
    if not df_wages.empty:
        transfers = pd.merge(transfers, df_wages[['PLAYER', 'GROSS P/W']], 
                             left_on='player_name', right_on='PLAYER', how='left')
        transfers['GROSS P/W'] = transfers['GROSS P/W'].apply(clean_money)
    else:
        transfers['GROSS P/W'] = 0

    # Merge Playstyle
    if not df_playstyle.empty:
        transfers['join_team'] = transfers.apply(
            lambda x: x.get('team_name') if 'team_name' in x else x.get('current_club'), axis=1
        ).apply(clean_team_name)
        
        df_playstyle['join_team'] = df_playstyle['Team'].apply(clean_team_name)
        style_map = df_playstyle[['join_team', 'Main_Style']].drop_duplicates()
        transfers = pd.merge(transfers, style_map, on='join_team', how='left')
        transfers['Main_Style'] = transfers['Main_Style'].fillna('Unknown')
    else:
        transfers['Main_Style'] = 'Unknown'

    # Load Elo
    if not os.path.exists(elo_start_file) or not os.path.exists(elo_end_file):
        return [], [], [], []

    df_elo_old = pd.read_csv(elo_start_file)
    df_elo_new = pd.read_csv(elo_end_file)
    
    # Elo Merge Logic (Simplified for brevity, assumes standard columns)
    col_old = next((c for c in df_elo_old.columns if 'elo' in c.lower()), None)
    col_new = next((c for c in df_elo_new.columns if 'elo' in c.lower()), None)
    
    df_elo_old['join_key'] = df_elo_old.iloc[:,0].apply(clean_team_name) # Assuming Team is 1st col
    df_elo_new['join_key'] = df_elo_new.iloc[:,0].apply(clean_team_name)
    
    elo_map = pd.merge(df_elo_old, df_elo_new, on='join_key', suffixes=('_old', '_new'))
    # Dynamic column grab
    old_elo_col = f"{col_old}_old"
    new_elo_col = f"{col_new}_new"
    
    # Build Dataset
    X_list, y_reg, y_class = [], [], []
    
    df_stats['join_name'] = df_stats['Player'].astype(str).str.lower().str.strip()
    style_dummies = pd.get_dummies(transfers['Main_Style'], prefix='Style')
    transfers = pd.concat([transfers, style_dummies], axis=1)
    style_cols = [c for c in transfers.columns if c.startswith('Style_')]
    
    for idx, row in transfers.iterrows():
        # Match Player & Team
        team_key = clean_team_name(row.get('team_name') or row.get('current_club'))
        p_name = str(row.get('player_name')).lower().strip()
        
        team_row = elo_map[elo_map['join_key'] == team_key]
        p_stats = df_stats[df_stats['join_name'] == p_name]
        
        if team_row.empty or p_stats.empty: continue
        
        try:
            # Features
            stats_vec = p_stats.iloc[0][BASE_FEATURES].fillna(0).values
            start_elo = team_row.iloc[0][old_elo_col]
            delta_elo = team_row.iloc[0][new_elo_col] - start_elo
            
            log_mv = np.log1p(clean_money(row.get('player_market_value_euro', 0)))
            log_wage = np.log1p(clean_money(row.get('GROSS P/W', 0)))
            style_vec = row[style_cols].fillna(0).values
            
            input_vec = np.concatenate([[start_elo], stats_vec, [log_mv, log_wage], style_vec])
            
            X_list.append(input_vec)
            y_reg.append(delta_elo)
            y_class.append(1 if delta_elo > 0 else 0)
        except: continue
            
    return X_list, y_reg, y_class, style_cols

def train_hybrid_engine():
    print("🚀 Starting Multi-Stage Training...")
    base_dir = os.getcwd()
    
    # Load Data
    stats_path = os.path.join(base_dir, STATS_FILE)
    if not os.path.exists(stats_path): print(f"❌ Missing {STATS_FILE}"); return
    df_stats = pd.read_csv(stats_path)
    
    wages_path = os.path.join(base_dir, WAGES_FILE)
    df_wages = pd.read_csv(wages_path) if os.path.exists(wages_path) else pd.DataFrame()
    
    playstyle_path = os.path.join(base_dir, PLAYSTYLE_FILE)
    df_playstyle = pd.read_csv(playstyle_path) if os.path.exists(playstyle_path) else pd.DataFrame()

    # --- 1. TRAIN DNA MODEL ---
    market_path = os.path.join(base_dir, MARKET_FILE)
    if os.path.exists(market_path):
        df_market = pd.read_csv(market_path)
        train_dna_engine(df_stats, df_market)
    
        # --- 2. TRAIN IMPACT MODELS ---
        subset_2024 = df_market[df_market['season_start_year'] == 2024]
        X, y_reg, y_class, style_cols = process_dataset(
            subset_2024, "elo_start_24.csv", "elo_end_24.csv",
            df_stats, df_wages, df_playstyle, "Season 2024/25"
        )
        
        if style_cols:
            with open(STYLE_COLUMNS_FILE, 'wb') as f: pickle.dump(style_cols, f)

        if X:
            X = np.array(X)
            X_train, X_test, y_r_train, y_r_test, y_c_train, y_c_test = train_test_split(
                X, np.array(y_reg), np.array(y_class), test_size=0.2, random_state=42
            )
            
            print("📈 Training XGBoost (Impact)...")
            reg = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=300, max_depth=6)
            reg.fit(X_train, y_r_train)
            reg.save_model(REGRESSION_MODEL)
            
            print("⚖️  Training Logistic (Success)...")
            clf = LogisticRegression(max_iter=2000)
            clf.fit(X_train, y_c_train)
            with open(CLASSIFIER_MODEL, 'wb') as f: pickle.dump(clf, f)
            
            print("\n✅ ALL SYSTEMS GO. Models Saved.")
        else:
            print("❌ No Transfer Data Matched.")
    else:
        print("❌ Market file missing.")

if __name__ == "__main__":
    train_hybrid_engine()
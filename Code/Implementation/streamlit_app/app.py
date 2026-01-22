import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
import pickle
import joblib 
import chemistry
import sys
import xgboost as xgb
import difflib 
import re 
import json
from sqlalchemy import create_engine, text
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="AI Football Intelligence", layout="wide", initial_sidebar_state="expanded")

# --- DATABASE CONNECTION SETTINGS ---
DB_USER = "postgres"
DB_PASS = "mshro3post"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "Football_project"
DB_URL = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# --- MODEL FILES ---
MODEL_FILE = "impact_model_hybrid.json"
LOGISTIC_FILE = "success_probability_model.pkl"
DNA_MODEL = "player_dna_model.pkl"
DNA_ENCODER = "dna_label_encoder.pkl"
STYLE_COLS_FILE = "style_columns.pkl"
CHEM_RF_MODEL = "chemistry_rf_model.pkl"
CHEM_ENCODER = "role_encoder.pkl"

SCOUT_ELO_MODEL = "scout_elo_model_best.json"
MODEL_FEATURES = "model_features.json"
KNN_MODEL = "knn_similarity.pkl"
KNN_SCALER = "knn_scaler.pkl"

# --- 2. DATA LOADING & CACHING ---
@st.cache_resource
def load_models():
    models = {}
    if os.path.exists(MODEL_FILE):
        reg = xgb.XGBRegressor()
        reg.load_model(MODEL_FILE)
        models['reg'] = reg
    if os.path.exists(LOGISTIC_FILE):
        with open(LOGISTIC_FILE, 'rb') as f: models['class'] = pickle.load(f)
    if os.path.exists(DNA_MODEL):
        with open(DNA_MODEL, 'rb') as f: models['dna'] = pickle.load(f)
    if os.path.exists(DNA_ENCODER):
        with open(DNA_ENCODER, 'rb') as f: models['dna_enc'] = pickle.load(f)
    if os.path.exists(STYLE_COLS_FILE):
        with open(STYLE_COLS_FILE, 'rb') as f: models['styles'] = pickle.load(f)

    if os.path.exists(KNN_MODEL):
        with open(KNN_MODEL, 'rb') as f: models['knn'] = pickle.load(f)
    if os.path.exists(KNN_SCALER):
        with open(KNN_SCALER, 'rb') as f: models['knn_scaler'] = pickle.load(f)

    if os.path.exists("dna_features.json"):
        with open("dna_features.json", "r") as f: 
            models['knn_features'] = json.load(f)
    else:
        models['knn_features'] = [
            'Gls_Standard_Per90', 'Ast_Standard_Per90', 'npxG_Per', 'xAG_Per', 
            'Sh_per_90_Standard', 'SoT_percent_Standard', 'G_per_Sh_Standard',
            'PrgP_Per90', 'PrgC_Carries_Per90', 'Att_Take_Per90', 'Succ_Take_Per90',
            'TklW_Tackles_Per90', 'Int_Def_Per90', 'Blocks_Blocks_Per90', 'Clr_Per90',
            'Won_percent_Aerial', 'SCA90_SCA', 'GCA90_GCA', 'Cmp_percent_Total',
            'Cmp_Short_Per90', 'Cmp_Medium_Per90', 'Cmp_Long_Per90', 
            'Att Pen_Touches_Per90', 'Recov_Per90'
        ]
               
    if os.path.exists(SCOUT_ELO_MODEL):
        elo_model = xgb.XGBRegressor()
        elo_model.load_model(SCOUT_ELO_MODEL)
        models['elo_model'] = elo_model
    if os.path.exists(MODEL_FEATURES):
        with open(MODEL_FEATURES, "r") as f: models['model_cols'] = json.load(f)
        
    return models

def aggressive_normalize(name):
    if pd.isna(name): return ""
    name = str(name).lower()
    name = name.replace(' fc', '').replace(' afc', '').replace(' &', '').replace(' and', '')
    name = re.sub(r'[^a-z0-9]', '', name)
    return name

def clean_txt(text):
    if not isinstance(text, str): return ""
    return text.lower().strip()

@st.cache_data
def load_data():
    data = {}
    try:
        engine = create_engine(DB_URL)
        conn = engine.connect()
        
        def standardize_cols(df):
            rename_map = {
                'team': 'Team', 'player': 'Player', 'comp': 'Comp', 
                'age': 'Age', 'starts_playing': 'Starts_Playing',
                'min_percent_playing.time': 'Min_percent_Playing.Time',
                'min_playing': 'Min_Playing',
                'pos': 'main_pos', 'position': 'main_pos'
            }
            cols_to_rename = {k: v for k, v in rename_map.items() if k in df.columns and v not in df.columns}
            return df.rename(columns=cols_to_rename)

        # 1. Load Main Stats
        df_stats = pd.read_sql("SELECT * FROM scout_outfield", conn)
        df_stats = standardize_cols(df_stats)
        if not df_stats.empty:
            if 'Comp' in df_stats.columns:
                df_stats['Comp'] = df_stats['Comp'].astype(str).str.replace(r'^[A-Z]{3}-', '', regex=True).str.strip()
            if 'Team' in df_stats.columns:
                df_stats['Team'] = df_stats['Team'].astype(str).str.strip().str.title()
                df_stats['Team_Norm'] = df_stats['Team'].apply(aggressive_normalize)
            df_stats['search_key'] = df_stats['Player'].apply(clean_txt)
        
        data['stats'] = df_stats

        # 2. Load DISPLAY Stats
        try:
            df_player_disp = pd.read_sql("SELECT * FROM player_display_stats", conn)
            df_player_disp = standardize_cols(df_player_disp)
            if 'Team' in df_player_disp.columns:
                df_player_disp['Team'] = df_player_disp['Team'].astype(str).str.strip().str.title()
                df_player_disp['Team_Norm'] = df_player_disp['Team'].apply(aggressive_normalize)
            data['player_display'] = df_player_disp
            
            df_team_disp = pd.read_sql("SELECT * FROM team_display_stats", conn)
            data['team_display'] = df_team_disp
        except Exception as e:
            st.warning(f"Display tables missing, falling back to raw stats: {e}")
            data['player_display'] = df_stats.copy()
            data['team_display'] = pd.DataFrame()

        # 3. Load Wages
        try:
            wages = pd.read_sql("SELECT * FROM wages", conn)
            if not wages.empty:
                w_cols = {c.lower(): c for c in wages.columns}
                gross_col = w_cols.get('gross p/w', 'GROSS P/W')
                player_col = w_cols.get('player', 'PLAYER')

                wages[gross_col] = wages[gross_col].astype(str).str.replace('€', '').str.replace(',', '')
                wages[gross_col] = pd.to_numeric(wages[gross_col], errors='coerce').fillna(0)
                
                # Merge into stats
                if not df_stats.empty:
                     df_stats = pd.merge(df_stats, wages[[player_col, gross_col]], 
                                         left_on='Player', right_on=player_col, how='left')
                     if gross_col != 'GROSS P/W':
                         df_stats['GROSS P/W'] = df_stats[gross_col]
        except Exception as e:
            st.warning(f"Could not load wages from DB: {e}")

        # 4. Load Matches
        matches = pd.read_sql("SELECT * FROM matches", conn)
        if not matches.empty:
            matches = matches.rename(columns={'team': 'team', 'opponent': 'opponent'})
            if 'team' in matches.columns:
                matches['team'] = matches['team'].astype(str).str.strip().str.title()
                matches['team_norm'] = matches['team'].apply(aggressive_normalize)
        data['matches'] = matches
        
        # 5. Load Elo
        data['elo'] = pd.read_sql("SELECT * FROM elo_history", conn)
        
        # 6. Load Chemistry
        data['chem'] = pd.read_sql("SELECT * FROM chemistry_clean", conn)
        
        # 7. Load Playstyles
        data['playstyles'] = pd.read_sql("SELECT * FROM team_playstyles", conn)
        
        # 8. Load Global Player Ratings
        df_ratings = pd.read_sql("SELECT * FROM ratings", conn)
        if not df_ratings.empty:
            if 'team_name' in df_ratings.columns:
                df_ratings['team_norm'] = df_ratings['team_name'].apply(aggressive_normalize)
            if 'player_name' in df_ratings.columns:
                df_ratings['player_norm'] = df_ratings['player_name'].apply(aggressive_normalize)
            
            if not df_stats.empty:
                meta_cols = ['Player', 'Age', 'Comp', 'main_pos']
                avail_meta = [c for c in meta_cols if c in df_stats.columns]
                stats_mini = df_stats[avail_meta].copy()
                stats_mini['player_norm'] = stats_mini['Player'].apply(aggressive_normalize)
                df_ratings = pd.merge(df_ratings, stats_mini[['player_norm'] + [c for c in avail_meta if c != 'Player']], on='player_norm', how='left')
                df_ratings['Age'] = df_ratings['Age'].fillna(25)
        
        data['ratings'] = df_ratings
        conn.close()

    except Exception as e:
        st.error(f"❌ CRITICAL DATABASE ERROR: {e}")
        return {}
        
    return data

models = load_models()
data = load_data()
df_stats = data.get('stats', pd.DataFrame()) 
df_ratings = data.get('ratings', pd.DataFrame())
df_player_disp = data.get('player_display', pd.DataFrame())
df_team_disp = data.get('team_display', pd.DataFrame())

# --- 3. HELPER FUNCTIONS & ML LOGIC ---

def merge_position_class(p: str) -> str:
    """Merges detailed positions into 6 core Tactical Classes."""
    if pd.isna(p): return np.nan
    p = str(p).lower().strip().replace(" ", "")

    if p == "gk": return "GK"
    if p in {"cb", "lb", "rb", "lwb", "rwb", "wb", "fb", "df"}: return "DF"
    if p.startswith("df"): return "DF"
    if p == "dm": return "DM"
    if p in {"cm", "mf"}: return "CM"
    if p == "am": return "AM"
    if p in {"lw", "rw", "lm", "rm", "fw", "cf", "ss"}: return "FW"
    if p.startswith("fw"): return "FW"

    return "Unknown"

@st.cache_resource(show_spinner="Training Position DNA Model...")
def train_position_dna_model(data):
    """
    Trains a lightweight Logistic Regression model to predict Position probabilities.
    Returns: (model_pipeline, class_labels)
    """
    df_mod = data.copy()
    
    # 1. Target Engineering
    if 'main_pos' in df_mod.columns:
        target_col = 'main_pos'
    elif 'Pos' in df_mod.columns:
        target_col = 'Pos'
    else:
        return None, None
        
    df_mod['target_class'] = df_mod[target_col].apply(merge_position_class)
    df_mod = df_mod[df_mod['target_class'] != "Unknown"]
    
    # 2. Feature Selection (Numeric only, drop ID/Meta cols)
    drop_cols = ['Player', 'Team', 'Nation', 'Pos', 'main_pos', 'target_class', 
                 'Born', 'Age', 'Matches', 'Starts', 'Mins', 'Url', '90s', 
                 'Comp', 'Season', 'League', 'Squad', 'player_market_value_euro']
    
    # Select numeric features that exist in df
    features = df_mod.select_dtypes(include=[np.number]).columns.tolist()
    features = [f for f in features if f not in drop_cols]
    
    if not features:
        return None, None

    X = df_mod[features]
    y = df_mod['target_class']
    
    # 3. Pipeline (Impute -> Scale -> LogReg)
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=500))
    ])
    
    pipeline.fit(X, y)
    return pipeline, pipeline.classes_

def get_elo_for_team_integrated(team_name, df_elo):
    if df_elo is None or df_elo.empty: return 1500
    search_key = clean_txt(team_name)
    col_team = next((c for c in df_elo.columns if 'team' in c.lower() or 'club' in c.lower()), None)
    col_elo = next((c for c in df_elo.columns if 'elo' in c.lower() and 'team' not in c.lower()), None)
    if not col_team or not col_elo: return 1500
    matches = df_elo[df_elo[col_team].astype(str).apply(lambda x: search_key in clean_txt(str(x)))]
    if not matches.empty: return matches.iloc[0][col_elo]
    return 1500

def predict_transfer_integrated(player_name, target_team_name):
    if 'elo_model' not in models or 'model_cols' not in models:
        return {"error": "New Elo Model (scout_elo_model_best.json) not loaded."}
    search_key = clean_txt(player_name)
    player_row = df_stats[df_stats['search_key'] == search_key]
    if player_row.empty: player_row = df_stats[df_stats['search_key'].str.contains(search_key)]
    if player_row.empty: return {"error": f"Player '{player_name}' not found in DB."}
    
    player_data = player_row.iloc[0].to_dict()
    current_elo = get_elo_for_team_integrated(target_team_name, data.get('elo', pd.DataFrame()))
    player_data['Team_Start_Elo'] = current_elo
    input_df = pd.DataFrame([player_data]).reindex(columns=models['model_cols'], fill_value=0)
    
    pred_change = models['elo_model'].predict(input_df)[0]
    final_elo = current_elo + pred_change
    
    verdict = "😐 NEUTRAL"
    if pred_change > 20: verdict = "🚀 GAME CHANGER"
    elif pred_change > 5: verdict = "✅ POSITIVE"
    elif pred_change < -5: verdict = "📉 NEGATIVE"
    elif pred_change < -20: verdict = "❌ FLOP RISK"
    
    return {
        "current_elo": round(current_elo), "predicted_impact": round(pred_change, 2),
        "forecasted_elo": round(final_elo), "verdict": verdict
    }

def calculate_tvi(chem_score, impact_score, market_value, weights):
    norm_chem = max(0, min(100, chem_score))
    norm_impact = max(0, min(100, 50 + (impact_score * 5)))
    norm_roi = max(0, 100 - (market_value / 1_000_000))
    tvi = (norm_chem * weights[0]) + (norm_impact * weights[1]) + (norm_roi * weights[2])
    return round(tvi, 1)

def get_strategy_weights(choice):
    if choice == '🏆 WIN NOW (Galactico)': return (0.10, 0.80, 0.10)
    if choice == '💰 MONEYBALL (Brighton)': return (0.30, 0.10, 0.60)
    if choice == '🧩 SYSTEM FIT (Pep)': return (0.70, 0.20, 0.10)
    return (0.33, 0.33, 0.33)

def get_best_formation(team_name):
    if 'matches' not in data or data['matches'].empty: return "4-3-3"
    matches = data['matches']
    target_norm = aggressive_normalize(team_name)
    if 'team_norm' not in matches.columns: return "4-3-3"
    
    if target_norm in matches['team_norm'].values: found_norm = target_norm
    else:
        closest = difflib.get_close_matches(target_norm, matches['team_norm'].unique().tolist(), n=1, cutoff=0.6)
        if closest: found_norm = closest[0]
        else: return "4-3-3"
        
    team_matches = matches[matches['team_norm'] == found_norm].copy()
    cols = {c.lower(): c for c in team_matches.columns}
    venue_col = cols.get('venue')
    home_form_col = cols.get('formation')
    
    if not venue_col or not home_form_col: return "4-3-3"
    
    formations_list = []
    for _, row in team_matches.iterrows():
        venue_val = str(row[venue_col]).lower().strip()
        if venue_val in ['home', 'h'] and pd.notna(row[home_form_col]):
            formations_list.append(str(row[home_form_col]))
        elif venue_val in ['away', 'a'] and pd.notna(row[home_form_col]):
             formations_list.append(str(row[home_form_col]))
            
    if not formations_list: return "4-3-3"
    try: return str(pd.Series(formations_list).value_counts().idxmax()).replace("◆", "").strip()
    except: return "4-3-3"

def get_formation_config(formation_name):
    gk = [{'regex': r'GK|Goal', 'count': 1, 'coords': [(50, 5)]}]
    lb = [{'regex': r'LB|LWB|Left Back', 'count': 1, 'coords': [(15, 25)]}]
    rb = [{'regex': r'RB|RWB|Right Back', 'count': 1, 'coords': [(85, 25)]}]
    cb_2 = [{'regex': r'CB|Central Defender', 'count': 2, 'coords': [(38, 25), (62, 25)]}]
    cdm_1 = [{'regex': r'CDM|Defensive Mid|CM', 'count': 1, 'coords': [(50, 42)]}]
    cm_2 = [{'regex': r'CM|Central Mid|CAM|CDM', 'count': 2, 'coords': [(35, 55), (65, 55)]}]
    lw = [{'regex': r'LW|Left Wing|LM|LAM', 'count': 1, 'coords': [(15, 80)]}]
    rw = [{'regex': r'RW|Right Wing|RM|RAM', 'count': 1, 'coords': [(85, 80)]}]
    st_1 = [{'regex': r'ST|Striker|CF|Centre Forward', 'count': 1, 'coords': [(50, 90)]}]
    return gk + lb + cb_2 + rb + cdm_1 + cm_2 + lw + rw + st_1

def predict_3_stage(candidate_row, team_meta):
    if 'reg' not in models or 'class' not in models: return 0.0, 0.0, 0.0
    feat_cols = ['Gls_Standard_Per90', 'Ast_Standard_Per90', 'npxG_Per', 'xAG_Per', 'PrgP_Per90', 'PrgC_Carries_Per90', 'TklW_Tackles_Per90', 'Int_Def_Per90', 'Won_percent_Aerial', 'SCA90_SCA']
    current_feats = []
    for f in feat_cols:
        try: current_feats.append(float(candidate_row.get(f, 0)))
        except: current_feats.append(0.0)
    stats_vec = np.array(current_feats).astype(float)
    current_elo = float(team_meta.get('elo', 1500))
    
    def clean_money(val):
        if pd.isna(val): return 0.0
        s = str(val).lower().replace('€', '').replace(',', '').replace('$', '').strip()
        if 'm' in s: return float(s.replace('m', '')) * 1_000_000
        if 'k' in s: return float(s.replace('k', '')) * 1_000
        try: return float(s)
        except: return 0.0
        
    wage = clean_money(candidate_row.get('GROSS P/W', 0))
    if wage < 100: wage = 20000.0 
    mv = clean_money(candidate_row.get('player_market_value_euro', 0))
    if mv < 100: mv = 5_000_000.0
    
    current_style = team_meta.get('style', 'Unknown')
    style_vec = [1 if f"Style_{current_style}" == c else 0 for c in models.get('styles', [])] if 'styles' in models else [0]*5
    input_vec = np.concatenate([[current_elo], stats_vec, [np.log1p(mv), np.log1p(wage)], style_vec])
    input_vec = np.nan_to_num(input_vec, nan=0.0, posinf=0.0, neginf=0.0)
    
    try:
        prob_success = models['class'].predict_proba([input_vec])[0][1] * 100
        elo_impact = models['reg'].predict([input_vec])[0]
        cost_est = max(100000, mv + (wage * 52 * 3))
        roi = (elo_impact * (prob_success/100) * 10_000_000) / cost_est 
        return round(prob_success, 1), round(elo_impact, 2), round(roi, 1)
    except: return 0.0, 0.0, 0.0

def get_team_meta(team_name):
    meta = {'elo': 1500, 'style': 'Unknown'}
    if 'elo' in data and not data['elo'].empty:
        elo_df = data['elo']
        col_team = next((c for c in elo_df.columns if 'team' in c.lower()), None)
        col_elo = next((c for c in elo_df.columns if 'elo' in c.lower() and 'team' not in c.lower()), None)
        if col_team and col_elo:
            match = elo_df[elo_df[col_team].astype(str).str.contains(team_name, case=False, na=False)]
            if not match.empty: meta['elo'] = match.iloc[0][col_elo]
    if 'playstyles' in data and not data['playstyles'].empty:
        df_style = data['playstyles']
        matches = difflib.get_close_matches(team_name, df_style['Team'].unique(), n=1, cutoff=0.5)
        if matches:
            match = df_style[df_style['Team'] == matches[0]]
            if not match.empty: meta['style'] = match.iloc[0]['Main_Style']
    return meta

# --- 4. VISUALIZATION MODULES ---
def draw_pitch_lineup(team_df, metric, formation_name="4-3-3"):
    slots_config = get_formation_config(formation_name)
    selected_players_indices = []
    lineup_rows = []
    
    for slot in slots_config:
        regex = slot['regex']
        count = slot['count']
        coords = slot['coords']
        candidates = team_df[(team_df['main_pos'].str.contains(regex, case=False, na=False)) & (~team_df.index.isin(selected_players_indices))].copy()
        if metric not in candidates.columns: candidates[metric] = 0
        candidates = candidates.sort_values(metric, ascending=False).head(count)
        
        if len(candidates) < count:
            needed = count - len(candidates)
            fallback = "Def" if "Back" in regex else "Mid" if "Mid" in regex else "For"
            extras = team_df[(team_df['main_pos'].str.contains(fallback, case=False, na=False)) & (~team_df.index.isin(selected_players_indices)) & (~team_df.index.isin(candidates.index))].sort_values(metric, ascending=False).head(needed)
            candidates = pd.concat([candidates, extras])
            
        selected_players_indices.extend(candidates.index.tolist())
        candidates = candidates.reset_index(drop=True)
        actual_count = min(len(candidates), len(coords))
        candidates = candidates.iloc[:actual_count].copy()
        candidates['x'] = [c[0] for c in coords[:actual_count]]
        candidates['y'] = [c[1] for c in coords[:actual_count]]
        lineup_rows.append(candidates)
        
    lineup = pd.concat(lineup_rows, ignore_index=True) if lineup_rows else pd.DataFrame()
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, y0=0, x1=100, y1=100, line=dict(color="white"), fillcolor="#40a860", layer="below")
    if not lineup.empty:
        fig.add_trace(go.Scatter(x=lineup['x'], y=lineup['y'], mode='markers+text', text=lineup['Player'], textposition="bottom center", marker=dict(size=18, color='white', line=dict(width=2, color='black')), hovertext=lineup.apply(lambda row: f"{row['Player']}<br>{metric}: {row[metric]}", axis=1), hoverinfo="text", customdata=lineup['Player']))
    fig.update_layout(xaxis=dict(visible=False, range=[0, 100]), yaxis=dict(visible=False, range=[0, 100]), height=600, margin=dict(l=0, r=0, t=0, b=0), plot_bgcolor="#40a860", dragmode='select', title=f"Starting XI ({formation_name})")
    event = st.plotly_chart(fig, use_container_width=True, on_select="rerun")
    return event

def show_global_stats_module(df_player, df_team):
    st.subheader("📊 Deep Dive Statistics")
    view_mode = st.radio("View Mode:", ["Player Stats", "Team Stats"], horizontal=True)
    df = df_player if view_mode == "Player Stats" else df_team
    if df.empty: return st.error("No Data available.")
    
    stat_mode = st.radio("Analysis Type:", ["Single Metric Ranking", "Multi-Metric Comparison"], horizontal=True)
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    if stat_mode == "Single Metric Ranking":
        c1, c2 = st.columns(2)
        target_stat = c1.selectbox("Select Metric", numeric_cols, index=0)
        top_n = c2.slider("Show Top N", 5, 50, 15)
        label_col = 'Player' if view_mode == "Player Stats" else 'Team'
        chart_data = df.sort_values(target_stat, ascending=False).head(top_n)
        fig = px.bar(chart_data, x=target_stat, y=label_col, orientation='h', color=target_stat, color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)
    else: 
        c1, c2 = st.columns(2)
        x_stat = c1.selectbox("X-Axis", numeric_cols, index=0)
        y_stat = c2.selectbox("Y-Axis", numeric_cols, index=1)
        label_col = 'Player' if view_mode == "Player Stats" else 'Team'
        fig = px.scatter(df, x=x_stat, y=y_stat, hover_name=label_col, color=label_col, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

# --- 5. INTERFACES ---

def manager_interface():
    st.sidebar.header("🏢 Manager Controls")
    if df_player_disp.empty: return st.error("Player display stats database is empty.")
    leagues = sorted(df_player_disp['Comp'].dropna().unique().tolist()) if 'Comp' in df_player_disp.columns else []
    selected_league = st.sidebar.selectbox("Select Competition", leagues)
    league_teams = sorted(df_player_disp[df_player_disp['Comp'] == selected_league]['Team'].unique().tolist()) if selected_league else []
    selected_team = st.sidebar.selectbox("Select Team", league_teams)
    
    team_df = df_player_disp[df_player_disp['Team'] == selected_team].copy()
    meta = get_team_meta(selected_team)
    formation = get_best_formation(selected_team)
    
    st.title(f"{selected_team} Dashboard")
    st.info(f"📋 **Style:** {meta['style']} | 📈 **Elo:** {meta.get('elo', 'N/A')} | 📐 **Form:** {formation}")
    
    tab1, tab2, tab3 = st.tabs(["🏟️ Lineup", "📉 Metrics", "📊 Advanced"])
    with tab1:
        st.subheader("Starting XI Visualizer")
        c1, c2 = st.columns([3, 1])
        with c2:
            metric_options = {"Minutes Played": "Min_Playing", "Starts": "Starts_Playing"}
            valid_options = {k: v for k, v in metric_options.items() if v in team_df.columns}
            if not valid_options: valid_options = {"Default": list(team_df.select_dtypes(include=np.number).columns)[0]}
            selected_label = st.selectbox("Select Basis:", list(valid_options.keys()))
            metric = valid_options[selected_label]
        with c1:
            event = draw_pitch_lineup(team_df, metric, formation_name=formation)

    with tab2:
        st.subheader("Squad Performance")
        positions = team_df['main_pos'].unique().tolist()
        sel_pos = st.multiselect("Filter Position", positions, default=positions)
        num_cols = team_df.select_dtypes(include=np.number).columns.tolist()
        sel_stat = st.selectbox("Metric", num_cols)
        filtered = team_df[team_df['main_pos'].isin(sel_pos)].sort_values(sel_stat, ascending=False)
        fig = px.bar(filtered, x='Player', y=sel_stat, color='Age')
        st.plotly_chart(fig, use_container_width=True)
        
    with tab3:
        show_global_stats_module(team_df, pd.DataFrame())

def scout_interface():
    st.sidebar.header("🕵️ Scout Controls")
    scout_mode = st.sidebar.radio("Navigation", ["🔍 Position Search", "🤖 AI Recruitment Hub", "📊 Global Stats DB"])
    
    def render_team_selector(key_prefix):
        c1, c2 = st.columns(2)
        with c1:
            leagues = sorted(df_stats['Comp'].dropna().unique().tolist()) if 'Comp' in df_stats.columns else []
            sel_league = st.selectbox("Select League", ["All Leagues"] + leagues, key=f"{key_prefix}_league_select")
        with c2:
            if sel_league != "All Leagues": team_opts = sorted(df_stats[df_stats['Comp'] == sel_league]['Team'].unique())
            else: team_opts = sorted(df_stats['Team'].unique())
            sel_team = st.selectbox("Select Your Club", team_opts, index=0, key=f"{key_prefix}_team_select")
        return sel_team

    def show_team_style(team_name):
        meta = get_team_meta(team_name)
        st.info(f"🛡️ **Current Team Style:** {meta.get('style', 'Unknown')}")
        return meta.get('style', 'Unknown'), meta

    if scout_mode == "🔍 Position Search":
        st.title("Position Explorer")
        if df_stats.empty: return st.error("Stats database is empty.")
        
        positions = sorted(df_stats['main_pos'].dropna().unique().tolist())
        target_pos = st.selectbox("Target Position", positions, key="pos_explorer_main")
        
        pos_df = df_stats[df_stats['main_pos'] == target_pos].copy()
        disp_cols = ['Player', 'Team', 'Age', 'Gls_Standard_Per90', 'Ast_Standard_Per90', 'player_market_value_euro']
        if 'Min_percent_Playing.Time' in pos_df.columns: disp_cols.insert(3, 'Min_percent_Playing.Time')
        disp_cols = [c for c in disp_cols if c in pos_df.columns]
        st.dataframe(pos_df[disp_cols].sort_values('player_market_value_euro' if 'player_market_value_euro' in pos_df.columns else 'Age', ascending=False).head(50), use_container_width=True)
        
        st.divider()
        st.subheader("🧬 Position DNA Analysis")
        st.info("Predicts a player's tactical versatility based on their stats DNA.")
        dna_col1, dna_col2 = st.columns([1, 2])
        
        with dna_col1:
            dna_player = st.selectbox("Select Player to Analyze", pos_df['Player'].unique(), key="dna_player_select")
            if st.button("🧬 Analyze DNA"):
                player_row = df_stats[df_stats['Player'] == dna_player]
                if player_row.empty: st.error("Player data not found.")
                else:
                    model, classes = train_position_dna_model(df_stats)
                    if model is None: st.error("Not enough numeric data.")
                    else:
                        try:
                            drop_cols = ['Player', 'Team', 'Nation', 'Pos', 'main_pos', 'target_class', 
                                         'Born', 'Age', 'Matches', 'Starts', 'Mins', 'Url', '90s', 
                                         'Comp', 'Season', 'League', 'Squad', 'player_market_value_euro']
                            numeric_cols = df_stats.select_dtypes(include=[np.number]).columns.tolist()
                            feature_cols = [f for f in numeric_cols if f not in drop_cols]
                            
                            X_input = player_row[feature_cols]
                            probs = model.predict_proba(X_input)[0]
                            dna_res = pd.DataFrame({'Position': classes, 'DNA %': (probs * 100).round(1)}).sort_values('DNA %', ascending=False)
                            
                            with dna_col2:
                                st.markdown(f"#### DNA Profile: **{dna_player}**")
                                best_role = dna_res.iloc[0]['Position']
                                if best_role == merge_position_class(target_pos): st.success(f"✅ Natural {best_role}.")
                                else: st.warning(f"⚠️ Hybrid Profile (Stats favor {best_role}).")
                                
                                st.bar_chart(dna_res.set_index('Position'), color="#3b8ed0")
                                st.dataframe(dna_res.style.background_gradient(cmap="Blues"), use_container_width=True)
                        except Exception as e: st.error(f"Analysis failed: {e}")

    elif scout_mode == "🤖 AI Recruitment Hub":
        st.title("🤖 AI Recruitment Center")
        ai_tab1, ai_tab2, ai_tab3, ai_tab4 = st.tabs(["🔮 Transfer Predictor", "🧬 Squad Chemistry", "🧪 Tactical Needs", "🕵️ Advanced Scout"])
        
        with ai_tab1:
            st.subheader("3-Stage Impact Engine (Legacy)")
            my_team = render_team_selector("legacy_main")
            my_style, meta = show_team_style(my_team)
            
            with st.expander("🔍 Filter Market Candidates", expanded=True):
                f1, f2 = st.columns(2)
                min_age, max_age = int(df_stats['Age'].min()), int(df_stats['Age'].max())
                sel_age = f1.slider("Age Range", min_age, max_age, (min_age, max_age), key="leg_slider_age")
                avail_pos = sorted(df_stats['main_pos'].dropna().unique())
                sel_pos = f2.multiselect("Position", avail_pos, key="leg_multi_pos")
                f3, f4 = st.columns(2)
                avail_leagues = sorted(df_stats['Comp'].dropna().unique())
                sel_leagues = f3.multiselect("League", avail_leagues, key="leg_multi_league")
                if sel_leagues: avail_teams = sorted(df_stats[df_stats['Comp'].isin(sel_leagues)]['Team'].unique())
                else: avail_teams = sorted(df_stats['Team'].unique())
                sel_teams = f4.multiselect("Club", avail_teams, key="leg_multi_team")
                
            market = df_stats[
                (df_stats['Age'] >= sel_age[0]) & (df_stats['Age'] <= sel_age[1])
            ]
            if sel_pos: market = market[market['main_pos'].isin(sel_pos)]
            if sel_leagues: market = market[market['Comp'].isin(sel_leagues)]
            if sel_teams: market = market[market['Team'].isin(sel_teams)]
            
            if market.empty: st.warning("No players found.")
            else:
                l_col1, l_col2 = st.columns([3, 1])
                l_player = l_col1.selectbox("Select Candidate", market['Player'].unique(), key="leg_select_candidate")
                if l_col2.button("Run Model", key="leg_run_btn"):
                    row = market[market['Player'] == l_player].iloc[0]
                    prob, impact, roi = predict_3_stage(row, meta)
                    st.divider()
                    g1, g2, g3 = st.columns(3)
                    with g1:
                        fig_prob = go.Figure(go.Indicator(mode="gauge+number", value=prob, title={'text': "Success Probability"}, gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "darkblue"}, 'steps': [{'range': [0, 40], 'color': "#ffcccc"}, {'range': [40, 70], 'color': "#ffffcc"}, {'range': [70, 100], 'color': "#ccffcc"}]}))
                        fig_prob.update_layout(height=250, margin=dict(t=50, b=20))
                        st.plotly_chart(fig_prob, use_container_width=True)
                        st.caption("ℹ️ **Success Rate:** % chance of performing above league average.")
                    with g2:
                        st.metric("Predicted Elo Impact", f"{impact:+.2f}")
                        st.info("ℹ️ **Elo Impact:** Points added to Team Rating.")
                    with g3:
                        fig_roi = go.Figure(go.Indicator(mode="gauge+number", value=min(roi, 10), title={'text': "ROI Score"}, gauge={'axis': {'range': [0, 10]}, 'bar': {'color': "gold"}}))
                        fig_roi.update_layout(height=250, margin=dict(t=50, b=20))
                        st.plotly_chart(fig_roi, use_container_width=True)
                        st.caption("ℹ️ **ROI:** Efficiency score.")

        with ai_tab2:
            st.subheader("🧬 Squad Chemistry Engine")
            if 'chemistry' not in sys.modules: st.error("Chemistry module not loaded.")
            else:
                chem_team = render_team_selector("chem_main")
                my_style_c, _ = show_team_style(chem_team)
                squad_ratings = pd.DataFrame()
                if not df_ratings.empty:
                    target_norm = aggressive_normalize(chem_team)
                    if 'team_norm' in df_ratings.columns: squad_ratings = df_ratings[df_ratings['team_norm'] == target_norm]
                    if squad_ratings.empty: 
                        mask = df_ratings['team_name'].astype(str).str.lower().str.contains(chem_team.lower().strip())
                        squad_ratings = df_ratings[mask]
                
                if squad_ratings.empty: st.warning(f"No ratings found for {chem_team}.")
                else:
                    st.success(f"Loaded {len(squad_ratings)} players.")
                    with st.expander("🔍 Filter Database", expanded=True):
                         c_f1, c_f2 = st.columns(2)
                         avail_pos_r = sorted(df_ratings['main_pos'].dropna().unique())
                         sel_pos_r = c_f1.multiselect("Position", avail_pos_r, key="chem_multi_pos")
                         min_age, max_age = int(df_ratings['Age'].min()), int(df_ratings['Age'].max())
                         sel_age_r = c_f2.slider("Age", min_age, max_age, (min_age, max_age), key="chem_slider_age")
                         
                         filtered_ratings = df_ratings[
                             (df_ratings['Age'] >= sel_age_r[0]) & (df_ratings['Age'] <= sel_age_r[1])
                         ]
                         if sel_pos_r: filtered_ratings = filtered_ratings[filtered_ratings['main_pos'].isin(sel_pos_r)]

                    sim_mode = st.radio("Tool:", ["🧪 Pairwise Simulator", "🔥 Best Partner Finder"], horizontal=True, key="chem_mode_radio")
                    if sim_mode == "🧪 Pairwise Simulator":
                        c1, c2 = st.columns(2)
                        p1_name = c1.selectbox("New Signing", filtered_ratings['player_name'].unique(), key="chem_select_p1")
                        p2_name = c2.selectbox("Existing Member", squad_ratings['player_name'].unique(), key="chem_select_p2")
                        if st.button("Check Chemistry"):
                             p1 = df_ratings[df_ratings['player_name'] == p1_name].iloc[0]
                             p2 = squad_ratings[squad_ratings['player_name'] == p2_name].iloc[0]
                             score = chemistry.predict_pair_chemistry(p1, p2)
                             st.metric("Link Score", f"{score:.1f}/100")
                             safe_score = max(0.0, min(1.0, score/100))
                             st.progress(safe_score)
                             if score > 80: st.success("✅ Excellent Fit")
                             elif score < 50: st.error("❌ Potential Clash")
                    elif sim_mode == "🔥 Best Partner Finder":
                         p1_name = st.selectbox("New Signing", filtered_ratings['player_name'].unique(), key="chem_best_p1")
                         top_n = st.number_input("Max Results", 1, 20, 5)
                         with st.expander("Filter Partners", expanded=False):
                             team_stats = df_stats[df_stats['Team'] == chem_team]
                             possible_role_cols = [c for c in team_stats.columns if 'role' in c.lower()]
                             stats_role_col = 'Role'
                             if possible_role_cols: stats_role_col = possible_role_cols[0] if 'Role' in possible_role_cols else possible_role_cols[0]
                             
                             sq_pos = sorted(team_stats['main_pos'].dropna().unique())
                             target_sq_pos = st.multiselect("Position", sq_pos)
                             sq_roles = sorted(team_stats[stats_role_col].dropna().unique()) if stats_role_col in team_stats.columns else []
                             target_sq_role = st.multiselect(f"Role ({stats_role_col})", sq_roles)
                             
                         if st.button("Find Best Partner"):
                             p1_row = df_ratings[df_ratings['player_name'] == p1_name].iloc[0]
                             valid_partners = team_stats.copy()
                             if target_sq_pos: valid_partners = valid_partners[valid_partners['main_pos'].isin(target_sq_pos)]
                             if target_sq_role and stats_role_col in valid_partners.columns: valid_partners = valid_partners[valid_partners[stats_role_col].isin(target_sq_role)]
                             
                             allowed_clean = [clean_txt(n) for n in valid_partners['Player'].unique()]
                             target_squad = squad_ratings[squad_ratings['player_name'].apply(clean_txt).isin(allowed_clean)]
                             
                             role_map = {}
                             if stats_role_col in team_stats.columns:
                                 for _, r in team_stats.iterrows(): role_map[clean_txt(r['Player'])] = r[stats_role_col]
                                 
                             if target_squad.empty: st.warning("No partners found.")
                             else:
                                 results = []
                                 progress = st.progress(0)
                                 for i, (_, p2) in enumerate(target_squad.iterrows()):
                                     if p2['player_name'] == p1_name: continue
                                     score = chemistry.predict_pair_chemistry(p1_row, p2)
                                     disp_role = role_map.get(clean_txt(p2['player_name']), "N/A")
                                     results.append({"Teammate": p2['player_name'], "Role": disp_role, "Chemistry": score})
                                     progress.progress((i+1)/len(target_squad))
                                 
                                 res_df = pd.DataFrame(results).sort_values("Chemistry", ascending=False).head(top_n)
                                 st.table(res_df.style.background_gradient(subset=['Chemistry'], cmap="Greens"))

        with ai_tab3:
            st.subheader("🧪 Tactical Gap Analysis")
            needs_team = render_team_selector("needs_main")
            show_team_style(needs_team)
            if 'chemistry' in sys.modules:
                needs = chemistry.identify_team_needs(df_stats, needs_team)
                if needs:
                    for role, urgency in needs.items(): st.warning(f"**{role}**: {urgency}")
                    market = df_stats[df_stats['Team'] != needs_team].copy()
                    if not market.empty:
                        market['Fit_Score'] = market.apply(lambda x: chemistry.calculate_fit_score(x, needs, df_stats), axis=1)
                        cols = [c for c in ['Player', 'Team', 'Age', 'Fit_Score', 'player_market_value_euro'] if c in market.columns]
                        st.dataframe(market.sort_values('Fit_Score', ascending=False).head(10)[cols], use_container_width=True)
                else: st.success("Squad balanced.")
            else: st.error("Chemistry Missing.")

        with ai_tab4:
            st.subheader("🕵️ Advanced Scouting (TVI)")
            adv_team = render_team_selector("adv_main")
            strategy_choice = st.selectbox("Strategy", ["⚖️ BALANCED", "🏆 WIN NOW", "💰 MONEYBALL", "🧩 SYSTEM FIT"], key="adv_strategy")
            player_search = st.selectbox("Search Player", sorted(df_stats['Player'].unique()), key="adv_player")
            
            if st.button("🚀 Run Analysis"):
                impact = predict_transfer_integrated(player_search, adv_team)
                if "error" in impact: st.error(impact['error'])
                else:
                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("Current Elo", impact['current_elo'])
                    k2.metric("Impact", f"{impact['predicted_impact']:+}")
                    k3.metric("Forecast", impact['forecasted_elo'])
                    k4.metric("Verdict", impact['verdict'])
                    
                    weights = get_strategy_weights(strategy_choice)
                    chem_score = 50.0
                    if 'chemistry' in sys.modules:
                         squad_df = df_stats[df_stats['Team'] == adv_team]
                         p_row = df_stats[df_stats['Player'] == player_search]
                         if not p_row.empty and not squad_df.empty:
                             chem_map = chemistry.get_squad_chemistry_map(p_row.iloc[0], squad_df)
                             if not chem_map.empty: chem_score = chem_map['Chemistry_Score'].mean()
                    
                    mkt_val = get_player_market_value(player_search, df_stats)
                    tvi = calculate_tvi(chem_score, impact['predicted_impact'], mkt_val, weights)
                    st.progress(int(tvi)/100)
                    st.metric("TVI Score", f"{tvi}/100")

    elif scout_mode == "📊 Global Stats DB":
        show_global_stats_module(df_player_disp, df_team_disp)

# --- 7. START SCREEN ---
if 'role' not in st.session_state: st.session_state['role'] = None
if st.session_state['role'] is None:
    st.markdown("<h1 style='text-align: center;'>⚽ Football Scouting System</h1>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        c_mgr, c_scout = st.columns(2)
        if c_mgr.button("👔 Manager Interface", use_container_width=True):
            st.session_state['role'] = 'Manager'
            st.rerun()
        if c_scout.button("🕵️ Scout Interface", use_container_width=True):
            st.session_state['role'] = 'Scout'
            st.rerun()
elif st.session_state['role'] == 'Manager':
    if st.sidebar.button("⬅️ Switch Role"):
        st.session_state['role'] = None
        st.rerun()
    manager_interface()
elif st.session_state['role'] == 'Scout':
    if st.sidebar.button("⬅️ Switch Role"):
        st.session_state['role'] = None
        st.rerun()
    scout_interface()
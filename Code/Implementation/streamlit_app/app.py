import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
import pickle
import chemistry  # Your chemistry module
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="AI Football Intelligence", layout="wide", initial_sidebar_state="expanded")

# FILE PATHS
MODEL_FILE = "impact_model_hybrid.json"
LOGISTIC_FILE = "success_probability_model.pkl"
DNA_MODEL = "player_dna_model.pkl"
DNA_ENCODER = "dna_label_encoder.pkl"
STYLE_COLS_FILE = "style_columns.pkl"
ELO_FILE = "elo_end_24.csv"
CHEMISTRY_FILE = "scouting_outputs/attacking_chemistry_2024.csv"
PLAYSTYLE_FILE = "team_playstyles.csv"
WAGES_FILE = "wages.csv"
STATS_FILE = "scout_data_weighted.csv"

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
    return models

@st.cache_data
def load_data():
    data = {}
    # Load Stats
    if os.path.exists(STATS_FILE):
        data['stats'] = pd.read_csv(STATS_FILE)
    else:
        st.error(f"❌ Critical Error: {STATS_FILE} not found.")
        return pd.DataFrame() # Return empty to prevent crash
        
    data['elo'] = pd.read_csv(ELO_FILE) if os.path.exists(ELO_FILE) else pd.DataFrame()
    data['chem'] = pd.read_csv(CHEMISTRY_FILE) if os.path.exists(CHEMISTRY_FILE) else pd.DataFrame()
    data['playstyles'] = pd.read_csv(PLAYSTYLE_FILE) if os.path.exists(PLAYSTYLE_FILE) else pd.DataFrame()
    
    # Merge Wages
    if os.path.exists(WAGES_FILE):
        wages = pd.read_csv(WAGES_FILE)
        # Clean currency
        wages['GROSS P/W'] = wages['GROSS P/W'].astype(str).str.replace('€', '').str.replace(',', '')
        wages['GROSS P/W'] = pd.to_numeric(wages['GROSS P/W'], errors='coerce').fillna(0)
        
        if not data['stats'].empty:
             data['stats'] = pd.merge(data['stats'], wages[['PLAYER', 'GROSS P/W']], 
                                      left_on='Player', right_on='PLAYER', how='left')
    return data

# Initialize
models = load_models()
data = load_data()
df_stats = data.get('stats', pd.DataFrame())

# --- 3. AI & HELPER FUNCTIONS ---

def get_player_dna(row):
    """Returns Position Probabilities (e.g., '🛡️ 60% DEF | ⚔️ 40% MID')"""
    if 'dna' not in models or 'dna_enc' not in models: return "N/A"
    
    feat_cols = [
        'Gls_Standard_Per90', 'Ast_Standard_Per90', 'npxG_Per', 'xAG_Per',
        'PrgP_Per90', 'PrgC_Carries_Per90', 'TklW_Tackles_Per90', 
        'Int_Def_Per90', 'Won_percent_Aerial', 'SCA90_SCA'
    ]
    vec = row[feat_cols].fillna(0).values.reshape(1, -1)
    probs = models['dna'].predict_proba(vec)[0]
    classes = models['dna_enc'].classes_
    indices = np.argsort(probs)[::-1]
    
    return f"{probs[indices[0]]*100:.0f}% {classes[indices[0]]} | {probs[indices[1]]*100:.0f}% {classes[indices[1]]}"

def predict_3_stage(candidate_row, team_meta):
    """Calculates Success%, Elo Impact, and ROI."""
    if 'reg' not in models or 'class' not in models: return 0, 0, 0
    
    feat_cols = [
        'Gls_Standard_Per90', 'Ast_Standard_Per90', 'npxG_Per', 'xAG_Per',
        'PrgP_Per90', 'PrgC_Carries_Per90', 'TklW_Tackles_Per90', 
        'Int_Def_Per90', 'Won_percent_Aerial', 'SCA90_SCA'
    ]
    stats_vec = candidate_row[feat_cols].fillna(0).values
    current_elo = team_meta['elo']
    
    wage = float(candidate_row.get('GROSS P/W', 5000))
    try:
        raw_mv = str(candidate_row.get('player_market_value_euro', 0)).replace('€', '').replace(',', '')
        mv = float(raw_mv)
    except: mv = 1000000

    input_vec = np.concatenate([
        [current_elo], stats_vec, 
        [np.log1p(mv), np.log1p(wage)], 
        [1 if f"Style_{team_meta['style']}" == c else 0 for c in models.get('styles', [])] if 'styles' in models else [0]*5
    ])
    
    prob_success = models['class'].predict_proba([input_vec])[0][1] * 100
    elo_impact = models['reg'].predict([input_vec])[0]
    cost_est = max(100000, mv + (wage * 52 * 3))
    roi = (elo_impact * (prob_success/100) * 1000000) / cost_est 
    
    return prob_success, elo_impact, roi

def get_team_meta(team_name):
    meta = {'elo': 1500, 'style': 'Unknown'}
    if not data['elo'].empty:
        match = data['elo'][data['elo'].apply(lambda x: str(x).lower().find(team_name.lower()) != -1, axis=1)]
        if not match.empty:
            col = next((c for c in match.columns if 'elo' in c.lower()), None)
            if col: meta['elo'] = match.iloc[0][col]
    if not data['playstyles'].empty:
        match = data['playstyles'][data['playstyles']['Team'] == team_name]
        if not match.empty: meta['style'] = match.iloc[0]['Main_Style']
    return meta

# --- 4. VISUALIZATION MODULES ---

def draw_pitch_lineup(team_df, metric='Min_Playing'):
    """
    Draws a 4-3-3 formation selecting the best players based on the metric.
    Interactive: Returns the player selected.
    """
    # 1. Select Best Players for 4-3-3
    # Mappings based on 'main_pos' or 'player_position'
    gk = team_df[team_df['main_pos'].str.contains('GK', na=False)].sort_values(metric, ascending=False).head(1)
    defs = team_df[team_df['main_pos'].str.contains('Def', na=False)].sort_values(metric, ascending=False).head(4)
    mids = team_df[team_df['main_pos'].str.contains('Mid', na=False)].sort_values(metric, ascending=False).head(3)
    atts = team_df[team_df['main_pos'].str.contains('For|Wing', na=False)].sort_values(metric, ascending=False).head(3)
    
    lineup = pd.concat([gk, defs, mids, atts])
    
    # 2. Assign Pitch Coordinates (Simple 4-3-3)
    # X = Width (0-100), Y = Length (0-100)
    coords = [
        (50, 5), # GK
        (15, 25), (38, 25), (62, 25), (85, 25), # Defenders
        (25, 55), (50, 50), (75, 55), # Mids
        (15, 85), (50, 90), (85, 85)  # Atts
    ]
    
    # Handle case where team has fewer players
    safe_len = min(len(lineup), len(coords))
    lineup = lineup.iloc[:safe_len].copy()
    current_coords = coords[:safe_len]
    
    lineup['x'] = [c[0] for c in current_coords]
    lineup['y'] = [c[1] for c in current_coords]
    
    # 3. Create Plot
    fig = go.Figure()
    
    # Draw Pitch (Green Rectangle)
    fig.add_shape(type="rect", x0=0, y0=0, x1=100, y1=100,
        line=dict(color="white"), fillcolor="#40a860", layer="below")
    
    # Add Players
    fig.add_trace(go.Scatter(
        x=lineup['x'], y=lineup['y'],
        mode='markers+text',
        text=lineup['Player'],
        textposition="bottom center",
        marker=dict(size=18, color='white', line=dict(width=2, color='black')),
        hovertext=lineup.apply(lambda row: f"{row['Player']}<br>{metric}: {row[metric]}", axis=1),
        hoverinfo="text",
        customdata=lineup['Player'] # Used for click event
    ))
    
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, visible=False, range=[0, 100]),
        yaxis=dict(showgrid=False, zeroline=False, visible=False, range=[0, 100]),
        height=600, margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor="#40a860", dragmode='select'
    )
    
    # Render with Selection Event
    event = st.plotly_chart(fig, use_container_width=True, on_select="rerun")
    return event

def show_global_stats_module(df):
    """Shared Stats Interface for both Managers and Scouts."""
    st.subheader("📊 Deep Dive Statistics")
    
    stat_mode = st.radio("Analysis Type:", ["Single Metric Ranking", "Multi-Metric Comparison"], horizontal=True)
    
    if stat_mode == "Single Metric Ranking":
        c1, c2, c3 = st.columns(3)
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        
        target_stat = c1.selectbox("Select Metric", numeric_cols, index=numeric_cols.index('Gls_Standard_Per90') if 'Gls_Standard_Per90' in numeric_cols else 0)
        top_n = c2.slider("Show Top N Players", 5, 50, 15)
        per_90 = c3.checkbox("Normalize Per 90 (if applicable)", value=True)
        
        # Prepare Data
        chart_data = df.copy()
        if per_90 and "Per90" not in target_stat and "Min_Playing" in df.columns:
            # Quick calc if user asks for per 90 on a raw stat
            chart_data[target_stat] = (chart_data[target_stat] / chart_data['Min_Playing']) * 90
            
        chart_data = chart_data.sort_values(target_stat, ascending=False).head(top_n)
        
        fig = px.bar(chart_data, x=target_stat, y='Player', orientation='h', 
                     color=target_stat, color_continuous_scale='Viridis',
                     title=f"Top {top_n} - {target_stat}", text_auto='.2f')
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
    else: # Multi-Metric
        c1, c2, c3 = st.columns(3)
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        
        x_stat = c1.selectbox("X-Axis Metric", numeric_cols, index=0)
        y_stat = c2.selectbox("Y-Axis Metric", numeric_cols, index=1)
        
        fig = px.scatter(df, x=x_stat, y=y_stat, hover_name='Player', color='main_pos',
                         title=f"{y_stat} vs {x_stat}", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

# --- 5. MAIN INTERFACES ---

def manager_interface():
    st.sidebar.header("🏢 Manager Controls")
    
    # 1. League & Team Selector
    leagues = sorted(df_stats['Comp'].dropna().unique().tolist())
    selected_league = st.sidebar.selectbox("Select Competition", leagues)
    
    league_teams = sorted(df_stats[df_stats['Comp'] == selected_league]['Team'].unique().tolist())
    selected_team = st.sidebar.selectbox("Select Team", league_teams)
    
    team_df = df_stats[df_stats['Team'] == selected_team].copy()
    meta = get_team_meta(selected_team)
    
    st.title(f"{selected_team} Manager Dashboard")
    st.info(f"📋 **Style:** {meta['style']} | 📈 **Elo:** {meta['elo']:.0f}")
    
    # TABS
    tab1, tab2, tab3 = st.tabs(["🏟️ Lineup & Squad", "📉 Performance Metrics", "📊 Advanced Stats"])
    
    # --- TAB 1: LINEUP ---
    with tab1:
        st.subheader("Starting XI Visualizer")
        c1, c2 = st.columns([3, 1])
        with c2:
            metric = st.selectbox("Select Basis:", ["Min_Playing", "Starts_Playing"])
            st.caption("Auto-selects best players in a 4-3-3 shape.")
        
        with c1:
            # Draw Pitch and capture click event
            event = draw_pitch_lineup(team_df, metric)
        
        # Handle Interactivity (Click on player)
        if event and event['selection']['points']:
            player_idx = event['selection']['points'][0]['point_index']
            # We need to re-sort the lineup exactly as the function did to find the player
            # (Simplified: Just search by name from the customdata we passed)
            p_name = event['selection']['points'][0]['customdata']
            
            st.divider()
            st.markdown(f"### 👤 Analysis: {p_name}")
            
            p_row = team_df[team_df['Player'] == p_name].iloc[0]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Age", p_row['Age'])
            c2.metric("Matches", p_row['MP_Playing'])
            c3.metric("Minutes", p_row['Min_Playing'])
            c4.metric("Goals/90", f"{p_row.get('Gls_Standard_Per90', 0):.2f}")
            
            st.text(f"Positional DNA: {get_player_dna(p_row)}")

    # --- TAB 2: PERFORMANCE METRICS ---
    with tab2:
        st.subheader("Squad Performance")
        
        c1, c2 = st.columns(2)
        positions = team_df['main_pos'].unique().tolist()
        sel_pos = c1.multiselect("Filter Position", positions, default=positions)
        
        num_cols = team_df.select_dtypes(include=np.number).columns.tolist()
        sel_stat = c2.selectbox("Metric to Analyze", num_cols, index=num_cols.index('Min_Playing') if 'Min_Playing' in num_cols else 0)
        
        filtered = team_df[team_df['main_pos'].isin(sel_pos)].sort_values(sel_stat, ascending=False)
        
        fig = px.bar(filtered, x='Player', y=sel_stat, color='Age', title=f"{sel_stat} by Player")
        st.plotly_chart(fig, use_container_width=True)

    # --- TAB 3: SHARED STATS ---
    with tab3:
        show_global_stats_module(team_df)

def scout_interface():
    st.sidebar.header("🕵️ Scout Controls")
    scout_mode = st.sidebar.radio("Navigation", ["🔍 Position Search", "🤖 AI Recruitment Hub", "📊 Global Stats DB"])
    
    if scout_mode == "🔍 Position Search":
        st.title("Position Explorer")
        
        positions = sorted(df_stats['main_pos'].dropna().unique().tolist())
        target_pos = st.selectbox("Target Position", positions)
        
        # Show Top players in that position based on Fit Score Logic or General Rating
        # For simplicity, we show top sorted by minutes (reliability) or value
        st.subheader(f"Top Talent: {target_pos}")
        
        pos_df = df_stats[df_stats['main_pos'] == target_pos].copy()
        
        # Simple Table
        st.dataframe(
            pos_df[['Player', 'Team', 'Age', 'Min_Playing', 'Gls_Standard_Per90', 'Ast_Standard_Per90']].sort_values('Min_Playing', ascending=False).head(20),
            use_container_width=True
        )

    elif scout_mode == "🤖 AI Recruitment Hub":
        st.title("🤖 AI Recruitment Center")
        
        # User needs to pick a team context for the AI to work
        my_team = st.selectbox("Select Your Club (for Context)", sorted(df_stats['Team'].unique()))
        meta = get_team_meta(my_team)
        
        ai_tab1, ai_tab2, ai_tab3 = st.tabs(["🔮 Transfer Predictor", "🧬 Squad DNA & Chemistry", "🧪 Tactical Needs"])
        
        with ai_tab1:
            st.subheader("3-Stage Impact Engine")
            st.info("Calculates: 1. Success Probability -> 2. Elo Impact -> 3. ROI")
            
            # Needs Analysis
            needs = chemistry.identify_team_needs(df_stats, my_team)
            if needs:
                market = df_stats[df_stats['Team'] != my_team].copy()
                market['Fit_Score'] = market.apply(lambda x: chemistry.calculate_fit_score(x, needs, df_stats), axis=1)
                candidates = market.sort_values('Fit_Score', ascending=False).head(10)
                
                results = []
                for _, row in candidates.iterrows():
                    prob, impact, roi = predict_3_stage(row, meta)
                    results.append({
                        'Player': row['Player'], 'Fit': row['Fit_Score'],
                        'Success %': prob, 'Impact': impact, 'ROI': roi
                    })
                
                st.dataframe(pd.DataFrame(results).style.background_gradient(subset=['ROI'], cmap="Greens"), use_container_width=True)
            else:
                st.write("No critical needs identified.")

        with ai_tab2:
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("🧬 DNA Profiler")
                target_p = st.selectbox("Analyze Player DNA", df_stats['Player'].unique())
                p_row = df_stats[df_stats['Player'] == target_p].iloc[0]
                st.metric("Positional Reality", get_player_dna(p_row))
            
            with c2:
                st.subheader("🔗 Link-Up Chemistry")
                chem_df = data['chem']
                if not chem_df.empty and 'team_name' in chem_df.columns:
                    team_chem = chem_df[chem_df['team_name'] == my_team].sort_values('link_value', ascending=False).head(10)
                    st.dataframe(team_chem[['player_name', 'next_player', 'link_value']], hide_index=True)
                else:
                    st.warning("Chemistry data unavailable.")

    elif scout_mode == "📊 Global Stats DB":
        show_global_stats_module(df_stats)

# --- 6. START SCREEN & ROUTING ---

if 'role' not in st.session_state:
    st.session_state['role'] = None

if st.session_state['role'] is None:
    st.markdown("<h1 style='text-align: center;'>⚽ Football Intelligence Suite</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center;'>Select Your Interface</h4>", unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        col_mgr, col_scout = st.columns(2)
        if col_mgr.button("👔 Manager Interface", use_container_width=True):
            st.session_state['role'] = 'Manager'
            st.rerun()
        if col_scout.button("🕵️ Scout Interface", use_container_width=True):
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
print("--- SCRIPT IS STARTING ---")
import pandas as pd
# ... rest of your imports
import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine, text
from rapidfuzz import process, fuzz
from tqdm import tqdm

# --- 1. CONFIGURATION ---
DB_URL = "postgresql://postgres:mshro3post@localhost:5432/Football_project"
engine = create_engine(DB_URL)

# FILE CONFIGURATION
FILES = {
    # GROUP 1: FILES WE READ TO FIND "TEAMS" 
    'team_sources': [
        'elo_end_24.csv', 'elo_end_23.csv', 'elo_new.csv', 'elo_old.csv',
        'elo_start_23.csv', 'elo_start_24.csv',
        'team_display_data.csv', 'MATCHES_FILE.csv', 'Team_Playstyles.csv',
        'transfers.csv' # Contains 'origin_club', 'new_club'
    ],

    # GROUP 2: FILES WE READ TO FIND "PLAYERS" 
    'player_sources': [
        'scout_outfield.csv', 'scout_gk.csv',
        'scout_outfield_composites.csv', 'scout_gk_composites.csv',
        'players_display_data.csv', 'global_player_ratings.csv',
        'market_values.csv', 'implied_transfers_2024.csv', 'wages.csv'
    ],

    # GROUP 3: DATA MAPPING (Filename -> Database Table Name)
    'data_map': {
        # -- Core Stats --
        'scout_outfield.csv': 'scout_outfield',
        'scout_gk.csv': 'scout_gk',
        'scout_outfield_composites.csv': 'scout_outfield_composites',
        'scout_gk_composites.csv': 'scout_gk_composites',
        
        # -- Longitudinal / History --
        'players_display_data.csv': 'player_display_stats',
        'team_display_data.csv': 'team_display_stats',
        'Team_Playstyles.csv': 'team_playstyles',
        
        # -- Matches & ELO --
        'MATCHES_FILE.csv': 'matches',
        'elo_end_24.csv': 'elo_history', 'elo_end_23.csv': 'elo_history',
        'elo_new.csv': 'elo_history', 'elo_old.csv': 'elo_history',
        
        # -- Market & Money --
        'transfers.csv': 'transfers',
        'implied_transfers_2024.csv': 'implied_transfers',
        'market_values.csv': 'market_values',
        'wages.csv': 'wages',
        'global_player_ratings.csv': 'ratings',
        
        # -- Complex/Raw --
        'chemistry_training_set.csv': 'chemistry_data' 
    }
}

# --- 2. TEXT NORMALIZATION ---
def aggressive_normalize(name):
    if pd.isna(name): return ""
    name = str(name).lower()
    name = re.sub(r'\bfc\b|\bafc\b|\bsc\b', '', name) 
    name = re.sub(r'[^a-z0-9]', ' ', name)
    return name.strip()

# --- 3. CLASS: INTELLIGENT MAPPER ---
class SmartMapper:
    def __init__(self, engine):
        self.engine = engine
        self.team_cache = {}   
        self.player_cache = {} 

    def load_caches(self):
        """Loads existing IDs from DB into memory"""
        with self.engine.connect() as conn:
            try:
                result = conn.execute(text("SELECT team_id, team_name FROM teams"))
                for row in result:
                    self.team_cache[aggressive_normalize(row.team_name)] = row.team_id
            except: pass

            try:
                # We load ALL players.
                result = conn.execute(text("SELECT player_id, player_name, team_id FROM players"))
                for row in result:
                    key = (aggressive_normalize(row.player_name), row.team_id)
                    self.player_cache[key] = row.player_id
            except: pass

    def get_team_id(self, raw_name, create=False):
        if pd.isna(raw_name): return None
        norm_name = aggressive_normalize(raw_name)
        
        if norm_name in self.team_cache: return self.team_cache[norm_name]
        
        if self.team_cache:
            match = process.extractOne(norm_name, self.team_cache.keys(), scorer=fuzz.token_sort_ratio)
            if match and match[1] > 88: 
                return self.team_cache[match[0]]
        
        if create:
            with self.engine.connect() as conn:
                # Truncate if name is too long for DB
                safe_name = str(raw_name)[:255]
                res = conn.execute(
                    text("INSERT INTO teams (team_name) VALUES (:name) RETURNING team_id"),
                    {"name": safe_name}
                )
                new_id = res.scalar()
                conn.commit()
            self.team_cache[norm_name] = new_id
            return new_id
        return None

    def get_player_id(self, name, team_id, create=False):
        key = (aggressive_normalize(name), team_id)
        if key in self.player_cache: return self.player_cache[key]
        
        if create:
            with self.engine.connect() as conn:
                safe_name = str(name)[:255]
                res = conn.execute(
                    text("INSERT INTO players (player_name, team_id) VALUES (:name, :tid) RETURNING player_id"),
                    {"name": safe_name, "tid": team_id}
                )
                new_id = res.scalar()
                conn.commit()
            self.player_cache[key] = new_id
            return new_id
        return None

# --- 4. SCHEMA CREATION ---
def create_schema():
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS teams (
                team_id SERIAL PRIMARY KEY,
                team_name VARCHAR(255) UNIQUE
            );
            CREATE TABLE IF NOT EXISTS players (
                player_id SERIAL PRIMARY KEY,
                player_name VARCHAR(255),
                team_id INT REFERENCES teams(team_id),
                UNIQUE(player_name, team_id)
            );
        """))
        conn.commit()
        print("✅ Base Schema Created.")

# --- 5. MIGRATION LOGIC ---
def run_migration():
    mapper = SmartMapper(engine)
    create_schema()
    mapper.load_caches()

    # --- STEP A: BUILD MASTER TEAMS LIST ---
    print("\n--- 1. Building Master Team List ---")
    
    # Extended list of possible column headers for Teams
    team_cols = ['Team', 'team', 'team_name', 'club', 'current_club', 
                 'origin_club', 'new_club', 'club_2']

    for fname in FILES['team_sources']:
        try:
            df = pd.read_csv(fname)
            # Find ANY column that looks like a team name
            found_cols = [c for c in df.columns if c in team_cols]
            
            for col in found_cols:
                for t_name in df[col].dropna().unique():
                    mapper.get_team_id(t_name, create=True)
                
        except Exception as e: print(f"⚠️ Skipping {fname}: {e}")
    
    print(f"✅ Total Teams: {len(mapper.team_cache)}")

    # --- STEP B: BUILD MASTER PLAYER LIST ---
    print("\n--- 2. Building Master Player List ---")
    
    # Extended list for Player Names
    player_cols = ['Player', 'player_name', 'player', 'PLAYER']
    
    for fname in FILES['player_sources']:
        try:
            df = pd.read_csv(fname)
            
            p_col = next((c for c in player_cols if c in df.columns), None)
            # Find team col (same list as above)
            t_col = next((c for c in team_cols if c in df.columns), None)
            
            if not p_col: continue

            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Scanning {fname}"):
                p_name = row[p_col]
                t_name = row[t_col] if t_col and pd.notna(row[t_col]) else None
                
                tid = mapper.get_team_id(t_name) if t_name else None
                mapper.get_player_id(p_name, tid, create=True)
                
        except Exception as e: print(f"⚠️ Skipping {fname}: {e}")

    # --- STEP C: UPLOAD DATA TABLES ---
    print("\n--- 3. Uploading Data Tables ---")
    
    for fname, table_name in FILES['data_map'].items():
        try:
            print(f"Processing {fname} -> {table_name}")
            df = pd.read_csv(fname)
            
            # --- 1. Identify Columns ---
            p_col = next((c for c in player_cols if c in df.columns), None)
            t_col = next((c for c in team_cols if c in df.columns), None)
            
            # --- 2. Map Team IDs ---
            for col in df.columns:
                if col in team_cols:
                    # Create a new column e.g., 'origin_club_id'
                    new_col_name = col.replace('club', 'team_id').replace('Team', 'team_id').replace('team', 'team_id')
                    if not new_col_name.endswith('_id'): new_col_name += '_id'
                    
                    # Map it
                    df[new_col_name] = df[col].apply(lambda x: mapper.get_team_id(x))

            # --- 3. Map Player IDs ---
            if p_col:
                # We try to use the PRIMARY team column to identify the player
                def get_pid(row):
                    # Pick the first valid team column we found to help identify the player
                    tid = row.get(t_col) if t_col else None
                    if tid and isinstance(tid, str): tid = mapper.get_team_id(tid) # If it wasn't mapped yet
                    
                    return mapper.get_player_id(row[p_col], tid)
                
                df['player_id'] = df.apply(get_pid, axis=1)
                
                # Special handling for chemistry or ratings: 
                if 'scout' in table_name:
                    df = df.dropna(subset=['player_id'])

            # --- 4. Upload ---           
            if 'elo' in table_name or 'display' in table_name:
                df['source_file'] = fname

            df.to_sql(table_name, engine, if_exists='append', index=False)
            
            # --- 5. Vector Conversion ---
            vector_cols = [c for c in df.columns if 'embedding' in c]
            if vector_cols:
                with engine.connect() as conn:
                    for v_col in vector_cols:
                        conn.execute(text(f"ALTER TABLE {table_name} ALTER COLUMN \"{v_col}\" TYPE vector USING \"{v_col}\"::vector"))
                        conn.commit()
                    
        except Exception as e:
            print(f"❌ Error uploading {fname}: {e}")

    print("\n✅ MIGRATION COMPLETE!")

if __name__ == "__main__":
    run_migration()
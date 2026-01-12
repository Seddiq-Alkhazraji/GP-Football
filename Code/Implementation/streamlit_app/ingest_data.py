import pandas as pd
import numpy as np
import psycopg2
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

# --- CONFIGURATION ---
CSV_FILE = "scout_data_weighted.csv"

# The 10 Key Metrics that define a player's "Style"
FEATURES = [
    'Gls_Standard_Per90', 'Ast_Standard_Per90', 'npxG_Per', 'xAG_Per',
    'PrgP_Per90', 'PrgC_Carries_Per90', 'TklW_Tackles_Per90', 
    'Int_Def_Per90', 'Won_percent_Aerial', 'SCA90_SCA'
]

def init_connection():
    try:
        conn = psycopg2.connect(**st.secrets["postgres"])
        return conn
    except Exception as e:
        print(f"❌ Connection Failed: {e}")
        return None

def process_and_upload():
    print("🚀 Starting Data Ingestion...")
    
    # 1. Load Data
    try:
        df = pd.read_csv(CSV_FILE)
        print(f"✅ Loaded {len(df)} rows from {CSV_FILE}")
    except FileNotFoundError:
        print(f"❌ Error: {CSV_FILE} not found.")
        return

    # 2. Data Cleaning & Normalization
    df[FEATURES] = df[FEATURES].fillna(0)
    
    scaler = MinMaxScaler()
    df_normalized = df.copy()
    df_normalized[FEATURES] = scaler.fit_transform(df[FEATURES])
    
    # 3. Connect to DB
    conn = init_connection()
    if not conn: return
    cur = conn.cursor()

    try:
        # A. Clean Start: Drop Old Tables
        print("🧹 Cleaning up old schema...")
        cur.execute("DROP TABLE IF EXISTS player_stats_vector;")
        cur.execute("DROP TABLE IF EXISTS players_metadata;")
        
        # B. Create Tables (Re-creating schema)
        cur.execute("""
            CREATE TABLE players_metadata (
                player_id SERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                team TEXT,
                main_pos TEXT,
                nation TEXT,
                age INTEGER,
                season TEXT
            );
        """)
        
        cur.execute("""
            CREATE TABLE player_stats_vector (
                vector_id SERIAL PRIMARY KEY,
                player_id INTEGER REFERENCES players_metadata(player_id) ON DELETE CASCADE,
                stats_embedding vector(10)
            );
        """)
        print("🏗️  Schema created successfully.")

        # 4. Loop through rows and insert
        count = 0
        for index, row in df.iterrows():
            # Insert Metadata
            cur.execute("""
                INSERT INTO players_metadata (name, team, main_pos, nation, age, season)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING player_id;
            """, (
                row.get('Player', 'Unknown'), 
                row.get('Team', 'Free Agent'),
                row.get('main_pos', 'N/A'),
                row.get('Nation', 'N/A'),
                int(row.get('Age', 0)) if pd.notnull(row.get('Age')) else 0,
                str(row.get('season', '2023'))
            ))
            
            player_id = cur.fetchone()[0]


            raw_values = df_normalized.iloc[index][FEATURES].values
            vector_values = [float(x) for x in raw_values] 

            # Insert Vector
            cur.execute("""
                INSERT INTO player_stats_vector (player_id, stats_embedding)
                VALUES (%s, %s);
            """, (player_id, str(vector_values)))
            
            count += 1
            if count % 500 == 0:
                print(f"   Processed {count} players...")

        conn.commit()
        print(f"🎉 SUCCESS! Uploaded {count} players to the database.")

    except Exception as e:
        conn.rollback()
        print(f"❌ Error during upload: {e}")
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    process_and_upload()
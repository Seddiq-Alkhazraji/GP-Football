print("--- SCRIPT STARTED ---")
import pandas as pd
import joblib
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

print("📉 Training 'Lite' Chemistry Model...")

# FILES
TRAIN_FILE = "scouting_outputs/chemistry_training_set.csv"
OUTPUT_MODEL = "chemistry_rf_model.pkl"
OUTPUT_ENCODER = "role_encoder.pkl"

# 1. Load Data
if not os.path.exists(TRAIN_FILE):
    if os.path.exists("chemistry_training_set.csv"):
        TRAIN_FILE = "chemistry_training_set.csv"
    else:
        print(f"❌ Error: Could not find training data.")
        exit()

df = pd.read_csv(TRAIN_FILE)
print(f"   Loaded {len(df)} rows.")

# 2. FIX ENCODING 
print("   ⚙️ Encoding Player Roles...")


le = LabelEncoder()

# We combine both columns to make sure the encoder learns EVERY possible role
all_roles = pd.concat([df['role_p1'], df['role_p2']]).astype(str).unique()
le.fit(all_roles)

# Create the missing columns
df['role_p1_enc'] = le.transform(df['role_p1'].astype(str))
df['role_p2_enc'] = le.transform(df['role_p2'].astype(str))

# Save the encoder
joblib.dump(le, OUTPUT_ENCODER)
print(f"   ✅ Saved new role encoder to '{OUTPUT_ENCODER}'")

# 3. Define Features
features = ['role_p1_enc', 'off_p1', 'def_p1', 'role_p2_enc', 'off_p2', 'def_p2']
X = df[features]
y = df['total_joint_impact']

# 4. Train "Lite" Model
print("   Fitting Lite Model (approx 10-30s)...")
rf_lite = RandomForestRegressor(
    n_estimators=20,      
    max_depth=8,          
    min_samples_split=10,
    n_jobs=-1, 
    random_state=42
)
rf_lite.fit(X, y)

# 5. Save Model
print("   Saving model...")
joblib.dump(rf_lite, OUTPUT_MODEL, compress=3)

print("✅ SUCCESS! Lite Model Saved.")
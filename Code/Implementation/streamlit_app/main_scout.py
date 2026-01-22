import pandas as pd
from predict_impact import predict_transfer, clean_txt
from chemistry import get_squad_chemistry_map, get_target_squad

# --- LOAD RESOURCES ---
df_scout = pd.read_csv("scout_outfield.csv") 
df_scout['search_key'] = df_scout['Player'].apply(clean_txt)

def get_player_market_value(player_name):
    """Fetches market value for ROI calculation."""
    search_key = clean_txt(player_name)
    row = df_scout[df_scout['search_key'] == search_key]
    if row.empty: return 10_000_000 
    return float(row.iloc[0].get('player_market_value_euro', 10_000_000))

# --- 🆕 DYNAMIC STRATEGY SELECTOR ---
def select_strategy():
    print("\n🧠 CHOOSE YOUR STRATEGY:")
    print("   1. 🏆 WIN NOW (Galactico)  -> Ignored Cost, focused on Elo Impact.")
    print("   2. 💰 MONEYBALL (Brighton) -> Focused on ROI & Future Value.")
    print("   3. 🧩 SYSTEM FIT (Pep)     -> Focused on Tactical Match.")
    print("   4. ⚖️  BALANCED (Standard) -> Equal weight to all.")
    print("   5. 🔧 CUSTOM               -> Enter your own weights.")
    
    choice = input("\n> Select Mode (1-5): ").strip()
    
    # Returns weights: (Fit, Impact, ROI)
    if choice == '1': return (0.10, 0.80, 0.10) # Galactico
    if choice == '2': return (0.30, 0.10, 0.60) # Moneyball
    if choice == '3': return (0.70, 0.20, 0.10) # System Fit
    if choice == '4': return (0.33, 0.33, 0.33) # Balanced
    
    if choice == '5':
        print("\nEnter weights (Must sum to 1.0, e.g., 0.5):")
        try:
            w_f = float(input("   Fit Weight:    "))
            w_i = float(input("   Impact Weight: "))
            w_r = float(input("   ROI Weight:    "))
            return (w_f, w_i, w_r)
        except:
            print("❌ Invalid number. Using Balanced.")
            return (0.33, 0.33, 0.33)
            
    return (0.33, 0.33, 0.33) # Default fallback

# --- MAIN CALCULATION ---
def calculate_tvi(fit_score, elo_impact, market_value, weights):
    w_fit, w_impact, w_roi = weights
    
    # 1. Normalize Inputs (0-100 scale)
    norm_fit = fit_score 
    
    # Impact: Map -5..+5 to 0..100
    norm_impact = max(0, min(100, (elo_impact + 5) * 10))

    # ROI: Map 0m..100m to 100..0
    m_val = market_value / 1_000_000
    norm_roi = max(0, 100 - m_val)

    # 2. Apply Dynamic Weights
    tvi = (norm_fit * w_fit) + (norm_impact * w_impact) + (norm_roi * w_roi)
    return round(tvi, 1)

def run_analysis_interactive(player_name, target_team):
    # 1. Ask User for Strategy
    current_weights = select_strategy()
    
    print(f"\n📋 SCOUT REPORT: {player_name} ➡️ {target_team}")
    print(f"⚙️  Strategy Weights: Fit {current_weights[0]*100:.0f}% | Impact {current_weights[1]*100:.0f}% | ROI {current_weights[2]*100:.0f}%")
    print("="*60)
    
    # 2. Run AI Models
    impact_res = predict_transfer(player_name, target_team)
    if "error" in impact_res: return print(f"❌ {impact_res['error']}")
    elo_change = impact_res['predicted_impact']
    
    squad = get_target_squad(target_team)
    chem_score = 50.0 
    if not squad.empty:
        p_row = df_scout[df_scout['search_key'] == clean_txt(player_name)]
        if not p_row.empty:
            chem_map = get_squad_chemistry_map(p_row.iloc[0], squad)
            if not chem_map.empty: chem_score = chem_map['Chemistry_Score'].mean()

    mkt_value = get_player_market_value(player_name)

    # 3. Calculate TVI with CHOSEN weights
    final_tvi = calculate_tvi(chem_score, elo_change, mkt_value, current_weights)

    # 4. Output
    print(f"\n📊 RESULTS")
    print(f"   • Fit Score:   {chem_score:.1f}")
    print(f"   • Impact:      {elo_change:+.2f} Elo")
    print(f"   • Cost:        €{mkt_value:,.0f}")
    print("-" * 30)
    print(f"🏆 FINAL SCORE:  {final_tvi} / 100")
    print("="*60)

if __name__ == "__main__":
    p = input("Enter Player Name: ")
    t = input("Enter Target Team: ")
    run_analysis_interactive(p, t)
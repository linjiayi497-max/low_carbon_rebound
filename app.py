import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# ==========================================
# CONFIG & CACHING
# ==========================================
st.set_page_config(page_title="Carbon Policy Dashboard", layout="wide", page_icon="🌍")

@st.cache_data
def load_data():
    try:
        national = pd.read_csv("data/national_footprint.csv")
    except FileNotFoundError:
        national = pd.DataFrame({'country': ['World Average'], 'co2_per_capita': [4.5]})
    return national

@st.cache_resource
def load_models():
    try:
        with open("models/direct_rebound_model.pkl", "rb") as f:
            model_direct = pickle.load(f)
        with open("models/psych_rebound_model.pkl", "rb") as f:
            model_psych = pickle.load(f)
        with open("models/feature_names.pkl", "rb") as f:
            feature_names = pickle.load(f)
        return model_direct, model_psych, feature_names
    except FileNotFoundError:
        st.error("Models not found! Please run train_models.py first.")
        st.stop()

national_data = load_data()
model_direct, model_psych, feature_names = load_models()

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("🌍 Carbon Dashboard")
st.sidebar.markdown("""
Welcome to the  Carbon Footprint Prediction & Policy Simulation Dashboard.
""")
page = st.sidebar.radio("Navigate", 
    ["1. Footprint Calculator", 
     "2. Rebound Prediction", 
     "3. Recommendations", 
     "4. Policy Simulation"]
)

# Initialize Session State for user inputs
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = {
        'income_level': 'Medium (¥50k - ¥150k)',
        'q1': "C) Buy low-carbon if budget allows",
        'q2': "C) Follow rules willingly",
        'q3': "C) Agree in principle",
        'days_ev': 1,
        'trips_hsr': 2,
        'diet_type': 'Omnivore',
        'appliance_efficiency': 3,
        'days_waste_sorting': 3,
        'meals_reduce_disposable': 7,
        'base_footprint_tonnes': 6.0
    }

def get_normalized_profile(profile):
    income_val = 'Medium'
    if 'Low' in profile.get('income_level', ''): income_val = 'Low'
    elif 'High' in profile.get('income_level', ''): income_val = 'High'
    
    q_scores = {
        "A) Buy standard":0, "B) Hesitate but buy standard":10, "C) Buy low-carbon if budget allows":20, "D) Always buy low-carbon":33,
        "A) Ignore it":0, "B) Do the minimum to avoid fines":10, "C) Follow rules willingly":20, "D) Advocate for others to join":33,
        "A) Strongly disagree":0, "B) Disagree, but understand":10, "C) Agree in principle":20, "D) Strongly agree and already practice it":34
    }
    env_score = q_scores.get(profile.get('q1'),0) + q_scores.get(profile.get('q2'),0) + q_scores.get(profile.get('q3'),0)
    
    return {
        'income_level': income_val,
        'env_identity_score': env_score,
        'freq_ev': min(profile.get('days_ev', 0) / 7.0, 1.0),
        'freq_hsr': min(profile.get('trips_hsr', 0) / 20.0, 1.0),
        'diet_type': profile.get('diet_type', 'Omnivore'),
        'appliance_efficiency': profile.get('appliance_efficiency', 3),
        'freq_waste_sorting': min(profile.get('days_waste_sorting', 0) / 7.0, 1.0),
        'freq_reduce_disposable': min(profile.get('meals_reduce_disposable', 0) / 21.0, 1.0),
        'base_footprint_tonnes': profile.get('base_footprint_tonnes', 6.0)
    }

# ==========================================
# HELPER: PREPARE USER INPUT FOR MODEL
# ==========================================
def prepare_features(profile):
    norm_profile = get_normalized_profile(profile)
    income_map = {'Low': 1, 'Medium': 2, 'High': 3}
    df = pd.DataFrame([norm_profile])
    
    df['income_numeric'] = df['income_level'].map(income_map)
    df = pd.get_dummies(df, columns=['diet_type'])
    
    # Ensure all columns from training are present
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
            
    # Also handle boolean to int conversion just in case
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)
            
    return df[feature_names]

# ==========================================
# PAGE 1: FOOTPRINT CALCULATOR
# ==========================================
if page == "1. Footprint Calculator":
    st.title("🌱 Personal Carbon Footprint Calculator")
    st.markdown("Input your consumption attributes below. The footprint calculation dynamically responds to your actions.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Demographics & Values")
        income_opts = ["Low (< ¥50k)", "Medium (¥50k - ¥150k)", "High (> ¥150k)"]
        st.session_state.user_profile['income_level'] = st.selectbox("Income Level", income_opts, index=income_opts.index(st.session_state.user_profile.get('income_level', "Medium (¥50k - ¥150k)")))
        
        st.markdown("**Environmental Acceptance Questionnaire**")
        q1_opts = ["A) Buy standard", "B) Hesitate but buy standard", "C) Buy low-carbon if budget allows", "D) Always buy low-carbon"]
        st.session_state.user_profile['q1'] = st.radio("1. If a low-carbon product costs 10% more, what do you do?", q1_opts, index=q1_opts.index(st.session_state.user_profile.get('q1', "C) Buy low-carbon if budget allows")))
        
        q2_opts = ["A) Ignore it", "B) Do the minimum to avoid fines", "C) Follow rules willingly", "D) Advocate for others to join"]
        st.session_state.user_profile['q2'] = st.radio("2. When local government introduces strict recycling rules:", q2_opts, index=q2_opts.index(st.session_state.user_profile.get('q2', "C) Follow rules willingly")))
        
        q3_opts = ["A) Strongly disagree", "B) Disagree, but understand", "C) Agree in principle", "D) Strongly agree and already practice it"]
        st.session_state.user_profile['q3'] = st.radio("3. How do you feel about restricting personal flights?", q3_opts, index=q3_opts.index(st.session_state.user_profile.get('q3', "C) Agree in principle")))

        st.subheader("Diet & Lifestyle")
        diet_opts = ["Vegetarian", "Vegan", "Omnivore"]
        st.session_state.user_profile['diet_type'] = st.selectbox("Diet Type", diet_opts, index=diet_opts.index(st.session_state.user_profile.get('diet_type', "Omnivore")) if st.session_state.user_profile.get('diet_type') in diet_opts else 2) 
        st.session_state.user_profile['appliance_efficiency'] = st.slider("Appliance Energy Efficiency Rating", 1, 5, st.session_state.user_profile.get('appliance_efficiency', 3))
        
    with col2:
        st.subheader("Transport Frequency")
        st.session_state.user_profile['days_ev'] = st.slider("Days per week using an EV", 0, 7, st.session_state.user_profile.get('days_ev', 1))
        st.session_state.user_profile['trips_hsr'] = st.slider("High Speed Rail trips per year", 0, 20, st.session_state.user_profile.get('trips_hsr', 2))
        
        st.subheader("Circular Economy")
        st.session_state.user_profile['days_waste_sorting'] = st.slider("Days per week sorting waste", 0, 7, st.session_state.user_profile.get('days_waste_sorting', 3))
        st.session_state.user_profile['meals_reduce_disposable'] = st.slider("Meals per week avoiding disposable utensils", 0, 21, st.session_state.user_profile.get('meals_reduce_disposable', 7))
        
    # Baseline calculations
    norm = get_normalized_profile(st.session_state.user_profile)
    base_calc = 10.0 # Base footprint
    if norm['income_level'] == 'Low': base_calc *= 0.6
    if norm['income_level'] == 'High': base_calc *= 1.5
    
    # Reductions
    reductions = (
        (norm['freq_ev'] * 0.8) +
        (norm['freq_hsr'] * 0.5) +
        ((norm['appliance_efficiency'] - 1) * 0.2) +
        (norm['freq_waste_sorting'] * 0.3) +
        (norm['freq_reduce_disposable'] * 0.2)
    )
    if norm['diet_type'] == 'Vegan': reductions += 1.0
    if norm['diet_type'] == 'Vegetarian': reductions += 0.6
    
    final_footprint = max(2.0, base_calc - reductions)
    st.session_state.user_profile['base_footprint_tonnes'] = final_footprint
    
    st.divider()
    st.subheader(f"Your Estimated Carbon Footprint: {final_footprint:.2f} tCO2")
    
    # Comparison chart using Plotly
    avg_country = national_data['country'].iloc[0] if len(national_data) > 0 else 'Global Avg'
    avg_value = national_data['co2_per_capita'].iloc[0] if len(national_data) > 0 else 4.5
    
    df_compare = pd.DataFrame({
        'Entity': ['You', f'Average ({avg_country})'],
        'Emissions (tCO2)': [final_footprint, avg_value]
    })
    
    fig = px.bar(df_compare, x='Entity', y='Emissions (tCO2)', color='Entity',
                 color_discrete_sequence=['#2ecc71', '#95a5a6'])
    fig.update_layout(title="How do you compare?", template="plotly_white")
    st.plotly_chart(fig)

# ==========================================
# PAGE 2: REBOUND PREDICTION (SHAP)
# ==========================================
elif page == "2. Rebound Prediction":
    st.title("🔍 Rebound Propensity Prediction")
    st.markdown("""
    Based on the working paper findings, efficiency gains often lead to **Direct Rebound** (using the efficient product more) 
    and **Psychological Rebound** (doing one good deed makes you slack on others). High environmental identity can trigger opposite masking effects.
    """)
    
    X_input = prepare_features(st.session_state.user_profile)
    
    direct_pred = model_direct.predict(X_input)[0]
    psych_pred = model_psych.predict(X_input)[0]
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Direct Rebound Propensity", value=f"{direct_pred:.2f}", delta="Higher risk of use-rebound" if direct_pred > 0.5 else "Low risk", delta_color="inverse")
    with col2:
        st.metric(label="Psychological Rebound Propensity", value=f"{psych_pred:.2f}", delta="Risk of moral licensing" if psych_pred > 0.5 else "Consistent habits", delta_color="inverse")

    norm_profile = get_normalized_profile(st.session_state.user_profile)
    if norm_profile['env_identity_score'] > 75 and psych_pred > 0.4:
        st.warning("⚠️ **Masking Effect Identified!** Even though you have strong environmental values, your adoption of certain behaviors (like waste sorting) is creating a psychological rebound effect (moral licensing).")
    
    st.divider()
    st.subheader("Feature Impact (SHAP Waterfall)")
    st.markdown("See exactly which of your attributes are driving these rebound risks.")
    
    model_choice = st.radio("Select Model Explanation:", ["Direct Rebound", "Psychological Rebound"])
    selected_model = model_direct if model_choice == "Direct Rebound" else model_psych
    
    # SHAP Explainer
    # TreeExplainer is fast for xgboost
    explainer = shap.TreeExplainer(selected_model)
    shap_values = explainer(X_input)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig, bbox_inches='tight')
    plt.clf()

# ==========================================
# PAGE 3: RECOMMENDATIONS
# ==========================================
elif page == "3. Recommendations":
    st.title("🎯 Personalised Recommendations")
    st.markdown("We rank the six key low-carbon behaviours based on your profile's net emissions reduction (accounting for your specific rebound propensity) and Non-Pecuniary Costs (NPC).")
    
    X_input = prepare_features(st.session_state.user_profile)
    direct_rebound = model_direct.predict(X_input)[0]
    psych_rebound = model_psych.predict(X_input)[0]
    
    # Hypothetical base reduction (tCO2) and NPC (0-10, lower is easier)
    behaviors = [
        {"name": "EV Adoption", "base_reduction": 1.5, "npc": 8},
        {"name": "High-Speed Rail", "base_reduction": 0.8, "npc": 5},
        {"name": "Energy-efficient Appliances", "base_reduction": 0.6, "npc": 3},
        {"name": "Waste Sorting", "base_reduction": 0.3, "npc": 4},
        {"name": "Reduce Disposables", "base_reduction": 0.2, "npc": 2},
        {"name": "Protein Substitution (Diet)", "base_reduction": 1.2, "npc": 9}
    ]
    
    # Adjust reductions based on rebound
    results = []
    for b in behaviors:
        penalty = 0
        # Direct rebound mostly hits technology efficiency
        if b['name'] in ["EV Adoption", "Energy-efficient Appliances"]:
            penalty += direct_rebound * b['base_reduction']
        # Psych rebound hits everyday behavioral acts heavily
        if b['name'] in ["Waste Sorting", "Reduce Disposables", "Protein Substitution (Diet)"]:
            penalty += psych_rebound * b['base_reduction'] * 0.5
            
        net_red = max(0, b['base_reduction'] - penalty)
        
        # Priority Score: Net reduction relative to the difficulty (NPC)
        score = net_red / (b['npc'] * 0.5)
        
        results.append({
            "Behavior": b['name'],
            "Base Potential": b['base_reduction'],
            "Net Potential": net_red,
            "Rebound Penalty": penalty,
            "Difficulty (NPC)": b['npc'],
            "Priority Score": score
        })
        
    df_rec = pd.DataFrame(results).sort_values("Priority Score", ascending=False)
    
    st.dataframe(
        df_rec.style.background_gradient(subset=["Net Potential", "Priority Score"], cmap="Greens")
                    .background_gradient(subset=["Rebound Penalty", "Difficulty (NPC)"], cmap="Reds")
    )
    
    top_habit = df_rec.iloc[0]['Behavior']
    st.success(f"**Top Recommendation:** Focus on **{top_habit}**. It offers the best balance of emissions reduction for your profile while avoiding major rebound traps or extreme behavioral costs.")

# ==========================================
# PAGE 4: POLICY SIMULATION
# ==========================================
elif page == "4. Policy Simulation":
    st.title("📊 Policy Simulation (Monte Carlo)")
    st.markdown("We simulate 1,000 citizens varying direct/psychological rebound rates under 3 policy scenarios: Baseline, Carbon Tax, and Tax + Nudges.")
    
    np.random.seed(42)
    n_sims = 1000
    
    base_reductions = np.random.normal(5.0, 1.0, n_sims)
    
    # Scenario 1: Baseline (High rebound variability based on paper)
    s1_rebound = np.random.uniform(0.3, 0.7, n_sims)
    s1_net = base_reductions * (1 - s1_rebound)
    
    # Scenario 2: Carbon Tax Only (Reduces direct rebound slightly, increases NPC)
    s2_rebound = np.random.uniform(0.2, 0.6, n_sims)
    s2_net = base_reductions * 1.2 * (1 - s2_rebound)
    
    # Scenario 3: Carbon Tax + Behavioral Nudge (Targets psych rebound & masking effects)
    s3_rebound = np.random.uniform(0.05, 0.3, n_sims)
    s3_net = base_reductions * 1.3 * (1 - s3_rebound)
    
    sim_df = pd.DataFrame({
        'Net Reductions (tCO2)': np.concatenate([s1_net, s2_net, s3_net]),
        'Scenario': ['1. Baseline'] * n_sims + ['2. Carbon Tax'] * n_sims + ['3. Tax + Nudges'] * n_sims
    })
    
    fig_box = px.box(sim_df, x='Scenario', y='Net Reductions (tCO2)', color='Scenario', 
                     title="Distribution of Emissions Reductions by Policy",
                     template="plotly_white")
    st.plotly_chart(fig_box)
    
    fig_hist = px.histogram(sim_df, x='Net Reductions (tCO2)', color='Scenario',
                            barmode='overlay', marginal='violin',
                            title="Emissions Reduction Density Curve",
                            template="plotly_white")
    fig_hist.update_traces(opacity=0.6)
    st.plotly_chart(fig_hist)
    
    st.info("💡 **Academic Insight**: Providing a carbon tax alone suffers from high variance due to direct rebound. Adding behavioral interventions (nudges) significantly compresses the variance and shifts the distribution to higher net reductions.")

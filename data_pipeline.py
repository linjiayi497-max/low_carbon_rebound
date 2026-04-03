import pandas as pd
import numpy as np
import requests
import io
import os

def download_owid_co2_data(output_path="data/national_footprint.csv"):
    """
    Downloads the real CO2 dataset from Our World in Data (often used in Kaggle).
    We filter it for the most recent year with complete data to use as our baseline.
    """
    url = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"
    print(f"Downloading real CO2 data from {url}...")
    
    # Check if file exists to avoid re-downloading during testing
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        print("File already exists. Skipping download.")
        return pd.read_csv(output_path)
    
    response = requests.get(url)
    if response.status_code == 200:
        df = pd.read_csv(io.StringIO(response.text))
        
        # Filter for the most recent year per country that has co2_per_capita data
        # We only want countries (iso_code is not null, and length is 3)
        countries_df = df[df['iso_code'].notnull() & (df['iso_code'].str.len() == 3)]
        latest_data = countries_df.dropna(subset=['co2_per_capita']).sort_values('year').groupby('country').tail(1)
        
        # Select relevant columns
        cols_to_keep = ['country', 'year', 'iso_code', 'population', 'gdp', 'co2', 'co2_per_capita']
        latest_data = latest_data[cols_to_keep]
        
        latest_data.to_csv(output_path, index=False)
        print(f"Saved national footprint data to {output_path}")
        return latest_data
    else:
        print("Failed to download data.")
        return None

def generate_synthetic_user_data(n_samples=5000, output_path="data/synthetic_users.csv"):
    """
    Generate synthetic data reflecting the paper's findings on:
    - Direct rebound effect
    - Psychological rebound
    - Masking effect (high vs low env identity)
    - Non-pecuniary costs (NPC)
    """
    np.random.seed(42)
    
    # 1. Base Profiles
    income_levels = np.random.choice(['Low', 'Medium', 'High'], size=n_samples, p=[0.3, 0.5, 0.2])
    # Environmental identity (0-100)
    env_identity = np.random.normal(loc=60, scale=15, size=n_samples)
    env_identity = np.clip(env_identity, 0, 100)
    
    # 2. Consumption Behaviours
    # Higher income -> higher base consumption
    # Higher env identity -> more likely to adopt low carbon behavior
    
    # EV frequency (0-1)
    ev_freq = np.random.uniform(0, 1, size=n_samples)
    ev_freq[income_levels == 'High'] += 0.2
    ev_freq[env_identity > 70] += 0.1
    ev_freq = np.clip(ev_freq, 0, 1)
    
    # HSR frequency (0-1)
    hsr_freq = np.random.uniform(0, 1, size=n_samples) + (env_identity / 200)
    hsr_freq = np.clip(hsr_freq, 0, 1)
    
    # Diet type
    diet_probs = np.zeros((n_samples, 3)) # Omnivore, Vegetarian, Vegan
    diet_probs[:, 0] = 1 - (env_identity / 150)
    diet_probs[:, 1] = (env_identity / 200)
    diet_probs[:, 2] = 1 - diet_probs[:, 0] - diet_probs[:, 1]
    
    # normalise
    diet_probs = diet_probs / diet_probs.sum(axis=1, keepdims=True)
    
    diets = [np.random.choice(['Omnivore', 'Vegetarian', 'Vegan'], p=diet_probs[i]) for i in range(n_samples)]
    
    # Efficiency appliance level (1-5)
    appliance_level = np.random.randint(1, 6, size=n_samples)
    appliance_level[income_levels == 'High'] = np.clip(appliance_level[income_levels == 'High'] + 1, 1, 5)
    
    # Waste sorting frequency (0-1)
    waste_sorting = np.clip(np.random.normal(env_identity / 100, 0.2, size=n_samples), 0, 1)
    
    # Reduce disposable utensils (0-1)
    reduce_disposable = np.clip(np.random.normal(env_identity / 120, 0.3, size=n_samples), 0, 1)

    df = pd.DataFrame({
        'income_level': income_levels,
        'env_identity_score': env_identity,
        'freq_ev': ev_freq,
        'freq_hsr': hsr_freq,
        'diet_type': diets,
        'appliance_efficiency': appliance_level,
        'freq_waste_sorting': waste_sorting,
        'freq_reduce_disposable': reduce_disposable
    })

    # Encode categorical to numeric for modelling logic
    df['income_numeric'] = df['income_level'].map({'Low': 1, 'Medium': 2, 'High': 3})
    
    # --- Formulate Rebound Propensities ---
    # According to paper:
    # 1. Direct rebound effect: Efficiency gains (e.g. EV, appliance) lead to more use.
    # Higher income -> higher direct rebound.
    # Masking effect: High env identity -> opposite rebound direction (lower direct rebound).
    
    base_direct = 0.2 + (df['income_numeric'] * 0.15)
    # Masking effect: if env_identity is high (>70), direct rebound drops significantly
    masking_impact = np.where(df['env_identity_score'] > 70, -0.3, 0.1)
    direct_rebound_score = base_direct + masking_impact + (df['freq_ev'] * 0.1) + (df['appliance_efficiency'] * 0.05) + np.random.normal(0, 0.05, n_samples)
    
    # 2. Psychological Rebound: 
    # Adopting one behavior (waste sorting) reduces motivation for others.
    # Interestingly, high env identity might have higher psychological rebound if they feel they've done their part.
    base_psych = 0.3 + (df['freq_waste_sorting'] * 0.2) + (df['freq_reduce_disposable'] * 0.1)
    # If they are doing extreme behaviors like Vegan, psychological rebound is high
    diet_impact = np.where(df['diet_type'] == 'Vegan', 0.2, np.where(df['diet_type'] == 'Vegetarian', 0.1, 0))
    psych_rebound_score = base_psych + diet_impact + (df['env_identity_score'] / 500) + np.random.normal(0, 0.05, n_samples)

    df['direct_rebound_score'] = np.clip(direct_rebound_score, 0, 1)
    df['psych_rebound_score'] = np.clip(psych_rebound_score, 0, 1)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {n_samples} synthetic user records at {output_path}")
    return df

if __name__ == "__main__":
    download_owid_co2_data()
    generate_synthetic_user_data()

import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def train_and_save_models(data_path="data/synthetic_users.csv", output_dir="models"):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading data from {data_path}...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print("Data file not found. Please run data_pipeline.py first.")
        return

    # Features to use
    feature_cols = [
        'income_numeric', 'env_identity_score', 'freq_ev', 'freq_hsr', 
        'appliance_efficiency', 'freq_waste_sorting', 'freq_reduce_disposable'
    ]
    
    # One-hot encode diet type for XGBoost
    df = pd.get_dummies(df, columns=['diet_type'], drop_first=False)
    
    # Get all dummy columns added (diet_type_Omnivore, diet_type_Vegan, diet_type_Vegetarian)
    diet_cols = [c for c in df.columns if c.startswith('diet_type_')]
    final_features = feature_cols + diet_cols

    X = df[final_features]
    y_direct = df['direct_rebound_score']
    y_psych = df['psych_rebound_score']

    # Train / Test split
    X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X, y_direct, test_size=0.2, random_state=42)
    X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X, y_psych, test_size=0.2, random_state=42)

    print("\n--- Training Direct Rebound Model ---")
    model_direct = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model_direct.fit(X_train_d, y_train_d)
    
    preds_d = model_direct.predict(X_test_d)
    print(f"Direct Rebound Model MSE: {mean_squared_error(y_test_d, preds_d):.4f}")
    print(f"Direct Rebound Model R2: {r2_score(y_test_d, preds_d):.4f}")
    
    with open(os.path.join(output_dir, 'direct_rebound_model.pkl'), 'wb') as f:
        pickle.dump(model_direct, f)

    print("\n--- Training Psychological Rebound Model ---")
    model_psych = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model_psych.fit(X_train_p, y_train_p)
    
    preds_p = model_psych.predict(X_test_p)
    print(f"Psychological Rebound Model MSE: {mean_squared_error(y_test_p, preds_p):.4f}")
    print(f"Psychological Rebound Model R2: {r2_score(y_test_p, preds_p):.4f}")

    with open(os.path.join(output_dir, 'psych_rebound_model.pkl'), 'wb') as f:
        pickle.dump(model_psych, f)

    # Save the feature names used so Streamlit ensures consistent input
    with open(os.path.join(output_dir, 'feature_names.pkl'), 'wb') as f:
        pickle.dump(final_features, f)

    print(f"\nModels and feature mapping saved to {output_dir}/")

if __name__ == "__main__":
    train_and_save_models()

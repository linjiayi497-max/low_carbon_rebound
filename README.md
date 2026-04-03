# Carbon Footprint Prediction & Policy Simulation Dashboard

An interactive Streamlit web application designed to predict carbon emissions, evaluate rebound propensities (Direct and Psychological), and simulate the effects of behavioral policies. 

This project operationalises findings from a working paper on demand-side low-carbon behaviour, tracking non-pecuniary costs (NPC) and masking effects among consumers.

## Features

1. **Personal Carbon Footprint Calculator**
   - Inputs for demographics and 6 key low-carbon behaviors: EV adoption, high-speed rail usage, energy-efficient appliances, waste sorting, reducing disposable utensils, and protein substitution (diet).
   - Real baseline data from the **Our World in Data** (Kaggle-equivalent CO2 footprints country data).

2. **Rebound Prediction**
   - Uses XGBoost models trained on synthetic behavioral data.
   - Outputs **Direct Rebound Propensity** and **Psychological Rebound Propensity**.
   - Includes full interpretability using **SHAP Waterfall charts** to identify driving factors in individual predictions.

3. **Personalised Recommendation Engine**
   - Evaluates behaviors against an individual's predicted rebound rates and absolute Non-Pecuniary Cost (difficulty of adoption).
   - Generates an actionable, ranked checklist of which lifestyle changes will strictly yield the highest *net* emission reductions.

4. **Policy Simulation (Monte Carlo)**
   - Simulates 1,000 citizens and visualizes different distributions of net emissions reduction using Plotly.
   - Compares: Baseline (No Intervention), Carbon Tax Only, and Carbon Tax + Nudges.

## Project Structure

- `app.py`: Main Streamlit application and UI logic.
- `data_pipeline.py`: Downloads external CO2 data and generates the synthetic survey respondent dataset.
- `train_models.py`: Trains the XGBoost regressors and saves valid `.pkl` files based on generated data.
- `data/`: Contains raw structure of `national_footprint.csv` and `synthetic_users.csv`.
- `models/`: Pickles for the trained ML models and feature mappers.

## How to Run Locally

1. **Install dependencies**
```bash
pip install -r requirements.txt
```

2. **Generate the Datasets** (First time only)
```bash
python data_pipeline.py
```

3. **Train the ML Models** (First time only)
```bash
python train_models.py
```

4. **Launch the Dashboard**
```bash
streamlit run app.py
```

## Academic Context Integration
- **Masking Effect**: Noticeable when testing High Environmental Identity profiles on Page 2 – these profiles show opposite, sometimes paradoxical rebound limits (e.g. 'moral licensing' via psychological rebound).
- **NPC Adjustment**: In Page 3, behaviors with high baseline reductions (like EVs) might be deprioritized for strict low-income groups due to NPC modeling.

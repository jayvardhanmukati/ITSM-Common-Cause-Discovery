# ITSM Common-Cause Discovery

## Overview
ITSM Common-Cause Discovery is a deep learning-based application designed to identify related IT service incidents and predict their potential common root causes.

The goal is to help IT support teams identify patterns across multiple incidents and investigate underlying problems instead of handling every incident separately.

## Key Features
- Incident analysis and classification
- Deep learning-based incident prediction using LSTM
- Common-cause discovery for related incidents
- Explainable AI using LIME
- Interactive Streamlit dashboard
- Incident grouping and visualization
- Model evaluation and performance analysis

## Technologies Used
- Python
- TensorFlow / Keras
- Streamlit
- Scikit-learn
- Pandas and NumPy
- LIME
- Plotly
- Joblib

## Project Structure
- `app_updated_v3.py` – Main Streamlit application
- `model.py` – Model implementation
- `model_new.py` – Deep learning model implementation
- `generate_data.py` – Dataset generation
- `common_cause.py` – Common-cause analysis
- `lstm_model.keras` – Trained LSTM model
- `incidents.csv` – Incident dataset
- `itsm_final_dataset.csv` – Processed dataset
- `confusion_matrix.png` – Model evaluation visualization
- `training_history.png` – Training performance visualization

## How to Run

1. Clone this repository:
   `git clone https://github.com/jayvardhanmukati/ITSM-Common-Cause-Discovery.git`

2. Navigate to the project folder.

3. Install the required dependencies:
   `pip install -r requirements.txt`

4. Run the application:
   `streamlit run app_updated_v3.py`

## Objective
To explore how deep learning and explainable AI can improve IT service management by identifying common patterns across incidents and supporting root-cause investigation.

---
Developed as an academic Deep Learning project.

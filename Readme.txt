# Disease Prediction using Symptoms

This project predicts diseases based on symptoms using machine learning (Random Forest + KMeans grouping).  
It includes Exploratory Data Analysis (EDA) and a Gradio web interface for easy interaction.

## 🚀 Features
- Data cleaning and preprocessing
- EDA with visualizations (distribution, correlations, top symptoms)
- KMeans clustering to group diseases
- Random Forest classification
- Interactive Gradio app for predictions

## 📂 Project Structure
- `data/diseases.csv` → Dataset  
- `app.py` → train and run the model

## ▶️ Run
```bash
pip install -r requirements.txt
python eda.py       # For analysis & plots
python model.py     # Train model
python app.py       # Launch Gradio interface

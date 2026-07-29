# 📊 Employee Salary Prediction

A professional end-to-end machine learning project that predicts whether an employee earns more than $50K based on demographic, educational, and work-related features.

## 🚀 Live Demo

Open the deployed app here:
https://employeesalarypredictionchiragjain-mrftbnjidtux9q2py8wzde.streamlit.app/

## 🧠 Project Overview

This project showcases a complete ML workflow for salary classification using the UCI Adult dataset. It includes:

- data preprocessing and encoding
- model training and evaluation
- interactive Streamlit predictions
- SHAP-based explainability
- model performance visualization
- PDF report generation

## 🛠️ Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Plotly
- SHAP
- FPDF
- Pillow

Optional for advanced explainability:
- SHAP (install with `pip install shap` if you want the interactive SHAP summary view)

## 📦 Project Structure

- `app.py` – interactive Streamlit dashboard
- `adult.csv` – dataset used for training and inference
- `best_model.pkl` – trained prediction model
- `requirements.txt` – dependencies
- `assets/architecture.svg` – architecture diagram

## 🧪 Model Performance

The app provides:

- accuracy and classification report
- confusion matrix
- ROC curve and AUC
- feature importance and SHAP explanation

## ▶️ How to Run

1. Clone the repository
2. Install dependencies:
   `pip install -r requirements.txt`
3. Start the app:
   `streamlit run app.py`

## 🏗️ Architecture

![Architecture Diagram](assets/architecture.svg)

## 🔮 Future Improvements

- add hyperparameter tuning
- compare more advanced models like XGBoost and CatBoost
- add user authentication and saved reports

## 👨‍💻 Author

Chirag Jain
B.Tech, NIT Kurukshetra
Interested in Machine Learning and Data Science

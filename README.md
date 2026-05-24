# 🩺 AI Diabetes Prediction System

A machine-learning powered diabetes risk assessment platform with user
authentication, prediction history, an AI chatbot, and PDF report generation.

**Final Year Project**

---

## ✨ Features

| Module | What it does |
|--------|--------------|
| 🔐 **Authentication** | Sign up / log in (PBKDF2-hashed passwords, SQLite) |
| 🔍 **Prediction Engine** | Best-of-5 ML models (Logistic Regression, Random Forest, Gradient Boosting, SVM, XGBoost) |
| 🧠 **Explainability** | Feature contribution chart — "why this prediction" |
| 📊 **Performance Dashboard** | Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix, ROC curve, model comparison |
| 📜 **Prediction History** | Per-user history saved to SQLite, exportable as CSV |
| 📄 **PDF Reports** | Downloadable, hospital-style patient report |
| 🤖 **Health Assistant** | Gemini-powered chatbot, context-aware of your latest prediction |
| 🔬 **Dataset Explorer** | Interactive EDA — distributions, correlation heatmap |

---

## 🚀 Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure Gemini key (for the chatbot)
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# 3. (Optional) Retrain the model
python train_model.py

# 4. Run the app
streamlit run app.py
```

The app opens at **http://localhost:8501**.

---

## 📂 Project structure

```
diabetes_prediction_pro/
├── app.py                  # Main Streamlit app (multi-page)
├── train_model.py          # Trains 5 models, picks the best, saves metrics
├── database.py             # SQLite auth + prediction history
├── pdf_report.py           # ReportLab PDF generator
├── diabetes.csv            # Pima Indians Diabetes dataset
├── trained_model.sav       # Pickled best model
├── model_metadata.json     # Metrics, feature importance, ROC curve points
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🧪 Model performance (test set)

| Model | Accuracy | F1 | ROC-AUC |
|-------|----------|-----|---------|
| Gradient Boosting ⭐ | 75.97% | 0.648 | 0.826 |
| XGBoost | 75.32% | 0.635 | 0.822 |
| Random Forest | 74.68% | 0.606 | 0.813 |
| SVM (RBF) | 74.03% | 0.600 | 0.796 |
| Logistic Regression | 70.78% | 0.546 | 0.813 |

**Best model:** Gradient Boosting (selected by ROC-AUC).

---

## 📊 Dataset

**Pima Indians Diabetes Database** — 768 samples, 8 clinical features:
Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age.
Source: UCI Machine Learning Repository.

Biologically impossible zero values in Glucose, BloodPressure, SkinThickness,
Insulin, and BMI are imputed with the column median during preprocessing.

---

## 🛠️ Tech stack

- **UI:** Streamlit, Plotly
- **ML:** scikit-learn, XGBoost
- **AI:** Google Gemini API
- **DB:** SQLite (built-in)
- **PDF:** ReportLab
- **Auth:** PBKDF2-SHA256 (200 000 iterations)

---

## ⚠️ Disclaimer

This project is for **academic and research purposes only**. The predictions
are not a clinical diagnosis. Always consult a qualified physician for medical
advice.

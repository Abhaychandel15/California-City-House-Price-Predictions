# 🏠 California City House Price Prediction

This project focuses on predicting **median house prices** in California cities using **Machine Learning regression models**.  
It follows a complete **end-to-end data science workflow**, including data preprocessing, feature engineering, model training, evaluation, and model persistence.

---

## 📊 Project Overview

Housing price prediction is a classic regression problem in data science.  
In this project, we use the **California Housing Dataset** to build and compare multiple regression models and evaluate their performance using **cross-validation**.

---

## ⚙️ Technologies Used

- Python 🐍
- Pandas
- NumPy
- Scikit-learn
- Machine Learning Pipelines
- Pickle (Model Persistence)
- Git & GitHub

---

## 📁 Project Structure

California-City-House-Price-Predictions/
│
├── main.py # Main training and evaluation script
├── main_old.py # Older version of training logic
├── housing.csv # Original dataset
├── input.csv # Input data for prediction
├── output.csv # Generated predictions
├── model.pkl # Trained ML model (saved)
├── pipeline.pkl # Preprocessing pipeline (saved)
├── requirements.txt # Project dependencies
├── .gitignore # Ignored files/folders
└── README.md # Project documentation

---

## 🔄 Machine Learning Workflow

1. **Data Loading**
   - Load housing dataset using Pandas

2. **Stratified Train-Test Split**
   - Based on `median_income` to ensure balanced distribution

3. **Data Preprocessing**
   - Numerical Features:
     - Missing value handling (Median Imputation)
     - Feature scaling (StandardScaler)
   - Categorical Features:
     - One-hot encoding (`ocean_proximity`)

4. **Pipeline Creation**
   - ColumnTransformer + Pipelines for clean preprocessing

5. **Model Training**
   - Linear Regression
   - Decision Tree Regressor
   - Random Forest Regressor

6. **Model Evaluation**
   - 10-fold Cross Validation
   - Metric: **Root Mean Squared Error (RMSE)**

7. **Model Saving**
   - Trained model saved using `pickle`
   - Pipeline saved to avoid retraining

---

## 📈 Models Used

| Model | Purpose |
|------|--------|
| Linear Regression | Baseline model |
| Decision Tree Regressor | Captures non-linear patterns |
| Random Forest Regressor | Ensemble model for better accuracy |

---

## 💾 Why Pickle Files?

- Prevents retraining the model every time
- Saves preprocessing + trained model
- Useful for deployment and real-world usage

Example:
```python
import pickle

model = pickle.load(open("model.pkl", "rb"))
pipeline = pickle.load(open("pipeline.pkl", "rb"))

🚀 How to Run the Project

1️⃣ Install Dependencies
pip install -r requirements.txt

2️⃣ Run Training Script
python main.py

🎯 Key Learnings

End-to-end Machine Learning project implementation

Proper use of Pipelines and ColumnTransformer

Cross-validation for reliable model evaluation

Model persistence using Pickle

Clean GitHub project structure

👤 Author

Abhay Chandel
Aspiring Data Scientist & Machine Learning Enthusiast

🔗 GitHub: https://github.com/Abhaychandel15

🔗 LinkedIn: https://www.linkedin.com/in/abhay-chandel-495b722a3/


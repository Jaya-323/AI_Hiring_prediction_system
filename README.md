# 💼 AI Hiring Prediction System

An AI-powered machine learning application that predicts whether a candidate is likely to be **Hired** or **Rejected** based on resume-related features such as experience, education, certifications, projects, salary expectation, and job role.

The project demonstrates the complete machine learning workflow—from data preprocessing and model training to deployment with Streamlit.

---

## 🚀 Features

- Predicts hiring decisions using Machine Learning
- Interactive web application built with Streamlit
- Professional dashboard interface
- Candidate summary after prediction
- Rule-based career recommendations
- Confidence score for predictions
- Model trained without relying on an AI Score feature for better real-world applicability

---

## 📊 Dataset

The dataset contains candidate information including:

- Experience (Years)
- Salary Expectation
- Projects Count
- Education
- Certifications
- Job Role
- Recruiter Decision (Target Variable)

---

## 🔄 Project Workflow

1. Data Loading
2. Exploratory Data Analysis (EDA)
3. Data Cleaning
4. Handling Missing Values
5. Feature Encoding
6. Feature Scaling
7. Train-Test Split
8. Model Training
9. Model Evaluation
10. Hyperparameter Tuning using GridSearchCV
11. Model Deployment with Streamlit

---

## 🤖 Machine Learning Models Used

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier
- XGBoost Classifier

---

## 📈 Model Performance

| Model | Accuracy |
|--------|---------:|
| Logistic Regression | 97.0% |
| Decision Tree | 97.0% |
| XGBoost | 97.0% |
| Gradient Boosting | 96.5% |
| Random Forest | 94.0% |

The deployed application uses the **XGBoost Classifier**, which provides strong predictive performance while using only candidate attributes (without an AI Score).

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Joblib
- Matplotlib
- Seaborn

---

## 📂 Project Structure

```
AI_Hiring_Prediction_System/
│
├── app.py
├── hiring_prediction_model.pkl
├── scaler.pkl
├── label_encoder.pkl
├── requirements.txt
├── README.md
├── notebook.ipynb
└── dataset.csv
```

---

## ▶️ Installation

Clone the repository

```bash
git clone https://github.com/your-username/AI_Hiring_Prediction_System.git
```

Move into the project folder

```bash
cd AI_Hiring_Prediction_System
```

Install the required libraries

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## 💻 Application

The Streamlit dashboard allows users to:

- Enter candidate details
- Predict hiring status
- View confidence score
- Receive profile improvement recommendations

---

## 📸 Application Preview
<img width="1917" height="885" alt="Screenshot 2026-06-30 191027" src="https://github.com/user-attachments/assets/23089b94-a20f-4896-b4e9-4e47ece1f92b" />



---

## 📌 Future Improvements

- Resume PDF upload
- Resume text extraction
- Skill extraction using NLP
- Explainable AI using SHAP
- Candidate ranking system
- Recruiter analytics dashboard
- Database integration

---

## 👨‍💻 Author

**Jaya Mishra**

If you found this project helpful, feel free to ⭐ the repository.

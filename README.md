# 🩺 Pima Diabetes Prediction – Machine Learning Project

![App Screenshot](Screenshot%202025-11-26%20082546.png)


This project uses a **Decision Tree Classifier** to predict whether a patient is likely to have **diabetes** based on medical data from the **Pima Indians Diabetes Dataset**.  
The project focuses purely on **Machine Learning**, data preprocessing, model training, evaluation, and prediction.

---

## 📊 Dataset Information

- **Dataset:** Pima Indians Diabetes Dataset  
- **Source:** UCI Machine Learning Repository  
- **URL:** [Download CSV](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv)  
- **Rows:** 768  
- **Columns:** 9  

### Columns
1. Pregnancies – Number of times pregnant  
2. Glucose – Plasma glucose concentration  
3. BloodPressure – Diastolic blood pressure  
4. SkinThickness – Triceps skinfold thickness  
5. Insulin – 2-hour serum insulin  
6. BMI – Body Mass Index  
7. DiabetesPedigreeFunction – Family diabetes function  
8. Age – Age in years  
9. Outcome – Target label (0 = Non-Diabetic, 1 = Diabetic)

---

## 🧬 Features and Target

- **Features (X):** Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age  
- **Target (y):** Outcome (0 = Non-Diabetic, 1 = Diabetic)

---

## 🛠️ Model Details

- **Algorithm:** Decision Tree Classifier  
- **Hyperparameter Tuning:** GridSearchCV  
- **Evaluation Metrics:** Accuracy, Confusion Matrix, Precision, Recall, F1-score, ROC-AUC  
- **Training Accuracy:** ~74%  
- **Testing Accuracy:** ~72%  

**Confusion Matrix Example:**
```
[[69 34]
 [11 48]]
```

---

## 📝 Steps to Run the Project

1. **Clone the repo**  
```bash
git clone http://github.com/subodhkryadav/pima-diabetes-prediction-ml
cd pima-diabetes-prediction-ml
```

2. **Install dependencies**  
```bash
pip install numpy pandas scikit-learn matplotlib seaborn joblib
```

3. **Load dataset and preprocess**  
- Replace zeros in certain columns with median values for Glucose, BloodPressure, SkinThickness, Insulin, BMI

4. **Train the model**  
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd

# Load and preprocess dataset
df = pd.read_csv("diabetes.csv", header=None)
# Split, fill missing values, etc.

# Train Decision Tree
dt = DecisionTreeClassifier()
grid_params = {
    'max_depth': [3, 4, 5, 6, 7, None],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 5, 10],
    'criterion': ['gini', 'entropy']
}
grid_search = GridSearchCV(dt, grid_params, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
```

5. **Evaluate the model**
```python
y_pred = grid_search.predict(X_test)
# Calculate accuracy, confusion matrix, classification report
```

6. **Save the model (optional)**
```python
import joblib
joblib.dump(grid_search.best_estimator_, "pima_diabetes_predictor.pkl")
```

---

## 📈 Evaluation Metrics

- **Accuracy:** 72% (Test set)  
- **Precision, Recall, F1-Score:**  
```
           0       0.86    0.67    0.75       103
           1       0.59    0.81    0.68        59
```

---

## 📂 File Structure

```
project/
│── diabetes.csv
│── pima_diabetes_predictor.pkl
│── Pima Indians Diabetes Prediction_DT.ipynb
│──app.py
│──templates/index.html
│──static/style.css
│── README.md
```

---

## 👤 Author

**Subodh Kumar Yadav**  

🔗 GitHub: [https://github.com/subodhkryadav](https://github.com/subodhkryadav)  
🔗 LinkedIn: [https://www.linkedin.com/in/subodh-kumar-yadav-522828293](https://www.linkedin.com/in/subodh-kumar-yadav-522828293)

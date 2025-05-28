# Depression Prediction using Machine Learning

This project explores various machine learning models to predict depression based on health and lifestyle indicators. The goal was to identify the best-performing model for reliable and accurate classification.

---

## 🧠 Objective

To build and evaluate multiple machine learning classifiers and select the best-performing model for predicting depression status.

---

## 📊 Methods Used

- Data Preprocessing & Cleaning
- Exploratory Data Analysis (EDA)
- **Feature Selection using Correlation Analysis**
- Class Imbalance Handling via **Random Oversampling**
- Model Training & Performance Evaluation
- Ensemble Modeling using StackingClassifier

---

## 🤖 Models Compared

- K-Nearest Neighbors (KNN)
- Naive Bayes
- Decision Tree
- Support Vector Machine (SVM)
- Random Forest
- AdaBoost
- Gradient Boosting
- XGBoost
- **Stacking Classifier** (Ensemble)

---

## 🏆 Best Model: StackingClassifier

- **Base Learners**:
  - KNN
  - Naive Bayes
  - Decision Tree
  - SVM
  - Random Forest
  - AdaBoost
  - Gradient Boosting
  - XGBoost
- **Final Estimator**: Logistic Regression
- **Train Accuracy**: 1.0  
- **Test Accuracy**: 0.9952

The stacking model combined the strengths of diverse learners and provided the highest performance, outperforming individual classifiers.

---

## 🔍 Future Improvements

- Plotting ROC Curve for performance visualization
- Hyperparameter tuning using `GridSearchCV`
- K-Fold Cross-validation for better generalization
- Deploying the final model using Flask or Streamlit

---

## 📁 Files

- `depression_prediction.ipynb`: Main Colab notebook
- `requirements.txt`: Python dependencies (optional)
- `data/`: Dataset file (if allowed)

---

## ⚙️ Installation

To install dependencies locally:

```bash
pip install -r requirements.txt

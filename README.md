## 🧠 Diabetes Prediction using Machine Learning Techniques

This project focuses on predicting the likelihood of diabetes in patients based on key medical attributes using supervised machine learning models. The goal is to assist in early diagnosis and improve preventive healthcare strategies through accurate predictions.

---

### 🔍 Problem Statement

Diabetes is one of the most prevalent chronic diseases globally. Early prediction can significantly improve patient outcomes. This project aims to develop a reliable prediction model using machine learning techniques on the Pima Indian Diabetes dataset.

---

### 🗂 Dataset

- **Source**: Pima Indian Diabetes Dataset (Kaggle / UCI)  
- **Features**:  
  - Pregnancies  
  - Glucose  
  - BloodPressure  
  - SkinThickness  
  - Insulin  
  - BMI  
  - DiabetesPedigreeFunction  
  - Age  
- **Target**: `Outcome`  
  - 0 → Non-diabetic  
  - 1 → Diabetic

---

### 🛠 Models Used

1. **K-Nearest Neighbors (KNN)**  
   - Simple, instance-based learning algorithm  
   - Moderate performance with limitations in scalability

2. **Decision Tree Classifier**  
   - Interpretable model using rule-based splitting  
   - Prone to overfitting on small datasets

3. **Naive Bayes Classifier**  
   - Probabilistic model effective for binary classification  
   - Delivered **highest accuracy** among baseline models

4. **Multi-Layer Perceptron (MLP)**  
   - Neural network model with one hidden layer  
   - Captured complex and non-linear patterns in data

5. **Fusion Model (Naive Bayes + MLP)**  
   - Combined predictions from Naive Bayes and MLP using soft voting  
   - ✅ **ROC-AUC Score**: `0.7538` 🎯  
   - Achieved **improved generalization** and better predictive performance

---

### 🔄 Fusion Strategy

A **soft voting ensemble** technique was used to combine the strengths of **Naive Bayes** (high baseline performance) and **MLP** (ability to capture non-linearity):

- Collected predicted probabilities using `predict_proba()` from both models  
- Averaged the probabilities for each class  
- Selected the final class label based on **maximum averaged probability**

This method worked effectively because:
- The models are **diverse** in their learning paradigms (probabilistic vs neural network)  
- It helped mitigate individual model bias and variance  
- ROC-AUC score improved, indicating **better discrimination power**

> 💡 *This fusion approach can be further enhanced using weighted averaging or stacked generalization (stacking).*  

---

### 📊 Evaluation Metrics

- Accuracy  
- Confusion Matrix  
- Precision, Recall, F1-score  
- ROC-AUC Score  
- Visualization of performance metrics and comparisons

---

### ✅ Highlights

- Performed feature scaling using StandardScaler  
- Evaluated and compared multiple models (KNN, Decision Tree, Naive Bayes, MLP)  
- Fusion approach implemented to enhance prediction robustness  
- Clean visualization of results for better interpretability  
- Modular and extensible code structure for future improvements

---

### 📌 Future Scope

- Hyperparameter tuning via GridSearch or RandomizedSearch  
- Explore more robust ensemble methods (e.g., stacking, bagging)  
- Deploy the model via Flask or Streamlit for real-time predictions  
- Use SHAP/feature importance analysis for model explainability

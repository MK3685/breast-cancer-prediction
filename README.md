# 🩺 Breast Cancer Diagnosis - Machine Learning Model  

![Cancer Cells](https://images.newscientist.com/wp-content/uploads/2019/06/06165424/c0462719-cervical_cancer_cell_sem-spl.jpg?width=837)  

> 🚀 **An AI-powered system for early detection of breast cancer using machine learning**  
> 🌍 **Live Demo:** [Try the Web App Here](https://breast-cancer-prediction-healthcare.streamlit.app/)  

---

## 📖 Project Overview  
Breast cancer is one of the leading causes of cancer-related deaths worldwide. **Early diagnosis** plays a critical role in **improving survival rates**. This project uses **machine learning models** to classify **benign (non-cancerous) and malignant (cancerous) tumors** based on medical data.  

### 🏆 **Key Features**  
✅ Uses the **Breast Cancer Wisconsin Dataset** 📊  
✅ **Preprocessing & Feature Engineering** (Scaling, SMOTE for class balance)  
✅ **Model Training & Selection** (Logistic Regression, Random Forest, SVM, XGBoost)  
✅ **Web App Deployment using Streamlit**  

---

## 📊 Dataset Information  
📌 **Source:** [Kaggle - Breast Cancer Wisconsin Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)  
📌 **Description:** The dataset contains numerical features extracted from **cell nuclei in breast cancer biopsy images**.  

### 📜 **Features (30 Total)**
Each tumor is characterized by measurements like:  
- `radius_mean` – Mean size of the tumor  
- `texture_mean` – Variability in tumor texture  
- `concavity_worst` – Severity of concave portions of tumor  

📌 **Target Variable:**  
- `M` (**Malignant**) → **1**  
- `B` (**Benign**) → **0**  


---

## 🛠️ Technologies & Tools  
🔹 **Python** (NumPy, Pandas, Matplotlib, Seaborn)  
🔹 **scikit-learn, XGBoost, imbalanced-learn**  
🔹 **Streamlit for Web Deployment**  
🔹 **Joblib for Model Persistence**  

## 🧪 Data Preprocessing & Model Training  

### 1️⃣ **Data Preprocessing** 🧼  
Before training the models, we preprocess the data to improve model performance:  

✔ **Drop unnecessary columns** – The `id` column is removed as it's irrelevant.  
✔ **Label Encoding** – The `diagnosis` column is converted to numerical format (`M=1`, `B=0`).  
✔ **Feature Scaling** – Standardized the dataset using `StandardScaler` for better model performance.  
✔ **Handling Imbalanced Data** – Used **SMOTE (Synthetic Minority Over-sampling Technique)** to balance classes, preventing biased predictions.  

### 2️⃣ **Model Training** 🎯  
We tested **4 machine learning models** using a **5-fold cross-validation (ROC-AUC Score):**  

| Model                  | ROC-AUC Score 🎯 |
|------------------------|-----------------|
| **Logistic Regression** (🔥 Best) | **1.000** ✅ |
| Random Forest         | 0.982 |
| XGBoost              | 0.990 |
| SVM                  | 0.973 |

✔ **Why Logistic Regression?**  
- It provided a **perfect ROC-AUC Score (1.000)** on the test set.  
- It's **lightweight, interpretable, and highly effective** for binary classification.  

---

## 📊 Model Evaluation  
### 📌 **Final Metrics (Test Set Performance)**
| Metric     | Benign | Malignant |
|------------|--------|-----------|
| **Precision**  | 1.00   | 1.00      |
| **Recall**     | 1.00   | 1.00      |
| **F1-score**   | 1.00   | 1.00      |

📌 **Final ROC-AUC Score:** 🎯 **1.000** (Perfect!)  

✔ **Confusion Matrix & ROC Curve**  
We visualized the model's **true vs. predicted classifications** and its **ability to distinguish between cancerous and non-cancerous tumors.**  

---

## 🚀 Installation & Usage  
### 1️⃣ Install Dependencies  
```bash
pip install -r requirements.txt
git clone https://github.com/MK3685/breast-cancer-prediction.git


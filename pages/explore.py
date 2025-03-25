# pages/explore.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.preprocessing import LabelEncoder

def main():
    st.title("🔍 Data Exploration & Model Analysis")
    
    # Load data
    df = pd.read_excel("cancer.xlsx").drop(columns=['id'])
    le = LabelEncoder()
    df['diagnosis'] = le.fit_transform(df['diagnosis'])
    
    # Section 1: Data Exploration
    st.header("Data Exploration")
        # Show processed data
    if st.checkbox("Show Processed Data"):
        st.dataframe(df)
   
     # Class distribution
    st.subheader("Class Distribution")
    fig, ax = plt.subplots()
    df['diagnosis'].value_counts().plot(kind='bar', ax=ax, color=['skyblue', 'salmon'])
    ax.set_title("Benign vs Malignant Cases")
    ax.set_xticklabels(['Benign (0)', 'Malignant (1)'], rotation=0)
    st.pyplot(fig)
    
    
    # Feature distributions
    st.subheader("Feature Distributions")
    selected_feature = st.selectbox("Select Feature", df.columns[1:])
    fig, ax = plt.subplots()
    sns.histplot(df[selected_feature], kde=True, ax=ax)
    st.pyplot(fig)
    
    # Correlation matrix
    st.subheader("Feature Correlation Matrix")
    numeric_df = df.select_dtypes(include=['number'])
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # Section 2: Model Evaluation
    st.header("🎯 Model Performance Analysis")
    
    try:
        # Load model and test data
        model_data = joblib.load('breast_cancer_model.pkl')
        feature_names = joblib.load('feature_names.pkl')
        X_test = joblib.load('X_test.pkl')
        y_test = joblib.load('y_test.pkl')
        
        # Model metrics
        st.subheader("Model Evaluation Metrics")
        
        # Create two columns for visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Confusion Matrix
            st.write("### Confusion Matrix")
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_estimator(
                model_data,
                X_test[feature_names],
                y_test,
                display_labels=['Benign', 'Malignant'],
                cmap='Blues',
                ax=ax
            )
            st.pyplot(fig)
        
        with col2:
            # ROC Curve
            st.write("### ROC Curve")
            fig, ax = plt.subplots()
            RocCurveDisplay.from_estimator(
                model_data,
                X_test[feature_names],
                y_test,
                ax=ax
            )
            st.pyplot(fig)
        
        # Feature Importance
        if hasattr(model_data.named_steps['model'], 'feature_importances_'):
            st.subheader("Feature Importance")
            importances = model_data.named_steps['model'].feature_importances_
            feat_importances = pd.Series(importances, index=feature_names)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            feat_importances.nlargest(10).sort_values().plot(kind='barh', ax=ax)
            ax.set_title("Top 10 Important Features")
            ax.set_xlabel("Importance Score")
            st.pyplot(fig)
            
    except FileNotFoundError:
        st.warning("Model artifacts not found. Train the model first!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")

if __name__ == "__main__":
    main()
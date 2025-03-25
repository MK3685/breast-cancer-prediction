import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load model and feature names
model = joblib.load('breast_cancer_model.pkl')
feature_names = joblib.load('feature_names.pkl')

def main():
    st.title("🩺 Breast Cancer Diagnosis Predictor")
    
    # Create input fields with decimal precision
    input_data = {}
    for feature in feature_names:
        input_data[feature] = st.number_input(
            label=f"{feature.replace('_', ' ').title()}",
            min_value=0.0,
            max_value=1000.0,  # Adjust based on your feature's range
            value=10.0,  # Default value
            step=0.00001,  # Allows 5 decimal increments
            format="%.5f"  # Displays 5 decimal places
        )
    
    if st.button("Predict Diagnosis"):
        # Create DataFrame with EXACTLY the same structure
        input_df = pd.DataFrame([input_data])[feature_names]
        
        # Ensure numeric types
        input_df = input_df.astype(float)
        
        # Make prediction
        try:
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]
            
            st.subheader("Results")
            if prediction == 1:
                st.error(f"Malignant ({(probability*100):.2f}% confidence)")
            else:
                st.success(f"Benign ({(1-probability)*100:.2f}% confidence)")
                
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
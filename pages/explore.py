import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

def main():
    st.title("🔍 Data Exploration")
    
    # Load and preprocess data
    df = pd.read_excel("cancer.xlsx").drop(columns=['id'])
    le = LabelEncoder()
    df['diagnosis'] = le.fit_transform(df['diagnosis'])
    
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
        
    # Correlation matrix fix
    st.subheader("Feature Correlation Matrix")
    numeric_df = df.select_dtypes(include=['number'])
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
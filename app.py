import streamlit as st
from sklearn.preprocessing import LabelEncoder


st.set_page_config(page_title="Breast Cancer Analysis", layout="wide")

# Initialize label encoder
le = LabelEncoder()

# Sidebar navigation
page = st.sidebar.selectbox("Choose Page", ["🩺 Predict Diagnosis","🔍 Explore Data"])


if page == "🔍 Explore Data":
    from pages import explore
    explore.main()
else:
    from pages import predict
    predict.main()
    
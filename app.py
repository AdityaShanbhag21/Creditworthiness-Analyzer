import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pipeline import load_and_preprocess, get_model

# Page Configuration
st.set_page_config(page_title="Credit Kaka", layout="centered")

st.title("🏦 Financial Loan Predictor")
st.markdown("---")

# Load Data and Pipeline
X, y, encoders = load_and_preprocess()

# 1. MODEL SELECTION (Keep in Sidebar or move to top)
st.sidebar.header("Decision Engine")
model_type = st.sidebar.selectbox("Select ML Model", ("XGBoost", "Random Forest", "Logistic Regression"))
model = get_model(model_type)
model.fit(X, y)

# 2. MAIN INPUT SECTION (Center of the page)
st.subheader("📝 Applicant Details")
with st.container():
    col_a, col_b = st.columns(2)
    
    with col_a:
        income = st.number_input("Annual Income (₹)", value=500000, step=10000)
        loan_amt = st.number_input("Loan Amount Requested (₹)", value=1000000, step=10000)
        cibil = st.slider("CIBIL Score", 300, 900, 700)
        dependents = st.number_input("Number of Dependents", 0, 10, 2)
        
    with col_b:
        term = st.selectbox("Loan Term (Years)", [2, 5, 7, 10, 15, 20])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        employed = st.selectbox("Self Employed?", ["Yes", "No"])
        
    st.markdown("#### 💎 Asset Valuation")
    col_c, col_d = st.columns(2)
    with col_c:
        res_asset = st.number_input("Residential Assets", value=0)
        com_asset = st.number_input("Commercial Assets", value=0)
    with col_d:
        lux_asset = st.number_input("Luxury Assets", value=0)
        bank_asset = st.number_input("Bank Balance Assets", value=0)

# 3. PREDICTION ACTION
st.markdown("---")
if st.button("🚀 Analyze Loan Request", use_container_width=True):
    total_assets = res_asset + com_asset + lux_asset + bank_asset
    
    # Prepare input for model
    input_df = pd.DataFrame([[
        dependents, income, loan_amt, term, cibil, total_assets,
        encoders['education'].transform([education])[0],
        encoders['self_employed'].transform([employed])[0]
    ]], columns=X.columns)

    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success(f"### Result: APPROVED (Confidence: {prob*100:.1f}%)")
        st.balloons()
    else:
        st.error(f"### Result: REJECTED (Risk: {(1-prob)*100:.1f}%)")
    
    # ... inside the "if st.button" block ...

    # Show Feature Importance so you can see why it's rejecting
    st.markdown("#### 🔍 Decision Factors (Why this result?)")
    importances = model.feature_importances_
    feat_importances = pd.Series(importances, index=X.columns)
    
    fig_imp, ax_imp = plt.subplots(figsize=(6, 3))
    feat_importances.nlargest(5).plot(kind='barh', color='skyblue')
    plt.title("Top Factors Influencing Decision")
    st.pyplot(fig_imp, use_container_width=False)

# 4. DATA VISUALIZATION (Downsized & Organized in Tabs)
st.markdown("---")
st.subheader("📊 Data Insights")
tab1, tab2 = st.tabs(["CIBIL Analysis", "Financial Distribution"])

with tab1:
    # Small Figure Size for cleaner look
    fig, ax = plt.subplots(figsize=(6, 3)) 
    sns.boxenplot(x=y.map({1:'Approved', 0:'Rejected'}), y=X['cibil_score'], ax=ax, palette='coolwarm')
    plt.title("CIBIL Score vs Loan Status")
    st.pyplot(fig, use_container_width=False) # Controlled width

with tab2:
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    sns.scatterplot(x=X['income_annum'], y=X['loan_amount'], hue=y.map({1:'Approved', 0:'Rejected'}), ax=ax2, alpha=0.6)
    plt.title("Income vs Loan Amount Correlation")
    st.pyplot(fig2, use_container_width=False)
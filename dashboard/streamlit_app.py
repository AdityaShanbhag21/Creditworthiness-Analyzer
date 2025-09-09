import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from prediction_service import CreditworthinesssPredictionService
from config import MODELS_DIR, PROCESSED_DATA_DIR, STREAMLIT_CONFIG, RISK_CATEGORIES

# Page config
st.set_page_config(
    page_title=STREAMLIT_CONFIG['page_title'],
    page_icon=STREAMLIT_CONFIG['page_icon'],
    layout=STREAMLIT_CONFIG['layout'],
    initial_sidebar_state=STREAMLIT_CONFIG['sidebar_state']
)

# Custom CSS
st.markdown(
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .risk-high {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        border-radius: 0.25rem;
    }
    .risk-medium {
        background-color: #fff8e1;
        border-left: 5px solid #ff9800;
        padding: 1rem;
        border-radius: 0.25rem;
    }
    .risk-low {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
        padding: 1rem;
        border-radius: 0.25rem;
    }
    .explanation-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
    }
</style>
, unsafe_allow_html=True)

@st.cache_resource
def load_prediction_service():
    """Load the prediction service (cached)"""
    try:
        preprocessors_path = PROCESSED_DATA_DIR / "preprocessors.joblib"
        service = CreditworthinesssPredictionService(MODELS_DIR, preprocessors_path)
        return service
    except Exception as e:
        st.error(f"Error loading prediction service: {str(e)}")
        return None

def create_gauge_chart(value, title, max_value=1):
    """Create a gauge chart for risk probability"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        delta = {'reference': 0.5},
        gauge = {
            'axis': {'range': [None, max_value]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 0.3], 'color': "lightgreen"},
                {'range': [0.3, 0.7], 'color': "yellow"},
                {'range': [0.7, 1], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.7
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def create_feature_importance_chart(feature_data):
    """Create feature importance visualization"""
    df = pd.DataFrame(list(feature_data.items()), columns=['Feature', 'Value'])
    
    fig = px.bar(
        df.head(10), 
        x='Value', 
        y='Feature',
        orientation='h',
        title='Top 10 Most Important Features',
        color='Value',
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(height=500, yaxis={'categoryorder':'total ascending'})
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🏦 AI-Powered Creditworthiness Analyzer</h1>', unsafe_allow_html=True)
    
    # Load prediction service
    service = load_prediction_service()
    
    if service is None:
        st.error("Failed to load prediction service. Please ensure models are trained and available.")
        return
    
    # Sidebar - Model Info
    with st.sidebar:
        st.header("🔧 Model Information")
        model_info = service.get_model_info()
        st.write(f"**Available Models:** {model_info['total_models']}")
        st.write(f"**Best Model:** {model_info['best_model']}")
        st.write(f"**Models:** {', '.join(model_info['available_models'])}")
        
        st.header("📊 Instructions")
        st.markdown("""
        **How to use:**
        1. Fill in the applicant information
        2. Click 'Analyze Creditworthiness'
        3. Review the risk assessment
        4. Check the detailed explanation
        5. Make informed lending decisions
        """)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🏠 Single Prediction", "📊 Batch Analysis", "📈 Model Performance", "ℹ️ About"])
    
    with tab1:
        st.header("Individual Credit Risk Assessment")
        
        # Input form
        with st.form("credit_assessment_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("👤 Personal Information")
                age = st.number_input("Age", min_value=18, max_value=100, value=35)
                marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
                education_level = st.selectbox("Education Level", ["High School", "Bachelor", "Master", "PhD"])
                
            with col2:
                st.subheader("💼 Employment Information")
                employment_status = st.selectbox("Employment Status", ["Employed", "Self-employed", "Unemployed", "Student"])
                employment_length = st.number_input("Employment Length (years)", min_value=0.0, max_value=50.0, value=5.0)
                annual_income = st.number_input("Annual Income ($)", min_value=0, value=65000)
                
            with col3:
                st.subheader("🏠 Housing & Loan Info")
                home_ownership = st.selectbox("Home Ownership", ["Own", "Rent", "Mortgage"])
                loan_amount = st.number_input("Loan Amount ($)", min_value=1000, value=25000)
                loan_purpose = st.selectbox("Loan Purpose", ["Home", "Auto", "Personal", "Education", "Business"])
            
            col4, col5 = st.columns(2)
            
            with col4:
                st.subheader("💳 Credit Information")
                credit_utilization_ratio = st.slider("Credit Utilization Ratio", 0.0, 1.0, 0.3, 0.01)
                num_credit_accounts = st.number_input("Number of Credit Accounts", min_value=0, max_value=20, value=4)
                num_previous_loans = st.number_input("Number of Previous Loans", min_value=0, max_value=20, value=2)
                
            with col5:
                st.subheader("💰 Financial Obligations")
                monthly_debt_payments = st.number_input("Monthly Debt Payments ($)", min_value=0, value=1500)
                
                # Calculated metrics (display only)
                debt_to_income = (monthly_debt_payments * 12 / annual_income) if annual_income > 0 else 0
                st.metric("Debt-to-Income Ratio", f"{debt_to_income:.2%}")
                
                loan_to_income = (loan_amount / annual_income) if annual_income > 0 else 0
                st.metric("Loan-to-Income Ratio", f"{loan_to_income:.2f}")
            
            # Submit button
            submitted = st.form_submit_button("🔍 Analyze Creditworthiness", type="primary")
        
        if submitted:
            # Prepare input data
            input_data = {
                'age': age,
                'annual_income': annual_income,
                'employment_status': employment_status,
                'employment_length': employment_length,
                'education_level': education_level,
                'home_ownership': home_ownership,
                'loan_amount': loan_amount,
                'loan_purpose': loan_purpose,
                'monthly_debt_payments': monthly_debt_payments,
                'credit_utilization_ratio': credit_utilization_ratio,
                'num_credit_accounts': num_credit_accounts,
                'num_previous_loans': num_previous_loans,
                'marital_status': marital_status
            }
            
            # Validate input
            validation = service.validate_input(input_data)
            
            if not validation['is_valid']:
                st.error("Please fix the following errors:")
                for error in validation['errors']:
                    st.error(f"❌ {error}")
                return
            
            if validation['warnings']:
                for warning in validation['warnings']:
                    st.warning(f"⚠️ {warning}")
            
            # Make prediction
            with st.spinner("Analyzing creditworthiness..."):
                try:
                    result = service.predict_single(input_data)
                    
                    # Display results
                    st.success("✅ Analysis Complete!")
                    
                    # Risk level display
                    risk_level = result['risk_level']
                    if risk_level == "High":
                        risk_class = "risk-high"
                    elif risk_level == "Medium":
                        risk_class = "risk-medium"
                    else:
                        risk_class = "risk-low"
                    
                    st.markdown(f'<div class="{risk_class}"><h3>Risk Level: {risk_level}</h3></div>', unsafe_allow_html=True)
                    
                    # Metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            label="Risk Probability",
                            value=f"{result['risk_probability']:.1%}",
                            delta=f"{result['risk_probability'] - 0.5:.1%}"
                        )
                    
                    with col2:
                        st.metric(
                            label="Approval Probability", 
                            value=f"{result['approval_probability']:.1%}",
                            delta=f"{result['approval_probability'] - 0.5:.1%}"
                        )
                    
                    with col3:
                        st.metric(
                            label="Model Used",
                            value=result['model_used'].replace('_', ' ').title()
                        )
                    
                    with col4:
                        recommendation = result['recommendation']['decision']
                        st.metric(
                            label="Recommendation",
                            value=recommendation
                        )
                    
                    # Gauge chart
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_gauge = create_gauge_chart(
                            result['risk_probability'], 
                            "Default Risk Probability"
                        )
                        st.plotly_chart(fig_gauge, use_container_width=True)
                    
                    with col2:
                        # Risk factors visualization
                        risk_factors = {
                            'Debt-to-Income': debt_to_income,
                            'Credit Utilization': credit_utilization_ratio,
                            'Loan-to-Income': min(loan_to_income, 5) / 5,  # Normalize
                            'Age Factor': max(0, (35 - age) / 35) if age < 35 else 0,
                            'Employment Risk': 1 if employment_status == 'Unemployed' else 0
                        }
                        
                        fig_factors = px.bar(
                            x=list(risk_factors.keys()),
                            y=list(risk_factors.values()),
                            title="Risk Factors Analysis",
                            color=list(risk_factors.values()),
                            color_continuous_scale='RdYlGn_r'
                        )
                        fig_factors.update_layout(height=300, showlegend=False)
                        st.plotly_chart(fig_factors, use_container_width=True)
                    
                    # Explanation
                    st.markdown('<div class="explanation-box">', unsafe_allow_html=True)
                    st.subheader("📋 Detailed Explanation")
                    for explanation in result['explanation']:
                        st.markdown(f"- {explanation}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Recommendations
                    st.subheader("💡 Lending Recommendations")
                    rec = result['recommendation']
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.info(f"**Decision:** {rec['decision']}")
                        st.info(f"**Confidence:** {rec['confidence']}")
                    
                    with col2:
                        st.write("**Suggested Terms:**")
                        for key, value in rec['suggested_terms'].items():
                            st.write(f"- **{key.replace('_', ' ').title()}:** {value}")
                
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
    
    with tab2:
        st.header("📊 Batch Credit Risk Analysis")
        
        st.markdown("""
        Upload a CSV file with multiple loan applications for batch processing.
        The CSV should contain the same columns as the single prediction form.
        """)
        
        # File upload
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                # Load data
                df = pd.read_csv(uploaded_file)
                st.write("**Data Preview:**")
                st.dataframe(df.head())
                
                if st.button("🔍 Analyze Batch", type="primary"):
                    # Convert DataFrame to list of dictionaries
                    input_data_list = df.to_dict('records')
                    
                    with st.spinner("Processing batch predictions..."):
                        # Make batch predictions
                        results = service.predict_batch(input_data_list)
                        
                        # Process results
                        successful_results = [r for r in results if 'error' not in r]
                        failed_results = [r for r in results if 'error' in r]
                        
                        if successful_results:
                            # Create results DataFrame
                            results_df = pd.DataFrame([
                                {
                                    'Applicant_ID': r['applicant_id'],
                                    'Risk_Level': r['risk_level'],
                                    'Risk_Probability': r['risk_probability'],
                                    'Approval_Probability': r['approval_probability'],
                                    'Recommendation': r['recommendation']['decision'],
                                    'Model_Used': r['model_used']
                                }
                                for r in successful_results
                            ])
                            
                            st.success(f"✅ Successfully processed {len(successful_results)} applications")
                            
                            # Display results
                            st.subheader("📋 Batch Results")
                            st.dataframe(results_df)
                            
                            # Summary statistics
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Total Applications", len(results_df))
                            
                            with col2:
                                high_risk_count = len(results_df[results_df['Risk_Level'] == 'High'])
                                st.metric("High Risk", high_risk_count)
                            
                            with col3:
                                approved_count = len(results_df[results_df['Recommendation'] == 'APPROVE'])
                                st.metric("Approved", approved_count)
                            
                            with col4:
                                avg_risk = results_df['Risk_Probability'].mean()
                                st.metric("Avg Risk Probability", f"{avg_risk:.1%}")
                            
                            # Visualizations
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Risk level distribution
                                risk_dist = results_df['Risk_Level'].value_counts()
                                fig_risk = px.pie(
                                    values=risk_dist.values,
                                    names=risk_dist.index,
                                    title="Risk Level Distribution"
                                )
                                st.plotly_chart(fig_risk, use_container_width=True)
                            
                            with col2:
                                # Recommendation distribution
                                rec_dist = results_df['Recommendation'].value_counts()
                                fig_rec = px.bar(
                                    x=rec_dist.index,
                                    y=rec_dist.values,
                                    title="Recommendation Distribution"
                                )
                                st.plotly_chart(fig_rec, use_container_width=True)
                            
                            # Risk probability histogram
                            fig_hist = px.histogram(
                                results_df,
                                x='Risk_Probability',
                                bins=20,
                                title="Risk Probability Distribution"
                            )
                            st.plotly_chart(fig_hist, use_container_width=True)
                            
                            # Download results
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results as CSV",
                                data=csv,
                                file_name="credit_risk_analysis_results.csv",
                                mime="text/csv"
                            )
                        
                        if failed_results:
                            st.warning(f"⚠️ {len(failed_results)} applications failed to process")
                            with st.expander("View Failed Applications"):
                                for result in failed_results:
                                    st.error(f"Applicant {result['applicant_id']}: {result['error']}")
            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
        
        else:
            # Sample data download
            st.subheader("📥 Sample Data")
            st.markdown("Download a sample CSV file to test batch processing:")
            
            sample_data = pd.DataFrame([
                {
                    'age': 35, 'annual_income': 65000, 'employment_status': 'Employed',
                    'employment_length': 5.0, 'education_level': 'Bachelor', 'home_ownership': 'Mortgage',
                    'loan_amount': 25000, 'loan_purpose': 'Home', 'monthly_debt_payments': 1500,
                    'credit_utilization_ratio': 0.3, 'num_credit_accounts': 4, 'num_previous_loans': 2,
                    'marital_status': 'Married'
                },
                {
                    'age': 28, 'annual_income': 45000, 'employment_status': 'Employed',
                    'employment_length': 2.0, 'education_level': 'High School', 'home_ownership': 'Rent',
                    'loan_amount': 15000, 'loan_purpose': 'Auto', 'monthly_debt_payments': 800,
                    'credit_utilization_ratio': 0.7, 'num_credit_accounts': 2, 'num_previous_loans': 1,
                    'marital_status': 'Single'
                },
                {
                    'age': 42, 'annual_income': 85000, 'employment_status': 'Self-employed',
                    'employment_length': 8.0, 'education_level': 'Master', 'home_ownership': 'Own',
                    'loan_amount': 35000, 'loan_purpose': 'Business', 'monthly_debt_payments': 2000,
                    'credit_utilization_ratio': 0.2, 'num_credit_accounts': 6, 'num_previous_loans': 3,
                    'marital_status': 'Married'
                }
            ])
            
            csv_sample = sample_data.to_csv(index=False)
            st.download_button(
                label="📥 Download Sample CSV",
                data=csv_sample,
                file_name="sample_credit_applications.csv",
                mime="text/csv"
            )
    
    with tab3:
        st.header("📈 Model Performance Dashboard")
        
        # Load model performance data
        try:
            import json
            from pathlib import Path
            
            eval_report_path = MODELS_DIR / "evaluation_report.json"
            if eval_report_path.exists():
                with open(eval_report_path, 'r') as f:
                    eval_report = json.load(f)
                
                # Model comparison
                st.subheader("🏆 Model Comparison")
                
                comparison_df = pd.DataFrame(eval_report['model_comparison']).T
                st.dataframe(comparison_df)
                
                # Best model highlight
                best_model = eval_report['recommendations']['best_overall_model']
                st.success(f"🥇 Best Overall Model: **{best_model}**")
                
                # Metrics visualization
                col1, col2 = st.columns(2)
                
                with col1:
                    # AUC Score comparison
                    auc_scores = {model: float(data['AUC Score']) 
                                 for model, data in eval_report['model_comparison'].items()}
                    
                    fig_auc = px.bar(
                        x=list(auc_scores.keys()),
                        y=list(auc_scores.values()),
                        title="Model AUC Score Comparison",
                        color=list(auc_scores.values()),
                        color_continuous_scale='Blues'
                    )
                    fig_auc.update_layout(showlegend=False)
                    st.plotly_chart(fig_auc, use_container_width=True)
                
                with col2:
                    # F1 Score comparison
                    f1_scores = {model: float(data['F1 Score']) 
                                for model, data in eval_report['model_comparison'].items()}
                    
                    fig_f1 = px.bar(
                        x=list(f1_scores.keys()),
                        y=list(f1_scores.values()),
                        title="Model F1 Score Comparison",
                        color=list(f1_scores.values()),
                        color_continuous_scale='Greens'
                    )
                    fig_f1.update_layout(showlegend=False)
                    st.plotly_chart(fig_f1, use_container_width=True)
                
                # Recommendations
                st.subheader("💡 Model Selection Guidance")
                guidance = eval_report['recommendations']['model_selection_guidance']
                
                for use_case, recommendation in guidance.items():
                    st.info(f"**{use_case}:** {recommendation}")
                
            else:
                st.warning("Model evaluation report not found. Please run model training first.")
        
        except Exception as e:
            st.error(f"Error loading model performance data: {str(e)}")
    
    with tab4:
        st.header("ℹ️ About the AI Creditworthiness Analyzer")
        
        st.markdown(
        ## 🎯 Purpose
        This application uses advanced machine learning algorithms to assess creditworthiness and predict loan default risk. 
        It helps financial institutions make informed lending decisions by analyzing multiple factors beyond traditional credit scores.
        
        ## 🧠 Machine Learning Models
        The system employs three different algorithms:
        
        ### 1. 📊 Logistic Regression
        - **Use Case:** Baseline model with high interpretability
        - **Strengths:** Fast predictions, clear feature importance
        - **Best For:** Understanding key risk factors
        
        ### 2. 🌳 Random Forest
        - **Use Case:** Ensemble method handling complex interactions
        - **Strengths:** Robust to outliers, handles missing data well
        - **Best For:** Balanced accuracy and interpretability
        
        ### 3. 🚀 XGBoost
        - **Use Case:** Gradient boosting for highest accuracy
        - **Strengths:** Superior performance, handles imbalanced data
        - **Best For:** Maximum predictive accuracy
        
        ## 📊 Key Features Analyzed
        
        ### Personal Information
        - Age and demographic factors
        - Education level and marital status
        
        ### Employment & Income
        - Employment status and stability
        - Annual income and employment length
        
        ### Financial Profile
        - Debt-to-income ratio
        - Credit utilization patterns
        - Existing credit accounts and loan history
        
        ### Loan Characteristics
        - Loan amount and purpose
        - Loan-to-income ratio
        
        ## 🔍 Risk Assessment Process
        
        1. **Data Preprocessing:** Clean and standardize input data
        2. **Feature Engineering:** Create derived financial metrics
        3. **Model Prediction:** Apply trained ML models
        4. **Risk Categorization:** Classify into Low/Medium/High risk
        5. **Explanation Generation:** Provide human-readable insights
        
        ## 📈 Performance Metrics
        
        - **AUC Score:** Area Under ROC Curve (discrimination ability)
        - **Precision:** Accuracy of positive predictions
        - **Recall:** Ability to identify all positive cases
        - **F1 Score:** Harmonic mean of precision and recall
        
        ## ⚠️ Important Disclaimers
        
        - This tool is for **informational purposes** and should not be the sole basis for lending decisions
        - Always comply with fair lending practices and regulations
        - Consider additional factors not captured in the model
        - Regular model retraining is recommended to maintain accuracy
        
        ## 🔄 Model Updates
        
        The models are trained on synthetic data for demonstration purposes. 
        For production use, retrain with your organization's historical loan data.
        
        ## 📞 Support
        
        For technical support or customization requests, please contact your development team.
        )
        
        # Technical specifications
        with st.expander("🔧 Technical Specifications"):
            st.markdown("""
            - **Framework:** Streamlit + Scikit-learn + XGBoost
            - **Data Processing:** Pandas + NumPy
            - **Visualization:** Plotly + Matplotlib
            - **Model Storage:** Joblib serialization
            - **Feature Engineering:** Custom transformations + SMOTE for imbalance
            - **Validation:** Cross-validation + Hold-out testing
            """)

if __name__ == "__main__":

    main()

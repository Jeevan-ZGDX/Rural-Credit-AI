import streamlit as st
import pandas as pd
import numpy as np
from models.credit_model import CreditScoringModel
from data.data_generator import DataGenerator
from utils.visualization import create_visualizations
from utils.explanations import explain_prediction
import plotly.express as px
import plotly.graph_objects as go

# Configure page
st.set_page_config(
    page_title="Rural Credit Scoring System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = CreditScoringModel()
if 'data_generator' not in st.session_state:
    st.session_state.data_generator = DataGenerator()

# Sidebar navigation
st.sidebar.title("🏦 Rural Credit Scoring")
st.sidebar.markdown("---")

page = st.sidebar.selectbox(
    "Navigate to:",
    ["Home", "Loan Application", "Model Training", "Dashboard", "Analytics"]
)

# Home Page
if page == "Home":
    st.title("Rural Small Business Credit Scoring System")
    st.markdown("""
    ### 🌾 Empowering Rural Entrepreneurs with AI-Driven Credit Assessment
    
    This system leverages non-traditional data sources to assess creditworthiness for rural small businesses
    that often lack formal credit history.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📱 Mobile Data Sources", "4 Types", "Call patterns, SMS, Data usage, Airtime")
    
    with col2:
        st.metric("💰 Financial Indicators", "6 Categories", "Mobile money, Utility bills, Cooperative records")
    
    with col3:
        st.metric("🤖 ML Models", "3 Algorithms", "Logistic Regression, Random Forest, XGBoost")
    
    st.markdown("---")
    
    # Key Features
    st.subheader("🚀 Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **📊 Alternative Data Sources:**
        - Mobile phone usage patterns
        - Mobile money transaction history
        - Utility payment consistency
        - Community cooperative records
        """)
        
        st.markdown("""
        **🧠 Machine Learning Models:**
        - Logistic Regression for baseline scoring
        - Random Forest for feature importance
        - XGBoost for advanced predictions
        """)
    
    with col2:
        st.markdown("""
        **🔍 Model Explainability:**
        - SHAP values for feature impact
        - Individual prediction explanations
        - Global feature importance
        """)
        
        st.markdown("""
        **⚖️ Fairness & Ethics:**
        - Bias detection and mitigation
        - Demographic parity analysis
        - Equal opportunity assessment
        """)
    
    # Sample data overview
    st.subheader("📈 Sample Data Overview")
    
    # Generate sample data for display
    sample_data = st.session_state.data_generator.generate_sample_data(100)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Credit Score Distribution**")
        fig = px.histogram(sample_data, x='credit_score', nbins=20, 
                          title="Credit Score Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**Approval Rate by Region**")
        approval_by_region = sample_data.groupby('region')['loan_approved'].mean().reset_index()
        fig = px.bar(approval_by_region, x='region', y='loan_approved',
                    title="Loan Approval Rate by Region")
        st.plotly_chart(fig, use_container_width=True)

elif page == "Loan Application":
    exec(open('pages/loan_application.py').read())

elif page == "Model Training":
    exec(open('pages/model_training.py').read())

elif page == "Dashboard":
    exec(open('pages/dashboard.py').read())

elif page == "Analytics":
    st.title("📊 Advanced Analytics")
    st.markdown("### Model Performance and Fairness Metrics")
    
    # Generate analysis data
    data = st.session_state.data_generator.generate_sample_data(1000)
    
    # Train model for analysis
    if st.button("Generate Analytics Report"):
        with st.spinner("Training models and generating analytics..."):
            X_train, X_test, y_train, y_test = st.session_state.model.prepare_data(data)
            models = st.session_state.model.train_models(X_train, y_train)
            
            # Model Performance Comparison
            st.subheader("🎯 Model Performance Comparison")
            
            performance_data = []
            for name, model in models.items():
                y_pred = model.predict(X_test)
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                
                performance_data.append({
                    'Model': name,
                    'Accuracy': accuracy_score(y_test, y_pred),
                    'Precision': precision_score(y_test, y_pred),
                    'Recall': recall_score(y_test, y_pred),
                    'F1-Score': f1_score(y_test, y_pred)
                })
            
            perf_df = pd.DataFrame(performance_data)
            st.dataframe(perf_df, use_container_width=True)
            
            # Fairness Analysis
            st.subheader("⚖️ Fairness Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Demographic parity by region
                test_data = data.iloc[X_test.index]
                test_data['prediction'] = models['Random Forest'].predict(X_test)
                
                fairness_by_region = test_data.groupby('region')['prediction'].mean().reset_index()
                fig = px.bar(fairness_by_region, x='region', y='prediction',
                            title="Approval Rate by Region (Demographic Parity)")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Feature importance fairness
                feature_names = X_train.columns
                importance = models['Random Forest'].feature_importances_
                
                fig = go.Figure(data=go.Bar(
                    x=importance[:10],
                    y=feature_names[:10],
                    orientation='h'
                ))
                fig.update_layout(title="Top 10 Feature Importance")
                st.plotly_chart(fig, use_container_width=True)
            
            # Model Explainability
            st.subheader("🔍 Global Model Explanation")
            
            # Sample prediction explanation
            sample_idx = 0
            explanation = explain_prediction(
                models['Random Forest'], 
                X_test.iloc[[sample_idx]], 
                X_train, 
                feature_names
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**SHAP Feature Impact**")
                st.dataframe(explanation['feature_impact'])
            
            with col2:
                st.markdown("**Prediction Details**")
                st.json({
                    "Predicted Probability": float(explanation['probability']),
                    "Prediction": explanation['prediction'],
                    "Confidence": f"{explanation['probability']:.2%}"
                })

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Rural Credit Scoring System v1.0**

Developed for financial inclusion in underserved rural communities.

*Powered by Machine Learning & Alternative Data*
""")

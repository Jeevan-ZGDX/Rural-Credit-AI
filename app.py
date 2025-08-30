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

# Custom CSS for more golden yellow highlights
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(45deg, #008080, #FFD700);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(255, 215, 0, 0.3);
    }
    
    .golden-header {
        color: #FFD700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        border-bottom: 3px solid #FFD700;
        padding-bottom: 10px;
    }
    
    .highlight-box {
        background: linear-gradient(135deg, #000000, #008080);
        border-left: 5px solid #FFD700;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(255, 215, 0, 0.2);
    }
    
    .golden-button {
        background: linear-gradient(45deg, #FFD700, #FFA500);
        color: black;
        border: none;
        padding: 10px 20px;
        border-radius: 25px;
        font-weight: bold;
        box-shadow: 0 4px 8px rgba(255, 215, 0, 0.4);
    }
    
    .sidebar .metric-container {
        background: linear-gradient(135deg, #008080, #FFD700);
        border-radius: 8px;
        padding: 8px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

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
    st.markdown('<h1 class="golden-header">🏦 Rural Small Business Credit Scoring System</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div class="highlight-box">
    <h3>🌾 Empowering Rural Entrepreneurs with AI-Driven Credit Assessment</h3>
    
    This system leverages non-traditional data sources to assess creditworthiness for rural small businesses
    that often lack formal credit history.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📱 Mobile Data Sources", "4 Types", "Call patterns, SMS, Data usage, Airtime")
    
    with col2:
        st.metric("💰 Financial Indicators", "6 Categories", "Mobile money, Utility bills, Cooperative records")
    
    with col3:
        st.metric("🤖 ML Models", "3 Algorithms", "Logistic Regression, Random Forest, XGBoost")
    
    st.markdown("---")
    
    # Key Features
    st.markdown('<h3 class="golden-header">🚀 Key Features</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="highlight-box">
        <strong style="color: #FFD700;">📊 Alternative Data Sources:</strong>
        <ul style="color: white;">
        <li>Mobile phone usage patterns</li>
        <li>Mobile money transaction history</li>
        <li>Utility payment consistency</li>
        <li>Community cooperative records</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="highlight-box">
        <strong style="color: #FFD700;">🧠 Machine Learning Models:</strong>
        <ul style="color: white;">
        <li>Logistic Regression for baseline scoring</li>
        <li>Random Forest for feature importance</li>
        <li>XGBoost for advanced predictions</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="highlight-box">
        <strong style="color: #FFD700;">🔍 Model Explainability:</strong>
        <ul style="color: white;">
        <li>SHAP values for feature impact</li>
        <li>Individual prediction explanations</li>
        <li>Global feature importance</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="highlight-box">
        <strong style="color: #FFD700;">⚖️ Fairness & Ethics:</strong>
        <ul style="color: white;">
        <li>Bias detection and mitigation</li>
        <li>Demographic parity analysis</li>
        <li>Equal opportunity assessment</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Sample data overview
    st.markdown('<h3 class="golden-header">📈 Sample Data Overview</h3>', unsafe_allow_html=True)
    
    # Generate sample data for display
    sample_data = st.session_state.data_generator.generate_sample_data(100)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Credit Score Distribution**")
        fig = px.histogram(sample_data, x='credit_score', nbins=20, 
                          title="Credit Score Distribution",
                          color_discrete_sequence=['#FFD700'])
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            title_font_color='#FFD700'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**Approval Rate by Region**")
        approval_by_region = sample_data.groupby('region')['loan_approved'].mean().reset_index()
        fig = px.bar(approval_by_region, x='region', y='loan_approved',
                    title="Loan Approval Rate by Region",
                    color_discrete_sequence=['#FFD700'])
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            title_font_color='#FFD700'
        )
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

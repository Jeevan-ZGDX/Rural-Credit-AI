import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils.explanations import explain_prediction

# Custom CSS for golden yellow styling
st.markdown("""
<style>
    .golden-header {
        color: #FFD700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        border-bottom: 3px solid #FFD700;
        padding-bottom: 10px;
    }
    
    .application-section {
        background: linear-gradient(135deg, #000000, #008080);
        border-left: 5px solid #FFD700;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(255, 215, 0, 0.2);
    }
    
    .success-box {
        background: linear-gradient(45deg, #008080, #FFD700);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(255, 215, 0, 0.3);
        text-align: center;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="golden-header">📋 Loan Application Form</h1>', unsafe_allow_html=True)
st.markdown('<h3 style="color: #FFD700;">Apply for Rural Small Business Credit</h3>', unsafe_allow_html=True)

# Create application form
with st.form("loan_application_form"):
    st.markdown('<h4 class="golden-header">👤 Personal Information</h4>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=80, value=35)
        gender = st.selectbox("Gender", ["Male", "Female"])
        education_level = st.selectbox("Education Level", 
                                     ["Primary", "Secondary", "Vocational", "University"])
    
    with col2:
        region = st.selectbox("Region", 
                            ["Northern", "Southern", "Eastern", "Western", "Central"])
        business_type = st.selectbox("Business Type", 
                                   ["Agriculture", "Retail", "Services", "Manufacturing", "Trading"])
        business_age = st.number_input("Business Age (years)", min_value=0.1, max_value=50.0, value=2.0)
    
    st.markdown('<h4 class="golden-header">💰 Financial Information</h4>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        monthly_income = st.number_input("Monthly Income", min_value=5000, max_value=200000, value=30000)
        savings_balance = st.number_input("Savings Balance", min_value=0, max_value=500000, value=15000)
    
    with col2:
        loan_amount = st.number_input("Requested Loan Amount", min_value=5000, max_value=100000, value=25000)
        loan_purpose = st.selectbox("Loan Purpose", 
                                  ["Business Expansion", "Equipment Purchase", "Working Capital", 
                                   "Inventory", "Other"])
    
    st.markdown('<h4 class="golden-header">📱 Alternative Data Assessment</h4>', unsafe_allow_html=True)
    st.markdown("*These scores are automatically calculated based on your mobile and payment history*")
    
    col1, col2 = st.columns(2)
    
    with col1:
        mobile_usage_score = st.slider("Mobile Usage Score", 0.0, 100.0, 65.0, 
                                     help="Based on call frequency, SMS usage, and data patterns")
        mobile_money_score = st.slider("Mobile Money Score", 0.0, 100.0, 70.0,
                                     help="Based on transaction frequency and amounts")
    
    with col2:
        utility_payment_score = st.slider("Utility Payment Score", 0.0, 100.0, 75.0,
                                        help="Based on electricity, water, and mobile bill payments")
        cooperative_score = st.slider("Cooperative Involvement Score", 0.0, 100.0, 55.0,
                                    help="Based on community group participation")
    
    st.subheader("Behavioral Patterns")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        airtime_consistency = st.slider("Airtime Top-up Consistency", 0.0, 1.0, 0.8)
        call_frequency = st.number_input("Average Daily Calls", min_value=0, max_value=100, value=15)
    
    with col2:
        repayment_consistency = st.slider("Repayment Consistency", 0.0, 1.0, 0.85)
        transaction_frequency = st.number_input("Weekly Transactions", min_value=0, max_value=50, value=8)
    
    with col3:
        bill_payment_consistency = st.slider("Bill Payment Consistency", 0.0, 1.0, 0.9)
        sms_frequency = st.number_input("Average Daily SMS", min_value=0, max_value=100, value=12)
    
    # Submit button
    submitted = st.form_submit_button("🚀 Submit Application", type="primary")
    
    if submitted:
        # Prepare applicant data
        applicant_data = {
            'age': age,
            'gender': gender,
            'region': region,
            'business_type': business_type,
            'education_level': education_level,
            'business_age': business_age,
            'monthly_income': monthly_income,
            'mobile_usage_score': mobile_usage_score,
            'airtime_consistency': airtime_consistency,
            'call_frequency': call_frequency,
            'sms_frequency': sms_frequency,
            'data_usage_trend': np.random.normal(0.5, 0.2),  # Simulated
            'mobile_money_score': mobile_money_score,
            'transaction_frequency': transaction_frequency,
            'avg_transaction_amount': monthly_income * 0.3,  # Estimated
            'repayment_consistency': repayment_consistency,
            'utility_payment_score': utility_payment_score,
            'bill_payment_consistency': bill_payment_consistency,
            'cooperative_score': cooperative_score,
            'savings_balance': savings_balance
        }
        
        # Train model if not already trained
        if not st.session_state.model.models:
            with st.spinner("Training credit scoring models..."):
                sample_data = st.session_state.data_generator.generate_sample_data(1000)
                X_train, X_test, y_train, y_test = st.session_state.model.prepare_data(sample_data)
                st.session_state.model.train_models(X_train, y_train)
        
        # Make prediction
        with st.spinner("Processing your application..."):
            try:
                prediction = st.session_state.model.predict_credit_score(applicant_data, 'Random Forest')
                
                # Display results
                st.success("✅ Application Processed Successfully!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Credit Score", prediction['credit_score'], 
                            delta=prediction['credit_score'] - 650)
                
                with col2:
                    st.metric("Approval Probability", f"{prediction['probability']:.1%}")
                
                with col3:
                    risk_color = {"Low Risk": "🟢", "Medium Risk": "🟡", 
                                "High Risk": "🟠", "Very High Risk": "🔴"}
                    st.metric("Risk Category", 
                            f"{risk_color.get(prediction['risk_category'], '⚪')} {prediction['risk_category']}")
                
                # Loan decision
                if prediction['approved']:
                    st.success(f"🎉 **LOAN APPROVED!** \n\nCongratulations! Your loan application for ${loan_amount:,} has been approved based on your credit assessment.")
                    
                    # Loan terms (simulated)
                    st.subheader("💰 Loan Terms")
                    
                    # Interest rate based on risk
                    base_rate = 12.0
                    if prediction['risk_category'] == "Low Risk":
                        interest_rate = base_rate - 2.0
                    elif prediction['risk_category'] == "Medium Risk":
                        interest_rate = base_rate
                    else:
                        interest_rate = base_rate + 2.0
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.info(f"**Interest Rate:** {interest_rate:.1f}% per annum")
                    with col2:
                        st.info(f"**Loan Term:** 12 months")
                    with col3:
                        monthly_payment = loan_amount * (interest_rate/100/12) * (1 + interest_rate/100/12)**12 / ((1 + interest_rate/100/12)**12 - 1)
                        st.info(f"**Monthly Payment:** ${monthly_payment:.0f}")
                
                else:
                    st.error(f"❌ **LOAN REJECTED** \n\nWe're sorry, but your application does not meet our current lending criteria. Your credit score of {prediction['credit_score']} is below our minimum threshold.")
                    
                    st.subheader("💡 Recommendations for Improvement")
                    st.markdown("""
                    - **Increase Mobile Money Usage**: Regular mobile money transactions demonstrate financial activity
                    - **Improve Payment Consistency**: Maintain regular utility and bill payments
                    - **Build Business History**: Continue operating your business to build a track record
                    - **Community Involvement**: Participate in local cooperatives or financial groups
                    - **Reapply in 6 months**: Work on these factors and reapply
                    """)
                
                # Explanation section
                st.subheader("🔍 Decision Explanation")
                
                # Get model explanation
                try:
                    processed_data = st.session_state.model.preprocess_single_applicant(applicant_data)
                    explanation = explain_prediction(
                        st.session_state.model.models['Random Forest'],
                        processed_data,
                        None,  # Training data not needed for this explanation
                        st.session_state.model.feature_names
                    )
                    
                    # Display top factors
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**🟢 Positive Factors**")
                        positive_factors = explanation['feature_impact'].head(5)
                        for _, row in positive_factors.iterrows():
                            if row.get('shap_value', row.get('impact_score', 0)) > 0:
                                st.write(f"• {row['feature']}: {row['value']:.2f}")
                    
                    with col2:
                        st.markdown("**🔴 Risk Factors**")
                        negative_factors = explanation['feature_impact'].head(5)
                        for _, row in negative_factors.iterrows():
                            if row.get('shap_value', row.get('impact_score', 0)) < 0:
                                st.write(f"• {row['feature']}: {row['value']:.2f}")
                
                except Exception as e:
                    st.warning("Unable to generate detailed explanation at this time.")
                
                # Visualization of applicant profile
                st.subheader("📊 Your Profile Analysis")
                
                # Create gauge chart for credit score
                import plotly.graph_objects as go
                
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=prediction['credit_score'],
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Credit Score", 'font': {'color': '#FFD700', 'size': 24}},
                    delta={'reference': 650},
                    gauge={
                        'axis': {'range': [300, 850], 'tickcolor': '#FFD700'},
                        'bar': {'color': "#FFD700"},
                        'steps': [
                            {'range': [300, 550], 'color': "rgba(128,128,128,0.3)"},
                            {'range': [550, 650], 'color': "rgba(255,165,0,0.3)"},
                            {'range': [650, 750], 'color': "rgba(0,128,128,0.3)"},
                            {'range': [750, 850], 'color': "rgba(255,215,0,0.3)"}
                        ],
                        'threshold': {
                            'line': {'color': "#FFD700", 'width': 4},
                            'thickness': 0.75,
                            'value': 650
                        }
                    }
                ))
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='white'
                )
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error processing application: {str(e)}")
                st.info("Please check your input data and try again.")

# Information sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ Application Tips")
st.sidebar.markdown("""
**Required Documents:**
- Business registration (if available)
- Recent mobile money statements
- Utility bill payment history
- Community group membership proof

**Scoring Factors:**
- Mobile usage patterns (30%)
- Payment consistency (35%)
- Business characteristics (20%)
- Community involvement (15%)

**Processing Time:**
- Instant preliminary assessment
- Final approval: 24-48 hours
- Fund disbursement: 3-5 business days
""")

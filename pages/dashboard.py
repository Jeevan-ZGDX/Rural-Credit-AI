import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime, timedelta

st.title("📊 Credit Scoring Dashboard")
st.markdown("### Comprehensive Analytics and Insights")

# Check if models are trained
if not st.session_state.model.models:
    st.warning("⚠️ No trained models found. Please train models first in the Model Training page.")
    
    # Quick train button
    if st.button("🚀 Quick Train Models"):
        with st.spinner("Training models with sample data..."):
            sample_data = st.session_state.data_generator.generate_sample_data(1000)
            X_train, X_test, y_train, y_test = st.session_state.model.prepare_data(sample_data)
            st.session_state.model.train_models(X_train, y_train)
        st.success("✅ Models trained successfully!")
        st.rerun()
else:
    # Generate comprehensive dataset for dashboard
    dashboard_data = st.session_state.data_generator.generate_sample_data(2000)
    
    # Key Performance Indicators
    st.subheader("📈 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_applications = len(dashboard_data)
        st.metric("Total Applications", f"{total_applications:,}")
    
    with col2:
        approval_rate = dashboard_data['loan_approved'].mean()
        st.metric("Overall Approval Rate", f"{approval_rate:.1%}")
    
    with col3:
        avg_credit_score = dashboard_data['credit_score'].mean()
        st.metric("Average Credit Score", f"{avg_credit_score:.0f}")
    
    with col4:
        high_risk_count = (dashboard_data['credit_score'] < 550).sum()
        st.metric("High Risk Applications", f"{high_risk_count:,}")
    
    # Regional Analysis
    st.subheader("🗺️ Regional Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Approval rate by region
        regional_stats = dashboard_data.groupby('region').agg({
            'loan_approved': ['mean', 'count'],
            'credit_score': 'mean',
            'monthly_income': 'mean'
        }).round(3)
        
        regional_stats.columns = ['Approval_Rate', 'Applications', 'Avg_Credit_Score', 'Avg_Income']
        regional_stats = regional_stats.reset_index()
        
        fig = px.bar(
            regional_stats,
            x='region',
            y='Approval_Rate',
            color='Approval_Rate',
            title='Loan Approval Rate by Region',
            labels={'Approval_Rate': 'Approval Rate', 'region': 'Region'},
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Credit score distribution by region
        fig = px.box(
            dashboard_data,
            x='region',
            y='credit_score',
            title='Credit Score Distribution by Region',
            labels={'credit_score': 'Credit Score', 'region': 'Region'}
        )
        fig.add_hline(y=650, line_dash="dash", line_color="red", 
                     annotation_text="Approval Threshold")
        st.plotly_chart(fig, use_container_width=True)
    
    # Business Type Analysis
    st.subheader("🏢 Business Type Analysis")
    
    business_stats = dashboard_data.groupby('business_type').agg({
        'loan_approved': ['mean', 'count'],
        'credit_score': 'mean',
        'monthly_income': 'mean',
        'business_age': 'mean'
    }).round(2)
    
    business_stats.columns = ['Approval_Rate', 'Applications', 'Avg_Credit_Score', 'Avg_Income', 'Avg_Business_Age']
    business_stats = business_stats.reset_index()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(
            business_stats,
            x='Avg_Income',
            y='Approval_Rate',
            size='Applications',
            color='Avg_Credit_Score',
            hover_data=['business_type'],
            title='Income vs Approval Rate by Business Type',
            labels={
                'Avg_Income': 'Average Monthly Income',
                'Approval_Rate': 'Approval Rate',
                'Avg_Credit_Score': 'Avg Credit Score'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Business type performance table
        st.markdown("**Business Type Performance Summary**")
        display_stats = business_stats.copy()
        display_stats['Approval_Rate'] = display_stats['Approval_Rate'].apply(lambda x: f"{x:.1%}")
        display_stats['Avg_Income'] = display_stats['Avg_Income'].apply(lambda x: f"${x:,.0f}")
        
        st.dataframe(display_stats, use_container_width=True)
    
    # Alternative Data Insights
    st.subheader("📱 Alternative Data Insights")
    
    alt_data_cols = ['mobile_usage_score', 'mobile_money_score', 'utility_payment_score', 'cooperative_score']
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Alternative data correlation with approval
        correlation_data = []
        for col in alt_data_cols:
            correlation = dashboard_data[col].corr(dashboard_data['loan_approved'])
            correlation_data.append({
                'Data Source': col.replace('_', ' ').title(),
                'Correlation': correlation
            })
        
        corr_df = pd.DataFrame(correlation_data)
        
        fig = px.bar(
            corr_df,
            x='Correlation',
            y='Data Source',
            orientation='h',
            title='Alternative Data Correlation with Loan Approval',
            labels={'Correlation': 'Correlation Coefficient'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Radar chart comparing approved vs rejected
        approved_avg = dashboard_data[dashboard_data['loan_approved'] == True][alt_data_cols].mean()
        rejected_avg = dashboard_data[dashboard_data['loan_approved'] == False][alt_data_cols].mean()
        
        categories = [col.replace('_score', '').replace('_', ' ').title() for col in alt_data_cols]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=approved_avg.values.tolist() + [approved_avg.values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name='Approved Loans',
            line_color='green'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=rejected_avg.values.tolist() + [rejected_avg.values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name='Rejected Loans',
            line_color='red'
        ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            title='Alternative Data Profile: Approved vs Rejected'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Model Performance Dashboard
    st.subheader("🤖 Model Performance Dashboard")
    
    if st.session_state.model.models:
        # Generate predictions for dashboard data
        sample_for_prediction = dashboard_data.sample(500)  # Sample for performance
        
        model_performance = {}
        
        for model_name, model in st.session_state.model.models.items():
            try:
                # Prepare data
                X_sample = st.session_state.model.preprocess_single_applicant(sample_for_prediction)
                
                # Make predictions
                predictions = model.predict(X_sample)
                probabilities = model.predict_proba(X_sample)[:, 1]
                
                # Calculate metrics
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                
                y_true = sample_for_prediction['loan_approved'].values
                
                model_performance[model_name] = {
                    'Accuracy': accuracy_score(y_true, predictions),
                    'Precision': precision_score(y_true, predictions),
                    'Recall': recall_score(y_true, predictions),
                    'F1_Score': f1_score(y_true, predictions)
                }
                
            except Exception as e:
                st.warning(f"Could not evaluate {model_name}: {str(e)}")
        
        if model_performance:
            perf_df = pd.DataFrame(model_performance).T.round(3)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Current Model Performance**")
                st.dataframe(perf_df, use_container_width=True)
            
            with col2:
                # Performance visualization
                metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score']
                fig = go.Figure()
                
                for metric in metrics:
                    fig.add_trace(go.Scatter(
                        x=list(perf_df.index),
                        y=perf_df[metric],
                        mode='lines+markers',
                        name=metric.replace('_', ' '),
                        line=dict(width=3),
                        marker=dict(size=8)
                    ))
                
                fig.update_layout(
                    title='Model Performance Comparison',
                    xaxis_title='Model',
                    yaxis_title='Score',
                    yaxis=dict(range=[0, 1]),
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Fairness Analysis
    st.subheader("⚖️ Fairness and Bias Analysis")
    
    fairness_cols = st.columns(2)
    
    with fairness_cols[0]:
        # Gender fairness
        gender_fairness = dashboard_data.groupby('gender')['loan_approved'].agg(['mean', 'count']).reset_index()
        gender_fairness.columns = ['Gender', 'Approval_Rate', 'Count']
        
        fig = px.bar(
            gender_fairness,
            x='Gender',
            y='Approval_Rate',
            title='Loan Approval Rate by Gender',
            labels={'Approval_Rate': 'Approval Rate'},
            color='Approval_Rate',
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Add fairness metrics
        overall_rate = dashboard_data['loan_approved'].mean()
        male_rate = dashboard_data[dashboard_data['gender'] == 'Male']['loan_approved'].mean()
        female_rate = dashboard_data[dashboard_data['gender'] == 'Female']['loan_approved'].mean()
        
        gender_parity = abs(male_rate - female_rate)
        
        st.metric(
            "Gender Demographic Parity",
            f"{gender_parity:.3f}",
            delta=f"{'Good' if gender_parity < 0.05 else 'Needs Attention'}",
            help="Difference in approval rates between genders (lower is better)"
        )
    
    with fairness_cols[1]:
        # Age group fairness
        dashboard_data['age_group'] = pd.cut(
            dashboard_data['age'], 
            bins=[18, 30, 40, 50, 70], 
            labels=['18-30', '31-40', '41-50', '51-70']
        )
        
        age_fairness = dashboard_data.groupby('age_group')['loan_approved'].agg(['mean', 'count']).reset_index()
        age_fairness.columns = ['Age_Group', 'Approval_Rate', 'Count']
        
        fig = px.bar(
            age_fairness,
            x='Age_Group',
            y='Approval_Rate',
            title='Loan Approval Rate by Age Group',
            labels={'Approval_Rate': 'Approval Rate', 'Age_Group': 'Age Group'},
            color='Approval_Rate',
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Age fairness metric
        age_rates = age_fairness['Approval_Rate'].values
        age_parity = max(age_rates) - min(age_rates)
        
        st.metric(
            "Age Group Demographic Parity",
            f"{age_parity:.3f}",
            delta=f"{'Good' if age_parity < 0.1 else 'Needs Attention'}",
            help="Range of approval rates across age groups (lower is better)"
        )
    
    # Risk Distribution Analysis
    st.subheader("⚠️ Risk Distribution Analysis")
    
    # Create risk categories
    dashboard_data['risk_category'] = pd.cut(
        dashboard_data['credit_score'],
        bins=[0, 550, 650, 750, 850],
        labels=['Very High Risk', 'High Risk', 'Medium Risk', 'Low Risk']
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk distribution pie chart
        risk_counts = dashboard_data['risk_category'].value_counts()
        fig = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title='Portfolio Risk Distribution'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Risk vs income relationship
        fig = px.box(
            dashboard_data,
            x='risk_category',
            y='monthly_income',
            title='Income Distribution by Risk Category'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Trends and Time Series (simulated)
    st.subheader("📅 Application Trends")
    
    # Simulate monthly application trends
    from datetime import datetime, timedelta
    
    monthly_data = []
    base_date = datetime.now() - timedelta(days=365)
    
    for i in range(12):
        month_date = base_date + timedelta(days=i*30)
        month_applications = np.random.poisson(100) + 80
        month_approvals = int(month_applications * (0.65 + np.random.normal(0, 0.1)))
        
        monthly_data.append({
            'Month': month_date.strftime('%Y-%m'),
            'Applications': month_applications,
            'Approvals': month_approvals,
            'Approval_Rate': month_approvals / month_applications
        })
    
    trends_df = pd.DataFrame(monthly_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=trends_df['Month'],
            y=trends_df['Applications'],
            mode='lines+markers',
            name='Applications',
            line=dict(color='blue', width=3)
        ))
        fig.add_trace(go.Scatter(
            x=trends_df['Month'],
            y=trends_df['Approvals'],
            mode='lines+markers',
            name='Approvals',
            line=dict(color='green', width=3)
        ))
        fig.update_layout(
            title='Monthly Applications and Approvals Trend',
            xaxis_title='Month',
            yaxis_title='Count'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.line(
            trends_df,
            x='Month',
            y='Approval_Rate',
            title='Monthly Approval Rate Trend',
            labels={'Approval_Rate': 'Approval Rate'}
        )
        fig.update_traces(line_color='orange', line_width=4)
        st.plotly_chart(fig, use_container_width=True)

# Sidebar insights
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Dashboard Insights")

if 'dashboard_data' in locals():
    st.sidebar.markdown(f"""
    **Key Statistics:**
    - Total Applications: {len(dashboard_data):,}
    - Approval Rate: {dashboard_data['loan_approved'].mean():.1%}
    - Average Credit Score: {dashboard_data['credit_score'].mean():.0f}
    
    **Top Performing Region:**
    {dashboard_data.groupby('region')['loan_approved'].mean().idxmax()}
    
    **Most Common Business Type:**
    {dashboard_data['business_type'].mode()[0]}
    """)

st.sidebar.markdown("---")
st.sidebar.subheader("🔄 Refresh Data")
if st.sidebar.button("Generate New Dashboard Data"):
    st.rerun()

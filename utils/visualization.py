import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def create_visualizations(data, prediction_results=None):
    """Create comprehensive visualizations for the credit scoring system"""
    
    visualizations = {}
    
    # 1. Credit Score Distribution
    fig_score_dist = px.histogram(
        data, 
        x='credit_score', 
        nbins=30,
        title='Credit Score Distribution',
        labels={'credit_score': 'Credit Score', 'count': 'Number of Applicants'},
        color_discrete_sequence=['#1f77b4']
    )
    fig_score_dist.add_vline(x=650, line_dash="dash", line_color="red", 
                            annotation_text="Approval Threshold")
    visualizations['score_distribution'] = fig_score_dist
    
    # 2. Approval Rate by Region
    approval_by_region = data.groupby('region')['loan_approved'].agg(['mean', 'count']).reset_index()
    approval_by_region.columns = ['region', 'approval_rate', 'total_applications']
    
    fig_approval = px.bar(
        approval_by_region, 
        x='region', 
        y='approval_rate',
        title='Loan Approval Rate by Region',
        labels={'approval_rate': 'Approval Rate', 'region': 'Region'},
        color='approval_rate',
        color_continuous_scale='RdYlGn'
    )
    visualizations['approval_by_region'] = fig_approval
    
    # 3. Income vs Credit Score Scatter
    fig_income_score = px.scatter(
        data, 
        x='monthly_income', 
        y='credit_score',
        color='loan_approved',
        size='business_age',
        hover_data=['business_type', 'region'],
        title='Monthly Income vs Credit Score',
        labels={
            'monthly_income': 'Monthly Income',
            'credit_score': 'Credit Score',
            'loan_approved': 'Loan Status'
        }
    )
    visualizations['income_vs_score'] = fig_income_score
    
    # 4. Alternative Data Scores Radar Chart
    avg_scores = data.groupby('loan_approved')[
        ['mobile_usage_score', 'mobile_money_score', 'utility_payment_score', 'cooperative_score']
    ].mean()
    
    categories = ['Mobile Usage', 'Mobile Money', 'Utility Payments', 'Cooperative Activity']
    
    fig_radar = go.Figure()
    
    # Approved loans
    fig_radar.add_trace(go.Scatterpolar(
        r=avg_scores.loc[True].values.tolist() + [avg_scores.loc[True].values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name='Approved Loans',
        line_color='green'
    ))
    
    # Rejected loans
    fig_radar.add_trace(go.Scatterpolar(
        r=avg_scores.loc[False].values.tolist() + [avg_scores.loc[False].values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name='Rejected Loans',
        line_color='red'
    ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title='Alternative Data Scores: Approved vs Rejected Loans'
    )
    visualizations['radar_chart'] = fig_radar
    
    # 5. Business Type Analysis
    business_analysis = data.groupby('business_type').agg({
        'loan_approved': ['mean', 'count'],
        'credit_score': 'mean',
        'monthly_income': 'mean'
    }).reset_index()
    
    business_analysis.columns = ['business_type', 'approval_rate', 'count', 'avg_score', 'avg_income']
    
    fig_business = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Approval Rate', 'Average Credit Score', 
                       'Average Income', 'Application Count'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Approval rate
    fig_business.add_trace(
        go.Bar(x=business_analysis['business_type'], y=business_analysis['approval_rate'],
               name='Approval Rate', marker_color='lightblue'),
        row=1, col=1
    )
    
    # Average credit score
    fig_business.add_trace(
        go.Bar(x=business_analysis['business_type'], y=business_analysis['avg_score'],
               name='Avg Credit Score', marker_color='lightgreen'),
        row=1, col=2
    )
    
    # Average income
    fig_business.add_trace(
        go.Bar(x=business_analysis['business_type'], y=business_analysis['avg_income'],
               name='Avg Income', marker_color='lightyellow'),
        row=2, col=1
    )
    
    # Application count
    fig_business.add_trace(
        go.Bar(x=business_analysis['business_type'], y=business_analysis['count'],
               name='Applications', marker_color='lightcoral'),
        row=2, col=2
    )
    
    fig_business.update_layout(
        title_text='Business Type Analysis',
        showlegend=False,
        height=600
    )
    visualizations['business_analysis'] = fig_business
    
    # 6. Consistency Metrics Heatmap
    consistency_cols = ['airtime_consistency', 'repayment_consistency', 'bill_payment_consistency']
    consistency_by_approval = data.groupby('loan_approved')[consistency_cols].mean()
    
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=consistency_by_approval.values,
        x=['Airtime Consistency', 'Repayment Consistency', 'Bill Payment Consistency'],
        y=['Rejected', 'Approved'],
        colorscale='RdYlGn',
        text=np.round(consistency_by_approval.values, 3),
        texttemplate="%{text}",
        textfont={"size": 12}
    ))
    
    fig_heatmap.update_layout(
        title='Payment Consistency by Loan Status',
        xaxis_title='Consistency Metrics',
        yaxis_title='Loan Status'
    )
    visualizations['consistency_heatmap'] = fig_heatmap
    
    # 7. Age and Business Age Distribution
    fig_age_dist = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Age Distribution', 'Business Age Distribution')
    )
    
    fig_age_dist.add_trace(
        go.Histogram(x=data['age'], name='Age', nbinsx=20, marker_color='skyblue'),
        row=1, col=1
    )
    
    fig_age_dist.add_trace(
        go.Histogram(x=data['business_age'], name='Business Age', nbinsx=20, marker_color='lightcoral'),
        row=1, col=2
    )
    
    fig_age_dist.update_layout(
        title_text='Age Distributions',
        showlegend=False
    )
    visualizations['age_distributions'] = fig_age_dist
    
    # 8. Feature Correlation Matrix (if prediction results available)
    if prediction_results is not None and 'feature_importance' in prediction_results:
        feature_importance = prediction_results['feature_importance']
        
        fig_importance = px.bar(
            feature_importance.head(10),
            x='importance',
            y='feature',
            orientation='h',
            title='Top 10 Feature Importance',
            labels={'importance': 'Importance Score', 'feature': 'Features'}
        )
        fig_importance.update_layout(yaxis={'categoryorder': 'total ascending'})
        visualizations['feature_importance'] = fig_importance
    
    return visualizations

def create_individual_prediction_viz(applicant_data, prediction_result, shap_values=None):
    """Create visualizations for individual prediction explanation"""
    
    individual_viz = {}
    
    # 1. Credit Score Gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prediction_result['credit_score'],
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Credit Score"},
        delta={'reference': 650},
        gauge={
            'axis': {'range': [300, 850]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [300, 550], 'color': "lightgray"},
                {'range': [550, 650], 'color': "yellow"},
                {'range': [650, 750], 'color': "lightgreen"},
                {'range': [750, 850], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 650
            }
        }
    ))
    individual_viz['credit_score_gauge'] = fig_gauge
    
    # 2. Risk Category Indicator
    risk_category = prediction_result['risk_category']
    risk_colors = {
        'Low Risk': 'green',
        'Medium Risk': 'yellow',
        'High Risk': 'orange',
        'Very High Risk': 'red'
    }
    
    fig_risk = go.Figure(go.Indicator(
        mode="number",
        value=1,
        title={'text': f"Risk Category: {risk_category}"},
        number={'font': {'color': risk_colors.get(risk_category, 'blue')}}
    ))
    individual_viz['risk_indicator'] = fig_risk
    
    # 3. Applicant Profile Summary
    profile_data = {
        'Metric': ['Age', 'Monthly Income', 'Business Age', 'Mobile Usage Score', 
                  'Mobile Money Score', 'Utility Payment Score'],
        'Value': [
            applicant_data.get('age', 'N/A'),
            f"${applicant_data.get('monthly_income', 0):,.0f}",
            f"{applicant_data.get('business_age', 0):.1f} years",
            f"{applicant_data.get('mobile_usage_score', 0):.1f}",
            f"{applicant_data.get('mobile_money_score', 0):.1f}",
            f"{applicant_data.get('utility_payment_score', 0):.1f}"
        ]
    }
    
    fig_profile = go.Figure(data=go.Table(
        header=dict(values=['Metric', 'Value'],
                   fill_color='paleturquoise',
                   align='left'),
        cells=dict(values=[profile_data['Metric'], profile_data['Value']],
                  fill_color='lavender',
                  align='left')
    ))
    fig_profile.update_layout(title='Applicant Profile Summary')
    individual_viz['profile_summary'] = fig_profile
    
    return individual_viz

def create_fairness_analysis_viz(data, protected_attributes=['gender', 'region']):
    """Create visualizations for fairness analysis"""
    
    fairness_viz = {}
    
    for attr in protected_attributes:
        if attr in data.columns:
            # Approval rate by protected attribute
            approval_by_attr = data.groupby(attr)['loan_approved'].mean().reset_index()
            
            fig = px.bar(
                approval_by_attr,
                x=attr,
                y='loan_approved',
                title=f'Loan Approval Rate by {attr.title()}',
                labels={'loan_approved': 'Approval Rate', attr: attr.title()}
            )
            
            # Add horizontal line for overall approval rate
            overall_rate = data['loan_approved'].mean()
            fig.add_hline(y=overall_rate, line_dash="dash", line_color="red",
                         annotation_text=f"Overall Rate: {overall_rate:.2%}")
            
            fairness_viz[f'approval_by_{attr}'] = fig
    
    return fairness_viz

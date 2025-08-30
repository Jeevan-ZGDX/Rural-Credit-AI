import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import xgboost as xgb

st.title("🤖 Model Training & Evaluation")
st.markdown("### Train and Evaluate Credit Scoring Models")

# Model training controls
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Training Configuration")
    
    # Data generation settings
    sample_size = st.slider("Training Data Size", 100, 5000, 1000, step=100)
    test_size = st.slider("Test Set Size (%)", 10, 40, 20, step=5)
    random_state = st.number_input("Random State", value=42, help="For reproducible results")

with col2:
    st.subheader("🎯 Model Selection")
    
    selected_models = st.multiselect(
        "Select Models to Train",
        ["Logistic Regression", "Random Forest", "XGBoost"],
        default=["Logistic Regression", "Random Forest", "XGBoost"]
    )

# Training button
if st.button("🚀 Train Models", type="primary"):
    if not selected_models:
        st.error("Please select at least one model to train.")
    else:
        # Generate training data
        with st.spinner("Generating training data..."):
            training_data = st.session_state.data_generator.generate_sample_data(sample_size)
            
        st.success(f"✅ Generated {len(training_data)} training samples")
        
        # Display data overview
        st.subheader("📈 Training Data Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Samples", len(training_data))
        with col2:
            approval_rate = training_data['loan_approved'].mean()
            st.metric("Approval Rate", f"{approval_rate:.1%}")
        with col3:
            avg_score = training_data['credit_score'].mean()
            st.metric("Avg Credit Score", f"{avg_score:.0f}")
        with col4:
            score_std = training_data['credit_score'].std()
            st.metric("Score Std Dev", f"{score_std:.0f}")
        
        # Data distribution visualization
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(training_data, x='credit_score', nbins=30, 
                             title="Credit Score Distribution")
            fig.add_vline(x=650, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            region_counts = training_data['region'].value_counts()
            fig = px.pie(values=region_counts.values, names=region_counts.index,
                        title="Applications by Region")
            st.plotly_chart(fig, use_container_width=True)
        
        # Prepare data for training
        with st.spinner("Preparing data for model training..."):
            X_train, X_test, y_train, y_test = st.session_state.model.prepare_data(training_data)
            
        st.info(f"Training set: {len(X_train)} samples | Test set: {len(X_test)} samples")
        
        # Train models
        with st.spinner("Training machine learning models..."):
            progress_bar = st.progress(0)
            
            models_trained = {}
            
            for i, model_name in enumerate(selected_models):
                st.write(f"Training {model_name}...")
                
                if model_name == "Logistic Regression":
                    from sklearn.linear_model import LogisticRegression
                    model = LogisticRegression(random_state=random_state, max_iter=1000)
                    model.fit(X_train, y_train)
                    models_trained[model_name] = model
                    
                elif model_name == "Random Forest":
                    from sklearn.ensemble import RandomForestClassifier
                    model = RandomForestClassifier(
                        n_estimators=100,
                        random_state=random_state,
                        max_depth=10,
                        min_samples_split=5
                    )
                    model.fit(X_train, y_train)
                    models_trained[model_name] = model
                    
                elif model_name == "XGBoost":
                    import xgboost as xgb
                    model = xgb.XGBClassifier(
                        random_state=random_state,
                        max_depth=6,
                        learning_rate=0.1,
                        n_estimators=100
                    )
                    model.fit(X_train, y_train)
                    models_trained[model_name] = model
                
                progress_bar.progress((i + 1) / len(selected_models))
            
            # Update session state
            st.session_state.model.models = models_trained
            
        st.success("✅ Model training completed successfully!")
        
        # Model evaluation
        st.subheader("🎯 Model Performance Evaluation")
        
        performance_data = []
        
        for model_name, model in models_trained.items():
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            performance_data.append({
                'Model': model_name,
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred),
                'Recall': recall_score(y_test, y_pred),
                'F1-Score': f1_score(y_test, y_pred),
                'AUC-ROC': roc_auc_score(y_test, y_pred_proba)
            })
        
        performance_df = pd.DataFrame(performance_data)
        
        # Display performance metrics
        st.dataframe(performance_df.round(4), use_container_width=True)
        
        # Performance comparison visualization
        col1, col2 = st.columns(2)
        
        with col1:
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
            fig = go.Figure()
            
            for metric in metrics:
                fig.add_trace(go.Scatter(
                    x=performance_df['Model'],
                    y=performance_df[metric],
                    mode='lines+markers',
                    name=metric,
                    line=dict(width=2),
                    marker=dict(size=8)
                ))
            
            fig.update_layout(
                title="Model Performance Comparison",
                xaxis_title="Model",
                yaxis_title="Score",
                yaxis=dict(range=[0, 1]),
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # ROC Curves
            from sklearn.metrics import roc_curve
            
            fig = go.Figure()
            
            for model_name, model in models_trained.items():
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                auc_score = roc_auc_score(y_test, y_pred_proba)
                
                fig.add_trace(go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode='lines',
                    name=f'{model_name} (AUC: {auc_score:.3f})',
                    line=dict(width=2)
                ))
            
            # Add diagonal line
            fig.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='Random Classifier',
                line=dict(dash='dash', color='gray')
            ))
            
            fig.update_layout(
                title="ROC Curves",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1])
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance analysis
        st.subheader("🔍 Feature Importance Analysis")
        
        # Get feature importance from Random Forest if available
        if "Random Forest" in models_trained:
            model = models_trained["Random Forest"]
            feature_names = st.session_state.model.feature_names
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Top 10 features bar chart
                fig = px.bar(
                    importance_df.head(10),
                    x='importance',
                    y='feature',
                    orientation='h',
                    title='Top 10 Feature Importance (Random Forest)',
                    labels={'importance': 'Importance Score'}
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Feature importance table
                st.markdown("**Feature Importance Rankings**")
                st.dataframe(
                    importance_df.round(4),
                    use_container_width=True,
                    height=400
                )
        
        # Confusion matrices
        st.subheader("🎯 Confusion Matrices")
        
        cols = st.columns(len(models_trained))
        
        for idx, (model_name, model) in enumerate(models_trained.items()):
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            
            with cols[idx]:
                fig = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=['Rejected', 'Approved'],
                    y=['Rejected', 'Approved'],
                    colorscale='Blues',
                    text=cm,
                    texttemplate="%{text}",
                    textfont={"size": 16}
                ))
                
                fig.update_layout(
                    title=f'{model_name}<br>Confusion Matrix',
                    xaxis_title='Predicted',
                    yaxis_title='Actual',
                    width=300,
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Model insights
        st.subheader("💡 Key Insights")
        
        best_model = performance_df.loc[performance_df['F1-Score'].idxmax(), 'Model']
        best_f1 = performance_df.loc[performance_df['F1-Score'].idxmax(), 'F1-Score']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info(f"**Best Performing Model**\n\n{best_model}\n\nF1-Score: {best_f1:.3f}")
        
        with col2:
            avg_approval_rate = training_data['loan_approved'].mean()
            st.info(f"**Dataset Balance**\n\nApproval Rate: {avg_approval_rate:.1%}\n\nRejection Rate: {1-avg_approval_rate:.1%}")
        
        with col3:
            if "Random Forest" in models_trained:
                top_feature = importance_df.iloc[0]['feature']
                top_importance = importance_df.iloc[0]['importance']
                st.info(f"**Most Important Feature**\n\n{top_feature}\n\nImportance: {top_importance:.3f}")
        
        # Save models option
        st.subheader("💾 Model Management")
        
        if st.button("Save Trained Models"):
            try:
                st.session_state.model.save_model('trained_models.joblib')
                st.success("✅ Models saved successfully!")
            except Exception as e:
                st.error(f"Error saving models: {str(e)}")

# Display current model status
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Model Status")

if hasattr(st.session_state, 'model') and st.session_state.model.models:
    st.sidebar.success(f"✅ {len(st.session_state.model.models)} models trained")
    for model_name in st.session_state.model.models.keys():
        st.sidebar.write(f"• {model_name}")
else:
    st.sidebar.warning("⚠️ No models trained yet")

st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ Training Information")
st.sidebar.markdown("""
**Model Types:**
- **Logistic Regression**: Fast, interpretable baseline
- **Random Forest**: Good performance, feature importance
- **XGBoost**: Advanced gradient boosting

**Evaluation Metrics:**
- **Accuracy**: Overall correctness
- **Precision**: Approved loan accuracy
- **Recall**: Catching qualified applicants
- **F1-Score**: Balanced performance
- **AUC-ROC**: Overall discrimination ability
""")

import pandas as pd
import numpy as np
# import shap  # Commented out for now
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

def explain_prediction(model, instance, X_train, feature_names, method='feature_importance'):
    """
    Explain individual predictions using feature importance method
    """
    try:
        # For now, use feature importance method instead of SHAP
        return explain_with_feature_importance(model, instance, feature_names)
            
    except Exception as e:
        # Fallback to simpler explanation method
        return explain_with_feature_importance(model, instance, feature_names)

def explain_with_feature_importance(model, instance, feature_names):
    """
    Fallback explanation using feature importance and feature values
    """
    try:
        # Get prediction
        prediction = model.predict(instance)[0]
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(instance)[0, 1]
        else:
            probability = prediction
        
        # Get feature importance if available
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
        else:
            # Use uniform importance as last resort
            importance = np.ones(len(feature_names)) / len(feature_names)
        
        # Calculate impact as importance * normalized feature value
        instance_values = instance.iloc[0].values
        normalized_values = (instance_values - instance_values.mean()) / (instance_values.std() + 1e-8)
        
        feature_impact = pd.DataFrame({
            'feature': feature_names,
            'value': instance_values,
            'importance': importance,
            'normalized_value': normalized_values,
            'impact_score': importance * normalized_values,
            'abs_impact': np.abs(importance * normalized_values)
        }).sort_values('abs_impact', ascending=False)
        
        return {
            'prediction': prediction,
            'probability': probability,
            'feature_impact': feature_impact,
            'method': 'Feature Importance'
        }
        
    except Exception as e:
        # Ultimate fallback
        return {
            'prediction': 0,
            'probability': 0.5,
            'feature_impact': pd.DataFrame({
                'feature': feature_names,
                'value': instance.iloc[0].values,
                'impact_score': np.zeros(len(feature_names))
            }),
            'method': 'Fallback',
            'error': str(e)
        }

def generate_explanation_text(explanation_result):
    """
    Generate human-readable explanation text
    """
    prediction = explanation_result['prediction']
    probability = explanation_result['probability']
    feature_impact = explanation_result['feature_impact']
    
    # Determine loan decision
    decision = "APPROVED" if prediction == 1 else "REJECTED"
    confidence = "High" if abs(probability - 0.5) > 0.3 else "Medium" if abs(probability - 0.5) > 0.15 else "Low"
    
    # Get top positive and negative factors
    positive_factors = feature_impact[feature_impact.get('shap_value', feature_impact.get('impact_score', 0)) > 0].head(3)
    negative_factors = feature_impact[feature_impact.get('shap_value', feature_impact.get('impact_score', 0)) < 0].head(3)
    
    explanation_text = f"""
    **Loan Decision: {decision}**
    **Confidence Level: {confidence}**
    **Approval Probability: {probability:.1%}**
    
    **Key Positive Factors:**
    """
    
    for _, factor in positive_factors.iterrows():
        explanation_text += f"\n• {factor['feature']}: {factor['value']:.2f}"
    
    explanation_text += "\n\n**Key Risk Factors:**"
    
    for _, factor in negative_factors.iterrows():
        explanation_text += f"\n• {factor['feature']}: {factor['value']:.2f}"
    
    explanation_text += f"""
    
    **Explanation Method: {explanation_result['method']}**
    
    This assessment is based on alternative data sources including mobile usage patterns, 
    payment history, and community involvement, specifically designed for rural small 
    businesses with limited traditional credit history.
    """
    
    return explanation_text

def calculate_fairness_metrics(y_true, y_pred, sensitive_attribute):
    """
    Calculate basic fairness metrics
    """
    metrics = {}
    
    # Overall accuracy
    overall_accuracy = (y_true == y_pred).mean()
    metrics['overall_accuracy'] = overall_accuracy
    
    # Group-specific accuracies
    for group in sensitive_attribute.unique():
        mask = sensitive_attribute == group
        if mask.sum() > 0:
            group_accuracy = (y_true[mask] == y_pred[mask]).mean()
            metrics[f'{group}_accuracy'] = group_accuracy
    
    # Demographic parity (equal approval rates)
    overall_approval_rate = y_pred.mean()
    metrics['overall_approval_rate'] = overall_approval_rate
    
    for group in sensitive_attribute.unique():
        mask = sensitive_attribute == group
        if mask.sum() > 0:
            group_approval_rate = y_pred[mask].mean()
            metrics[f'{group}_approval_rate'] = group_approval_rate
            metrics[f'{group}_demographic_parity'] = abs(group_approval_rate - overall_approval_rate)
    
    # Equal opportunity (equal true positive rates)
    if y_true.sum() > 0:  # If there are positive cases
        overall_tpr = ((y_true == 1) & (y_pred == 1)).sum() / (y_true == 1).sum()
        metrics['overall_tpr'] = overall_tpr
        
        for group in sensitive_attribute.unique():
            mask = sensitive_attribute == group
            group_positives = (y_true[mask] == 1).sum()
            if group_positives > 0:
                group_tpr = ((y_true[mask] == 1) & (y_pred[mask] == 1)).sum() / group_positives
                metrics[f'{group}_tpr'] = group_tpr
                metrics[f'{group}_equal_opportunity'] = abs(group_tpr - overall_tpr)
    
    return metrics

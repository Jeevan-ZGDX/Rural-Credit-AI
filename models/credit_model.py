import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

class CreditScoringModel:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        
    def prepare_data(self, df):
        """Prepare data for model training"""
        # Create a copy to avoid modifying original data
        data = df.copy()
        
        # Encode categorical variables
        categorical_columns = ['region', 'business_type', 'education_level', 'gender']
        
        for col in categorical_columns:
            if col in data.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    data[col] = self.label_encoders[col].fit_transform(data[col])
                else:
                    data[col] = self.label_encoders[col].transform(data[col])
        
        # Define features and target
        feature_columns = [
            'age', 'monthly_income', 'business_age', 'mobile_usage_score',
            'mobile_money_score', 'utility_payment_score', 'cooperative_score',
            'airtime_consistency', 'call_frequency', 'sms_frequency', 
            'data_usage_trend', 'transaction_frequency', 'avg_transaction_amount',
            'repayment_consistency', 'bill_payment_consistency', 'savings_balance',
            'region', 'business_type', 'education_level', 'gender'
        ]
        
        # Filter available columns
        available_features = [col for col in feature_columns if col in data.columns]
        
        X = data[available_features]
        y = data['loan_approved'] if 'loan_approved' in data.columns else data['credit_score'] > 650
        
        self.feature_names = available_features
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale numerical features
        numerical_features = X_train.select_dtypes(include=[np.number]).columns
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        X_train_scaled[numerical_features] = self.scaler.fit_transform(X_train[numerical_features])
        X_test_scaled[numerical_features] = self.scaler.transform(X_test[numerical_features])
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_models(self, X_train, y_train):
        """Train multiple ML models"""
        
        # Logistic Regression
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_model.fit(X_train, y_train)
        self.models['Logistic Regression'] = lr_model
        
        # Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=100, 
            random_state=42, 
            max_depth=10,
            min_samples_split=5
        )
        rf_model.fit(X_train, y_train)
        self.models['Random Forest'] = rf_model
        
        # XGBoost
        xgb_model = xgb.XGBClassifier(
            random_state=42,
            max_depth=6,
            learning_rate=0.1,
            n_estimators=100
        )
        xgb_model.fit(X_train, y_train)
        self.models['XGBoost'] = xgb_model
        
        return self.models
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models"""
        results = {}
        
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            results[name] = {
                'accuracy': np.mean(y_pred == y_test),
                'auc_score': roc_auc_score(y_test, y_pred_proba),
                'classification_report': classification_report(y_test, y_pred, output_dict=True),
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
        
        return results
    
    def predict_credit_score(self, applicant_data, model_name='Random Forest'):
        """Predict credit score for a single applicant"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
        
        model = self.models[model_name]
        
        # Prepare applicant data
        processed_data = self.preprocess_single_applicant(applicant_data)
        
        # Make prediction
        probability = model.predict_proba(processed_data)[0, 1]
        prediction = model.predict(processed_data)[0]
        
        # Convert to credit score (300-850 range)
        credit_score = int(300 + (probability * 550))
        
        return {
            'credit_score': credit_score,
            'probability': probability,
            'approved': prediction,
            'risk_category': self.get_risk_category(credit_score)
        }
    
    def preprocess_single_applicant(self, applicant_data):
        """Preprocess data for a single applicant"""
        # Convert to DataFrame
        if isinstance(applicant_data, dict):
            df = pd.DataFrame([applicant_data])
        else:
            df = applicant_data.copy()
        
        # Encode categorical variables
        categorical_columns = ['region', 'business_type', 'education_level', 'gender']
        
        for col in categorical_columns:
            if col in df.columns and col in self.label_encoders:
                try:
                    df[col] = self.label_encoders[col].transform(df[col])
                except ValueError:
                    # Handle unseen categories
                    df[col] = 0
        
        # Ensure all required features are present
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0
        
        # Select and order features
        df = df[self.feature_names]
        
        # Scale numerical features
        numerical_features = df.select_dtypes(include=[np.number]).columns
        df[numerical_features] = self.scaler.transform(df[numerical_features])
        
        return df
    
    def get_risk_category(self, credit_score):
        """Categorize risk based on credit score"""
        if credit_score >= 750:
            return "Low Risk"
        elif credit_score >= 650:
            return "Medium Risk"
        elif credit_score >= 550:
            return "High Risk"
        else:
            return "Very High Risk"
    
    def get_feature_importance(self, model_name='Random Forest'):
        """Get feature importance for interpretation"""
        if model_name not in self.models:
            return None
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        
        return None
    
    def save_model(self, filepath):
        """Save trained models"""
        model_data = {
            'models': self.models,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath):
        """Load trained models"""
        model_data = joblib.load(filepath)
        self.models = model_data['models']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_names = model_data['feature_names']

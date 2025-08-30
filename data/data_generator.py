import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class DataGenerator:
    def __init__(self):
        self.regions = ['Northern', 'Southern', 'Eastern', 'Western', 'Central']
        self.business_types = ['Agriculture', 'Retail', 'Services', 'Manufacturing', 'Trading']
        self.education_levels = ['Primary', 'Secondary', 'Vocational', 'University']
        self.gender_options = ['Male', 'Female']
        
    def generate_sample_data(self, num_samples=1000):
        """Generate realistic sample data for rural small businesses"""
        
        np.random.seed(42)  # For reproducibility
        
        data = []
        
        for i in range(num_samples):
            # Basic demographics
            age = np.random.normal(38, 12)
            age = max(18, min(70, age))  # Ensure reasonable age range
            
            gender = np.random.choice(self.gender_options)
            region = np.random.choice(self.regions)
            business_type = np.random.choice(self.business_types)
            education_level = np.random.choice(self.education_levels)
            
            # Business characteristics
            business_age = np.random.exponential(3)  # Most businesses are relatively new
            business_age = max(0.1, min(20, business_age))
            
            # Income based on business type and region (in local currency)
            base_income = {
                'Agriculture': 25000,
                'Retail': 35000,
                'Services': 40000,
                'Manufacturing': 50000,
                'Trading': 45000
            }[business_type]
            
            region_multiplier = {
                'Northern': 0.8,
                'Southern': 1.2,
                'Eastern': 0.9,
                'Western': 1.1,
                'Central': 1.3
            }[region]
            
            monthly_income = base_income * region_multiplier * (1 + np.random.normal(0, 0.3))
            monthly_income = max(10000, monthly_income)
            
            # Mobile phone usage patterns
            mobile_usage_score = self.generate_mobile_usage_score(age, education_level, region)
            airtime_consistency = np.random.beta(2, 2)  # Consistency in airtime top-ups
            call_frequency = np.random.poisson(25)  # Calls per day
            sms_frequency = np.random.poisson(15)   # SMS per day
            data_usage_trend = np.random.normal(0.5, 0.2)  # Data usage growth trend
            
            # Mobile money transactions
            mobile_money_score = self.generate_mobile_money_score(monthly_income, business_type)
            transaction_frequency = np.random.poisson(8)  # Transactions per week
            avg_transaction_amount = monthly_income * np.random.uniform(0.1, 0.5)
            repayment_consistency = np.random.beta(3, 1.5)  # Higher values = better consistency
            
            # Utility payments
            utility_payment_score = self.generate_utility_score(monthly_income, region)
            bill_payment_consistency = np.random.beta(2.5, 1.5)
            
            # Community/cooperative involvement
            cooperative_score = self.generate_cooperative_score(business_type, region)
            savings_balance = monthly_income * np.random.uniform(0.5, 3.0)
            
            # Calculate overall credit score
            credit_score = self.calculate_credit_score(
                age, monthly_income, business_age, mobile_usage_score,
                mobile_money_score, utility_payment_score, cooperative_score,
                airtime_consistency, repayment_consistency, bill_payment_consistency
            )
            
            # Loan approval decision (with some randomness)
            approval_threshold = 600 + np.random.normal(0, 50)
            loan_approved = credit_score > approval_threshold
            
            data.append({
                'applicant_id': f'APP_{i+1:04d}',
                'age': int(age),
                'gender': gender,
                'region': region,
                'business_type': business_type,
                'education_level': education_level,
                'business_age': round(business_age, 1),
                'monthly_income': round(monthly_income, 2),
                'mobile_usage_score': round(mobile_usage_score, 2),
                'airtime_consistency': round(airtime_consistency, 3),
                'call_frequency': call_frequency,
                'sms_frequency': sms_frequency,
                'data_usage_trend': round(data_usage_trend, 3),
                'mobile_money_score': round(mobile_money_score, 2),
                'transaction_frequency': transaction_frequency,
                'avg_transaction_amount': round(avg_transaction_amount, 2),
                'repayment_consistency': round(repayment_consistency, 3),
                'utility_payment_score': round(utility_payment_score, 2),
                'bill_payment_consistency': round(bill_payment_consistency, 3),
                'cooperative_score': round(cooperative_score, 2),
                'savings_balance': round(savings_balance, 2),
                'credit_score': int(credit_score),
                'loan_approved': loan_approved,
                'created_at': datetime.now() - timedelta(days=random.randint(0, 365))
            })
        
        return pd.DataFrame(data)
    
    def generate_mobile_usage_score(self, age, education_level, region):
        """Generate mobile usage score based on demographics"""
        base_score = 0.5
        
        # Age factor (younger people tend to use mobile more)
        if age < 30:
            base_score += 0.2
        elif age > 50:
            base_score -= 0.1
        
        # Education factor
        education_bonus = {
            'Primary': 0,
            'Secondary': 0.1,
            'Vocational': 0.15,
            'University': 0.25
        }[education_level]
        
        # Region factor (urban vs rural infrastructure)
        region_bonus = {
            'Northern': -0.1,
            'Southern': 0.1,
            'Eastern': -0.05,
            'Western': 0.05,
            'Central': 0.2
        }[region]
        
        score = base_score + education_bonus + region_bonus + np.random.normal(0, 0.1)
        return max(0.1, min(1.0, score)) * 100
    
    def generate_mobile_money_score(self, income, business_type):
        """Generate mobile money usage score"""
        base_score = 0.4
        
        # Income factor
        if income > 50000:
            base_score += 0.3
        elif income > 30000:
            base_score += 0.2
        
        # Business type factor
        business_bonus = {
            'Agriculture': 0.1,
            'Retail': 0.3,
            'Services': 0.2,
            'Manufacturing': 0.15,
            'Trading': 0.35
        }[business_type]
        
        score = base_score + business_bonus + np.random.normal(0, 0.15)
        return max(0.1, min(1.0, score)) * 100
    
    def generate_utility_score(self, income, region):
        """Generate utility payment score"""
        base_score = 0.6
        
        # Income reliability
        if income > 40000:
            base_score += 0.2
        elif income < 20000:
            base_score -= 0.2
        
        # Regional infrastructure
        region_factor = {
            'Northern': -0.15,
            'Southern': 0.05,
            'Eastern': -0.1,
            'Western': 0,
            'Central': 0.15
        }[region]
        
        score = base_score + region_factor + np.random.normal(0, 0.1)
        return max(0.1, min(1.0, score)) * 100
    
    def generate_cooperative_score(self, business_type, region):
        """Generate community/cooperative involvement score"""
        base_score = 0.3
        
        # Business type factor (some types more community-oriented)
        business_factor = {
            'Agriculture': 0.4,  # Farmers often in cooperatives
            'Retail': 0.2,
            'Services': 0.1,
            'Manufacturing': 0.15,
            'Trading': 0.25
        }[business_type]
        
        # Regional community strength
        region_factor = {
            'Northern': 0.3,
            'Southern': 0.2,
            'Eastern': 0.35,
            'Western': 0.25,
            'Central': 0.1
        }[region]
        
        score = base_score + business_factor + region_factor + np.random.normal(0, 0.15)
        return max(0.1, min(1.0, score)) * 100
    
    def calculate_credit_score(self, age, income, business_age, mobile_score, 
                             money_score, utility_score, coop_score,
                             airtime_consistency, repayment_consistency, 
                             bill_consistency):
        """Calculate overall credit score"""
        
        # Base score
        base_score = 400
        
        # Age factor (optimal age range for business)
        if 25 <= age <= 50:
            age_score = 100
        elif 20 <= age <= 60:
            age_score = 80
        else:
            age_score = 50
        
        # Income score (normalized)
        income_score = min(150, (income / 1000) * 2)
        
        # Business experience
        business_score = min(80, business_age * 10)
        
        # Alternative data scores
        alt_data_score = (
            mobile_score * 0.3 +
            money_score * 0.35 +
            utility_score * 0.2 +
            coop_score * 0.15
        )
        
        # Consistency bonuses
        consistency_bonus = (
            airtime_consistency * 30 +
            repayment_consistency * 50 +
            bill_consistency * 40
        )
        
        total_score = (
            base_score + age_score + income_score + 
            business_score + alt_data_score + consistency_bonus
        )
        
        # Add some randomness
        total_score += np.random.normal(0, 25)
        
        return max(300, min(850, total_score))
    
    def generate_single_applicant(self, **kwargs):
        """Generate data for a single applicant with optional overrides"""
        sample = self.generate_sample_data(1)
        
        # Override with provided values
        for key, value in kwargs.items():
            if key in sample.columns:
                sample[key] = value
        
        return sample.iloc[0].to_dict()

# Overview

This is a Rural Small Business Credit Scoring System designed to assess creditworthiness for rural entrepreneurs who often lack formal credit history. The system leverages machine learning models trained on non-traditional data sources including mobile usage patterns, utility payment history, mobile money transactions, and cooperative financial records. Built with Streamlit, it provides an interactive web interface for loan applications, model training, and comprehensive analytics dashboards.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Framework**: Streamlit-based web application with multi-page navigation
- **UI Structure**: Modular page-based design with sidebar navigation
- **Key Pages**: Home dashboard, loan application form, model training interface, analytics dashboard
- **Visualization**: Plotly for interactive charts and graphs
- **State Management**: Streamlit session state for maintaining model instances and data generators

## Backend Architecture
- **Model Framework**: Scikit-learn ecosystem with multiple ML algorithms
- **Supported Models**: Logistic Regression, Random Forest, XGBoost for credit scoring
- **Data Processing**: Pandas for data manipulation, NumPy for numerical operations
- **Feature Engineering**: Automatic encoding of categorical variables, standardization of numerical features
- **Model Persistence**: Joblib for saving and loading trained models

## Data Architecture
- **Data Generation**: Synthetic data generator creating realistic rural business profiles
- **Feature Categories**: 
  - Demographics (age, gender, education, region)
  - Business characteristics (type, age, income)
  - Alternative data sources (mobile usage, mobile money, utility payments, cooperative records)
- **Data Flow**: In-memory processing with pandas DataFrames
- **Feature Engineering**: Label encoding for categorical variables, standardization for numerical features

## Model Training Pipeline
- **Training Process**: Automated train-test split with configurable parameters
- **Model Evaluation**: Classification metrics including ROC-AUC, confusion matrix, classification reports
- **Model Comparison**: Side-by-side evaluation of multiple algorithms
- **Hyperparameter Control**: Configurable training parameters through UI

## Explainability Framework
- **SHAP Integration**: Model-agnostic explanations for individual predictions
- **Feature Impact Analysis**: Ranked feature importance with positive/negative contribution breakdown
- **Visualization**: Interactive explanation charts for transparency in credit decisions

# External Dependencies

## Core ML Libraries
- **scikit-learn**: Primary machine learning framework for model training and evaluation
- **XGBoost**: Gradient boosting implementation for advanced modeling
- **SHAP**: Model explainability and feature attribution analysis

## Web Framework
- **Streamlit**: Complete web application framework for interactive dashboards
- **Plotly**: Interactive visualization library for charts and graphs

## Data Processing
- **pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing and array operations

## Utilities
- **joblib**: Model serialization and persistence
- **warnings**: Error handling and cleanup

## Development Tools
- **datetime**: Date and time manipulation for synthetic data generation
- **random**: Pseudo-random number generation for data simulation

Note: The system currently uses synthetic data generation for demonstration purposes but is architected to easily integrate with real data sources including mobile carrier APIs, utility payment systems, and financial cooperative databases.
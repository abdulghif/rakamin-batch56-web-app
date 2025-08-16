import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .prediction-result {
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .churn-yes {
        background-color: #ffebee;
        color: #c62828;
        border: 2px solid #ef5350;
    }
    .churn-no {
        background-color: #e8f5e8;
        color: #2e7d32;
        border: 2px solid #66bb6a;
    }
</style>
""", unsafe_allow_html=True)

class ChurnPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.model_accuracy = None
        
    def load_or_create_sample_data(self):
        """Load data or create sample data for demonstration"""
        # Create sample data that matches your dataset structure
        np.random.seed(42)
        n_samples = 10000
        
        data = {
            'age': np.random.randint(18, 80, n_samples),
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'purchase_amount': np.random.uniform(100, 50000, n_samples),
            'tenure': np.random.randint(1, 60, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Create churn based on some logical rules (for demonstration)
        churn_prob = (
            0.1 +  # base probability
            0.3 * (df['age'] > 60) +  # older customers more likely to churn
            0.2 * (df['tenure'] < 6) +  # new customers more likely to churn
            0.15 * (df['purchase_amount'] < 1000)  # low-value customers more likely to churn
        )
        
        df['churn'] = np.random.binomial(1, churn_prob, n_samples)
        
        return df
    
    def preprocess_data(self, df):
        """Preprocess the data for training"""
        df_processed = df.copy()
        
        # Encode gender
        if 'gender' in df_processed.columns:
            df_processed['gender_encoded'] = self.label_encoder.fit_transform(df_processed['gender'])
        
        return df_processed
    
    def train_model(self, df, model_type='Random Forest'):
        """Train the churn prediction model"""
        # Preprocess data
        df_processed = self.preprocess_data(df)
        
        # Features and target
        feature_columns = ['age', 'gender_encoded', 'purchase_amount', 'tenure']
        X = df_processed[feature_columns]
        y = df_processed['churn']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        if model_type == 'Random Forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'Logistic Regression':
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        
        self.model.fit(X_train_scaled, y_train)
        
        # Calculate accuracy
        y_pred = self.model.predict(X_test_scaled)
        self.model_accuracy = accuracy_score(y_test, y_pred)
        
        self.is_trained = True
        
        return {
            'accuracy': self.model_accuracy,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'X_test': X_test_scaled,
            'y_test': y_test,
            'y_pred': y_pred
        }
    
    def predict_churn(self, age, gender, purchase_amount, tenure):
        """Predict churn for a single customer"""
        if not self.is_trained:
            return None, None
        
        # Encode gender
        gender_encoded = 1 if gender == 'Male' else 0
        
        # Create feature array
        features = np.array([[age, gender_encoded, purchase_amount, tenure]])
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Make prediction
        churn_prob = self.model.predict_proba(features_scaled)[0]
        churn_prediction = self.model.predict(features_scaled)[0]
        
        return churn_prediction, churn_prob

# Initialize the predictor
@st.cache_resource
def get_predictor():
    return ChurnPredictor()

predictor = get_predictor()

# Main app
def main():
    st.markdown('<h1 class="main-header">🎯 Customer Churn Prediction System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🎛️ Model Configuration")
    
    # Model training section
    st.sidebar.subheader("Model Training")
    model_type = st.sidebar.selectbox(
        "Select Model Type",
        ["Random Forest", "Logistic Regression"],
        index=0
    )
    
    if st.sidebar.button("🚀 Train Model", type="primary"):
        with st.spinner("Training model... Please wait"):
            # Load sample data
            df = predictor.load_or_create_sample_data()
            
            # Train model
            results = predictor.train_model(df, model_type)
            
            st.sidebar.success(f"✅ Model trained successfully!")
            st.sidebar.metric("Model Accuracy", f"{results['accuracy']:.3f}")
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Model Performance", "📈 Data Analysis"])
    
    with tab1:
        st.header("Make Churn Prediction")
        
        if not predictor.is_trained:
            st.warning("⚠️ Please train the model first using the sidebar.")
            st.info("Click 'Train Model' in the sidebar to get started!")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Customer Information")
                
                age = st.number_input(
                    "Age",
                    min_value=18,
                    max_value=100,
                    value=30,
                    help="Customer's age in years"
                )
                
                gender = st.selectbox(
                    "Gender",
                    ["Female", "Male"],
                    index=0,
                    help="Customer's gender"
                )
                
                purchase_amount = st.number_input(
                    "Purchase Amount (IDR)",
                    min_value=0.0,
                    value=10000000.0,  # 10 million IDR
                    step=100000.0,
                    format="%.0f",
                    help="Total purchase amount in Indonesian Rupiah"
                )
                
                tenure = st.number_input(
                    "Tenure (months)",
                    min_value=1,
                    max_value=120,
                    value=12,
                    help="Number of months as customer"
                )
            
            with col2:
                st.subheader("Prediction Results")
                
                if st.button("🎯 Predict Churn", type="primary"):
                    churn_pred, churn_prob = predictor.predict_churn(age, gender, purchase_amount, tenure)
                    
                    if churn_pred is not None:
                        # Display prediction
                        if churn_pred == 1:
                            st.markdown(
                                '<div class="prediction-result churn-yes">🚨 HIGH RISK - Customer Likely to Churn</div>',
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                '<div class="prediction-result churn-no">✅ LOW RISK - Customer Likely to Stay</div>',
                                unsafe_allow_html=True
                            )
                        
                        # Probability gauge
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number+delta",
                            value = churn_prob[1] * 100,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Churn Probability (%)"},
                            delta = {'reference': 50},
                            gauge = {
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 30], 'color': "lightgreen"},
                                    {'range': [30, 70], 'color': "yellow"},
                                    {'range': [70, 100], 'color': "lightcoral"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 70
                                }
                            }
                        ))
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Probability breakdown
                        col3, col4 = st.columns(2)
                        with col3:
                            st.metric("Stay Probability", f"{churn_prob[0]:.1%}")
                        with col4:
                            st.metric("Churn Probability", f"{churn_prob[1]:.1%}")
    
    with tab2:
        st.header("Model Performance")
        
        if not predictor.is_trained:
            st.warning("⚠️ Please train the model first to see performance metrics.")
        else:
            st.success(f"✅ Model is trained and ready! Accuracy: {predictor.model_accuracy:.3f}")
            
            # Feature importance (for Random Forest)
            if hasattr(predictor.model, 'feature_importances_'):
                st.subheader("📊 Feature Importance")
                
                feature_names = ['Age', 'Gender', 'Purchase Amount', 'Tenure']
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': predictor.model.feature_importances_
                })
                importance_df = importance_df.sort_values('Importance', ascending=True)
                
                fig = px.bar(
                    importance_df, 
                    x='Importance', 
                    y='Feature',
                    orientation='h',
                    title="Feature Importance in Churn Prediction"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Data Analysis")
        
        # Load and display sample data analysis
        df = predictor.load_or_create_sample_data()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Customers", len(df))
        with col2:
            st.metric("Churn Rate", f"{df['churn'].mean():.1%}")
        with col3:
            st.metric("Avg Age", f"{df['age'].mean():.1f}")
        with col4:
            st.metric("Avg Tenure", f"{df['tenure'].mean():.1f} months")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Age distribution by churn
            fig = px.histogram(
                df, 
                x='age', 
                color='churn',
                nbins=20,
                title='Age Distribution by Churn Status',
                labels={'churn': 'Churn Status'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Purchase amount vs churn
            fig = px.box(
                df, 
                x='churn', 
                y='purchase_amount',
                title='Purchase Amount by Churn Status'
            )
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
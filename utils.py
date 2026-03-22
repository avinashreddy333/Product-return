import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from pathlib import Path

class PredictionUtils:
    """
    Utility class for the Streamlit application
    """
    
    def __init__(self):
        self.model_data = None
        self.feature_columns = None
        self.label_encoders = {}
        
    def load_model(self, model_path='model.pkl'):
        """
        Load the trained model and preprocessing objects
        """
        try:
            if not Path(model_path).exists():
                st.error(f"Model file {model_path} not found. Please train the model first.")
                return False
                
            self.model_data = joblib.load(model_path)
            self.model = self.model_data['model']
            self.scaler = self.model_data['scaler']
            self.label_encoders = self.model_data['label_encoders']
            self.feature_columns = self.model_data['feature_columns']
            self.model_type = self.model_data['model_type']
            
            return True
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return False
    
    def preprocess_input(self, input_dict):
        """
        Preprocess user input for prediction
        """
        try:
            # Convert input dictionary to DataFrame
            input_df = pd.DataFrame([input_dict])
            
            # Handle categorical variables
            categorical_columns = ['Product_Category', 'Customer_Location', 'Payment_Method']
            
            for col in categorical_columns:
                if col in input_df.columns and col in self.label_encoders:
                    # Handle unseen categories
                    if input_df[col].iloc[0] not in self.label_encoders[col].classes_:
                        # Assign the most common class or a default
                        input_df[col] = self.label_encoders[col].classes_[0]
                    
                    input_df[col] = self.label_encoders[col].transform(input_df[col])
            
            # Ensure all required columns are present
            for col in self.feature_columns:
                if col not in input_df.columns:
                    input_df[col] = 0  # Default value
            
            # Select only the required columns in the correct order
            input_df = input_df[self.feature_columns]
            
            # Scale numerical features
            numerical_columns = input_df.select_dtypes(include=['int64', 'float64']).columns
            input_df[numerical_columns] = self.scaler.transform(input_df[numerical_columns])
            
            return input_df
            
        except Exception as e:
            st.error(f"Error preprocessing input: {str(e)}")
            return None
    
    def make_prediction(self, input_dict):
        """
        Make prediction on user input
        """
        try:
            if self.model is None:
                st.error("Model not loaded")
                return None, None
            
            # Preprocess input
            input_df = self.preprocess_input(input_dict)
            if input_df is None:
                return None, None
            
            # Make prediction
            prediction = self.model.predict(input_df)[0]
            probabilities = self.model.predict_proba(input_df)[0]
            
            return prediction, probabilities
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            return None, None
    
    def get_feature_importance(self):
        """
        Get feature importance from the model
        """
        try:
            if self.model_type == 'random_forest' and hasattr(self.model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': self.feature_columns,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False)
                return importance_df
            else:
                return None
        except Exception as e:
            st.error(f"Error getting feature importance: {str(e)}")
            return None
    
    @staticmethod
    def create_gauge_chart(probability, title="Return Probability"):
        """
        Create a gauge chart for probability visualization
        """
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = probability * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': title},
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
        
        fig.update_layout(height=400)
        return fig
    
    @staticmethod
    def create_feature_importance_chart(importance_df):
        """
        Create feature importance bar chart
        """
        if importance_df is None:
            return None
            
        fig = px.bar(
            importance_df.head(10),
            x='importance',
            y='feature',
            orientation='h',
            title='Top 10 Feature Importance',
            labels={'importance': 'Importance Score', 'feature': 'Feature'}
        )
        
        fig.update_layout(height=500)
        return fig
    
    @staticmethod
    def create_data_overview_chart(df):
        """
        Create data overview visualizations
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Return Distribution', 'Product Category Distribution', 
                          'Price Distribution', 'Rating Distribution'),
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "histogram"}, {"type": "histogram"}]]
        )
        
        # Return distribution
        return_counts = df['Returned'].value_counts()
        fig.add_trace(
            go.Pie(
                labels=['Not Returned', 'Returned'],
                values=return_counts.values,
                name="Return Status"
            ),
            row=1, col=1
        )
        
        # Product category distribution
        category_counts = df['Product_Category'].value_counts().head(5)
        fig.add_trace(
            go.Bar(
                x=category_counts.index,
                y=category_counts.values,
                name="Product Categories",
                marker_color='lightblue'
            ),
            row=1, col=2
        )
        
        # Price distribution
        fig.add_trace(
            go.Histogram(
                x=df['Product_Price'],
                nbinsx=20,
                name="Price Distribution",
                marker_color='lightgreen'
            ),
            row=2, col=1
        )
        
        # Rating distribution
        fig.add_trace(
            go.Histogram(
                x=df['Product_Rating'],
                nbinsx=10,
                name="Rating Distribution",
                marker_color='salmon'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            showlegend=False,
            title_text="Dataset Overview"
        )
        
        return fig
    
    @staticmethod
    def get_data_summary(df):
        """
        Get summary statistics of the dataset
        """
        summary = {
            'total_records': len(df),
            'return_rate': f"{df['Returned'].mean():.2%}",
            'avg_price': f"${df['Product_Price'].mean():.2f}",
            'avg_rating': f"{df['Product_Rating'].mean():.1f}",
            'avg_delivery_time': f"{df['Delivery_Time'].mean():.1f} days"
        }
        return summary
    
    @staticmethod
    def validate_input(input_dict):
        """
        Validate user input
        """
        errors = []
        
        # Price validation
        if input_dict.get('Product_Price', 0) <= 0:
            errors.append("Product price must be greater than 0")
        
        if input_dict.get('Product_Price', 0) > 10000:
            errors.append("Product price seems unusually high")
        
        # Age validation
        age = input_dict.get('Customer_Age', 0)
        if age < 18 or age > 100:
            errors.append("Customer age must be between 18 and 100")
        
        # Rating validation
        rating = input_dict.get('Product_Rating', 0)
        if rating < 1 or rating > 5:
            errors.append("Product rating must be between 1 and 5")
        
        # Delivery time validation
        delivery_time = input_dict.get('Delivery_Time', 0)
        if delivery_time < 1 or delivery_time > 30:
            errors.append("Delivery time must be between 1 and 30 days")
        
        # Return history rate validation
        return_rate = input_dict.get('Return_History_Rate', 0)
        if return_rate < 0 or return_rate > 1:
            errors.append("Return history rate must be between 0 and 1")
        
        return errors

# Constants for the application
PRODUCT_CATEGORIES = ['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books', 
                     'Toys', 'Beauty', 'Automotive', 'Food', 'Health']

CUSTOMER_LOCATIONS = ['New York', 'California', 'Texas', 'Florida', 'Illinois', 
                     'Pennsylvania', 'Ohio', 'Georgia', 'Michigan', 'North Carolina']

PAYMENT_METHODS = ['Credit Card', 'Debit Card', 'PayPal', 'Cash on Delivery', 'UPI']

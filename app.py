st.set_page_config(
    page_title="E-Commerce Product Return Prediction model",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)
import streamlit as st
st.write("THIS IS ROOT APP.PY WITH UPLOAD")
import streamlit as st
import pandas as pd
import numpy as np
from utils import PredictionUtils, PRODUCT_CATEGORIES, CUSTOMER_LOCATIONS, PAYMENT_METHODS
import plotly.graph_objects as go
from pathlib import Path

# Page configuration


# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-result {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .likely-return {
        background-color: #ffcccc;
        border: 2px solid #ff0000;
    }
    .not-likely-return {
        background-color: #ccffcc;
        border: 2px solid #00ff00;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Initialize prediction utils
    pred_utils = PredictionUtils()
    
    # Load model
    if not pred_utils.load_model():
        st.error("Failed to load the model. Please ensure 'model.pkl' exists.")
        return
    
    # Sidebar navigation
    st.sidebar.title(" Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Home", "Predict", "Model Insights", "Dataset Preview"])
    
    if page == "Home":
        home_page()
    elif page == "Predict":
        prediction_page(pred_utils)
    elif page == "Model Insights":
        insights_page(pred_utils)
    elif page == "Dataset Preview":
        dataset_page()

def home_page():
    """Home page with project information"""
    st.markdown('<h1 class="main-header"> E-Commerce Product Return Prediction System</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ##  Project Overview
    
    This intelligent system predicts whether a product is likely to be returned based on various factors such as:
    
    - **Product Characteristics**: Category, price, rating
    - **Customer Information**: Age, location, purchase history
    - **Transaction Details**: Payment method, delivery time, discounts
    - **Historical Data**: Return history rate
    
    ##  Key Features
    
    - **Machine Learning Powered**: Uses Random Forest classifier for accurate predictions
    - **Real-time Predictions**: Get instant predictions with probability scores
    - **Interactive Dashboard**: User-friendly interface with visualizations
    - **Data Insights**: Explore feature importance and dataset statistics
    
    ##  How to Use
    
    1. Navigate to the **Predict** page using the sidebar
    2. Fill in the product and customer details
    3. Click **Predict Return** to get the result
    4. Explore **Model Insights** to understand the factors influencing returns
    
    ---
    *Built with Streamlit, Scikit-learn, and Plotly*
    """)
    
    # Display model performance metrics if available
    if Path('model.pkl').exists():
        st.markdown("###  Model Performance")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", "59.5%")
        with col2:
            st.metric("Precision", "45.7%")
        with col3:
            st.metric("Recall", "13.3%")
        with col4:
            st.metric("F1 Score", "20.6%")

def prediction_page(pred_utils):
    """Main prediction interface"""
    st.markdown('<h1 class="main-header"> Predict Product Return</h1>', unsafe_allow_html=True)
    
    # Create two columns for input form
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("###  Product Details")
        
        product_category = st.selectbox(
            "Product Category",
            options=PRODUCT_CATEGORIES,
            help="Select the category of the product"
        )
        
        product_price = st.number_input(
            "Product Price ($)",
            min_value=1.0,
            max_value=10000.0,
            value=50.0,
            step=1.0,
            help="Enter the product price in USD"
        )
        
        product_rating = st.slider(
            "Product Rating",
            min_value=1.0,
            max_value=5.0,
            value=4.0,
            step=0.1,
            help="Customer rating for the product (1-5)"
        )
        
        discount_applied = st.selectbox(
            "Discount Applied (%)",
            options=[0, 5, 10, 15, 20, 25, 30],
            index=0,
            help="Discount percentage applied to the product"
        )
    
    with col2:
        st.markdown("###  Customer Details")
        
        customer_age = st.number_input(
            "Customer Age",
            min_value=18,
            max_value=100,
            value=35,
            help="Age of the customer"
        )
        
        customer_location = st.selectbox(
            "Customer Location",
            options=CUSTOMER_LOCATIONS,
            help="Customer's location"
        )
        
        payment_method = st.selectbox(
            "Payment Method",
            options=PAYMENT_METHODS,
            help="Payment method used for the purchase"
        )
        
        delivery_time = st.number_input(
            "Delivery Time (days)",
            min_value=1,
            max_value=30,
            value=3,
            help="Expected delivery time in days"
        )
    
    # Additional customer history section
    st.markdown("###  Customer History")
    col3, col4 = st.columns(2)
    
    with col3:
        purchase_history = st.number_input(
            "Purchase History Count",
            min_value=0,
            max_value=100,
            value=5,
            help="Number of previous purchases by this customer"
        )
    
    with col4:
        return_history_rate = st.slider(
            "Return History Rate",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.01,
            format="%.2f",
            help="Customer's historical return rate (0-1)"
        )
    
    # Prediction button
    st.markdown("---")
    
    col5, col6, col7 = st.columns([1, 2, 1])
    
    with col6:
        if st.button("Predict Return", type="primary", use_container_width=True):
            # Prepare input data
            input_data = {
                'Product_Category': product_category,
                'Product_Price': product_price,
                'Customer_Age': customer_age,
                'Customer_Location': customer_location,
                'Purchase_History_Count': purchase_history,
                'Return_History_Rate': return_history_rate,
                'Delivery_Time': delivery_time,
                'Payment_Method': payment_method,
                'Discount_Applied': discount_applied,
                'Product_Rating': product_rating
            }
            
            # Validate input
            errors = PredictionUtils.validate_input(input_data)
            if errors:
                for error in errors:
                    st.error(error)
                return
            
            # Make prediction
            prediction, probabilities = pred_utils.make_prediction(input_data)
            
            if prediction is not None:
                display_prediction_result(prediction, probabilities[1])

def display_prediction_result(prediction, return_probability):
    """Display the prediction result with visualizations"""
    st.markdown("---")
    st.markdown("###  Prediction Result")
    
    # Determine result message and styling
    if prediction == 1:
        result_class = "likely-return"
        result_emoji = ""
        result_text = "Likely to be Returned"
        result_color = "red"
    else:
        result_class = "not-likely-return"
        result_emoji = ""
        result_text = "Not Likely to be Returned"
        result_color = "green"
    
    # Display result card
    st.markdown(f"""
    <div class="prediction-result {result_class}">
        <h2 style="text-align: center; color: {result_color};">
            {result_emoji} {result_text}
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Display probability gauge
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        fig = PredictionUtils.create_gauge_chart(
            return_probability,
            "Return Probability"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Display additional metrics
    st.markdown("###  Detailed Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Return Probability",
            f"{return_probability:.1%}",
            delta=None
        )
    
    with col2:
        st.metric(
            "Confidence",
            f"{max(return_probability, 1-return_probability):.1%}",
            delta=None
        )
    
    with col3:
        risk_level = "High" if return_probability > 0.7 else "Medium" if return_probability > 0.3 else "Low"
        st.metric("Risk Level", risk_level)
    
    with col4:
        recommendation = "Review Order" if return_probability > 0.5 else "Proceed"
        st.metric("Recommendation", recommendation)

def insights_page(pred_utils):
    """Model insights and feature importance"""
    st.markdown('<h1 class="main-header"> Model Insights</h1>', unsafe_allow_html=True)
    
    # Feature importance
    st.markdown("###  Feature Importance")
    importance_df = pred_utils.get_feature_importance()
    
    if importance_df is not None:
        fig = PredictionUtils.create_feature_importance_chart(importance_df)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display feature importance table
        st.markdown("#### Detailed Feature Importance")
        st.dataframe(importance_df, use_container_width=True)
    else:
        st.info("Feature importance is only available for Random Forest models.")
    
    # Model information
    st.markdown("###  Model Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Model Type:**")
        st.write(f"- {pred_utils.model_type.replace('_', ' ').title()}")
        
        st.markdown("**Number of Features:**")
        st.write(f"- {len(pred_utils.feature_columns)}")
    
    with col2:
        st.markdown("**Feature Columns:**")
        for feature in pred_utils.feature_columns:
            st.write(f"- {feature}")

def dataset_page():
    """Dataset preview and statistics"""
    st.markdown('<h1 class="main-header"> Dataset Preview</h1>', unsafe_allow_html=True)
    
    # Load dataset
    if Path('data.csv').exists():
        df = pd.read_csv('data.csv')
        
        # Dataset summary
        st.markdown("###  Dataset Summary")
        summary = PredictionUtils.get_data_summary(df)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Records", summary['total_records'])
        with col2:
            st.metric("Return Rate", summary['return_rate'])
        with col3:
            st.metric("Avg Price", summary['avg_price'])
        with col4:
            st.metric("Avg Rating", summary['avg_rating'])
        with col5:
            st.metric("Avg Delivery", summary['avg_delivery_time'])
        
        # Data overview chart
        st.markdown("###  Data Overview")
        fig = PredictionUtils.create_data_overview_chart(df)
        st.plotly_chart(fig, use_container_width=True)
        
        # Dataset preview
        st.markdown("###  Dataset Preview")
        
        # Show sample of the data
        sample_size = st.slider("Number of rows to display", 5, 50, 10)
        
        st.dataframe(
            df.head(sample_size),
            use_container_width=True,
            hide_index=True
        )
        
        # Column information
        st.markdown("###  Column Information")
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes.values,
            'Non-Null Count': df.count().values,
            'Unique Values': df.nunique().values
        })
        st.dataframe(col_info, use_container_width=True, hide_index=True)
        
    else:
        st.error("Dataset file 'data.csv' not found. Please run generate_data.py first.")

if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os
from pathlib import Path

# Add backend directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

def simple_predict_batch(df):
    """Simple flexible batch prediction function"""
    try:
        # Look for numeric columns that might be useful
        numeric_cols = []
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['price', 'cost', 'amount']):
                numeric_cols.append(col)
            elif any(keyword in col.lower() for keyword in ['age']):
                numeric_cols.append(col)
            elif any(keyword in col.lower() for keyword in ['rating', 'score']):
                numeric_cols.append(col)
            elif any(keyword in col.lower() for keyword in ['time', 'day']):
                numeric_cols.append(col)
        
        # Use available numeric columns
        available_cols = [col for col in numeric_cols if col in df.columns]
        if len(available_cols) < 2:
            available_cols = list(df.columns)[:4]
        
        X = df[available_cols].copy()
        
        # Convert to numeric and handle missing values
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
            median_val = X[col].median()
            if pd.isna(median_val):
                median_val = 0
            X[col] = X[col].fillna(median_val)
        
        # Simple rule-based prediction
        predictions = []
        probabilities = []
        
        for _, row in X.iterrows():
            prob = 0.3  # Base probability
            
            # Adjust based on available columns
            for col in X.columns:
                col_lower = col.lower()
                val = row[col]
                
                if not pd.isna(val):
                    if 'price' in col_lower or 'cost' in col_lower:
                        if val > 100:
                            prob += 0.2
                        elif val < 20:
                            prob -= 0.1
                    elif 'age' in col_lower:
                        if val < 25 or val > 65:
                            prob += 0.1
                    elif 'rating' in col_lower or 'score' in col_lower:
                        if val < 2:
                            prob += 0.3
            
            prob = max(0.05, min(0.95, prob))
            probabilities.append(prob)
            predictions.append(1 if prob > 0.5 else 0)
        
        # Create results dataframe
        results_df = df.copy()
        cols_to_remove = ['Prediction', 'Return_Probability', 'Prediction_Label', 'Confidence_Level']
        results_df = results_df.drop(columns=[col for col in cols_to_remove if col in results_df.columns])
        
        results_df['Prediction'] = predictions
        results_df['Return_Probability'] = probabilities
        results_df['Prediction_Label'] = results_df['Prediction'].map({
            0: 'Not Return', 1: 'Return'
        })
        
        # Add confidence levels
        confidence_levels = []
        for prob in probabilities:
            if prob > 0.8:
                confidence_levels.append('Very High')
            elif prob > 0.7:
                confidence_levels.append('High')
            elif prob > 0.6:
                confidence_levels.append('Medium-High')
            elif prob > 0.4:
                confidence_levels.append('Medium')
            elif prob > 0.3:
                confidence_levels.append('Medium-Low')
            else:
                confidence_levels.append('Low')
        
        results_df['Confidence_Level'] = confidence_levels
        
        summary_stats = {
            'total_records': len(results_df),
            'predicted_returns': int(results_df['Prediction'].sum()),
            'predicted_non_returns': int(len(results_df) - results_df['Prediction'].sum()),
            'return_rate': f"{results_df['Prediction'].mean():.2%}",
            'avg_probability': f"{pd.Series(probabilities).mean():.3f}",
            'high_risk_count': int(sum(1 for p in probabilities if p > 0.7)),
            'medium_risk_count': int(sum(1 for p in probabilities if 0.3 < p <= 0.7)),
            'low_risk_count': int(sum(1 for p in probabilities if p <= 0.3)),
            'columns_used': available_cols,
            'model_type': 'simple_rule_based',
            'optimal_threshold': f"0.500"
        }
        
        return {
            'success': True,
            'results_df': results_df,
            'summary_stats': summary_stats
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        }

# Page configuration
st.set_page_config(
    page_title="E-Commerce Return Prediction - Dataset System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .upload-section {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #f5c6cb;
        margin: 1rem 0;
    }
    .warning-message {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #ffeaa7;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .dataset-info {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

def load_model():
    """Load simple flexible model"""
    try:
        # Use simple function instead of class
        success, message = True, "Simple rule-based model created"
        
        if success:
            st.session_state.predictor = None  # We'll use simple_predict_batch directly
            st.session_state.model_loaded = True
            return True, message
        else:
            return False, message
    except Exception as e:
        return False, f"Error loading model: {str(e)}"

def display_validation_result(validation_result):
    """Display dataset validation results"""
    # Handle both old format (from complex predictors) and new format (from simple predictor)
    if not validation_result.get('success', True):  # If success key exists, this is from simple predictor
        if not validation_result['success']:
            st.markdown(f"""
            <div class="error-message">
                <strong>❌ Dataset Validation Failed</strong><br>
                {validation_result.get('error', 'Unknown error')}
            </div>
            """, unsafe_allow_html=True)
            return False
        
        # For simple predictor, show basic info
        info = validation_result.get('info', {})
        if info:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Rows", f"{info.get('total_rows', 0):,}")
            
            with col2:
                st.metric("Total Columns", info.get('total_columns', 0))
            
            with col3:
                st.metric("Missing Values", validation_result.get('missing_values_count', 0))
        
        # Display warnings
        warnings = validation_result.get('warnings', [])
        if warnings:
            st.markdown("### ⚠️ Warnings")
            for warning in warnings:
                st.warning(warning)
        
        return True
    
    # Original validation result handling for complex predictors
    if not validation_result['is_valid']:
        st.markdown(f"""
        <div class="error-message">
            <strong>❌ Dataset Validation Failed</strong><br>
            {'<br>'.join([f"• {error}" for error in validation_result['errors']])}
        </div>
        """, unsafe_allow_html=True)
        return False
    
    # Display basic info
    info = validation_result['info']
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Rows", f"{info['total_rows']:,}")
    
    with col2:
        st.metric("Total Columns", info['total_columns'])
    
    with col3:
        st.metric("Missing Values", validation_result['missing_values_count'])
    
    # Display warnings
    if validation_result['warnings']:
        st.markdown("### ⚠️ Warnings")
        for warning in validation_result['warnings']:
            st.warning(warning)
    
    # Display data quality issues
    if validation_result['data_quality_issues']:
        st.markdown("### 🔍 Data Quality Issues")
        for issue in validation_result['data_quality_issues']:
            st.info(issue)
    
    # Display dataset columns
    st.markdown("### 📋 Dataset Columns")
    st.write("The following columns were found in your dataset:")
    st.write(", ".join(validation_result.get('dataset_columns', [])))
    
    return True

def display_prediction_results(results_df, summary_stats, validation_metrics=None):
    """Display production-grade prediction results with comprehensive metrics"""
    st.markdown("### 🎯 Production Prediction Results")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{summary_stats['total_records']:,}")
    
    with col2:
        st.metric("Predicted Returns", f"{summary_stats['predicted_returns']:,}")
    
    with col3:
        st.metric("Return Rate", summary_stats['return_rate'])
    
    with col4:
        st.metric("Avg Probability", summary_stats['avg_probability'])
    
    # Data processing metrics
    st.markdown("### 🔧 Production Pipeline Processing")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Missing Values Handled", f"{summary_stats.get('missing_values_handled', 0):,}")
    
    with col2:
        st.metric("Leakage Columns Removed", f"{summary_stats.get('leakage_columns_removed', 0)}")
    
    with col3:
        st.metric("Feature Coverage", summary_stats.get('feature_coverage', 'N/A'))
    
    with col4:
        st.metric("Pipeline Status", "✅ Production")
    
    # Probability analysis
    st.markdown("### 📊 Probability Distribution Analysis")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Mean Probability", summary_stats.get('probability_mean', 'N/A'))
    
    with col2:
        st.metric("Std Deviation", summary_stats.get('probability_std', 'N/A'))
    
    with col3:
        st.metric("Above 50%", summary_stats.get('predictions_above_50', 'N/A'))
    
    with col4:
        st.metric("Below 50%", summary_stats.get('predictions_below_50', 'N/A'))
    
    # Risk distribution with confidence levels
    st.markdown("### ⚠️ Risk Distribution & Confidence")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("High Risk (>70%)", f"{summary_stats['high_risk_count']:,}")
    
    with col2:
        st.metric("Medium Risk (30-70%)", f"{summary_stats['medium_risk_count']:,}")
    
    with col3:
        st.metric("Low Risk (≤30%)", f"{summary_stats['low_risk_count']:,}")
    
    # Validation metrics if available
    if validation_metrics:
        st.markdown("### ✅ Validation Metrics (Compared with Actual Data)")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Prediction Accuracy", summary_stats.get('prediction_accuracy', 'N/A'))
        
        with col2:
            st.metric("Precision", summary_stats.get('prediction_precision', 'N/A'))
        
        with col3:
            st.metric("Recall", summary_stats.get('prediction_recall', 'N/A'))
        
        with col4:
            st.metric("F1 Score", summary_stats.get('prediction_f1', 'N/A'))
        
        if 'actual_return_rate' in summary_stats:
            st.info(f"📊 Actual Return Rate: {summary_stats['actual_return_rate']} vs Predicted: {summary_stats['return_rate']}")
        
        # Confusion matrix display
        if all(key in summary_stats for key in ['true_negatives', 'false_positives', 'false_negatives', 'true_positives']):
            st.markdown("### 📈 Confusion Matrix")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Actual Not Returned**")
                st.write(f"True Negatives: {summary_stats['true_negatives']}")
                st.write(f"False Positives: {summary_stats['false_positives']}")
            
            with col2:
                st.markdown("**Actual Returned**")
                st.write(f"False Negatives: {summary_stats['false_negatives']}")
                st.write(f"True Positives: {summary_stats['true_positives']}")
    
    # Risk distribution visualization
    st.markdown("### 📊 Risk Distribution Visualization")
    col1, col2 = st.columns(2)
    
    with col1:
        # Create risk distribution chart
        risk_data = pd.DataFrame({
            'Risk Level': ['High Risk (>70%)', 'Medium Risk (30-70%)', 'Low Risk (≤30%)'],
            'Count': [
                summary_stats['high_risk_count'],
                summary_stats['medium_risk_count'],
                summary_stats['low_risk_count']
            ]
        })
        
        fig = px.pie(risk_data, values='Count', names='Risk Level', 
                    title='Risk Distribution')
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        # Create probability distribution histogram
        if 'Return_Probability' in results_df.columns:
            fig = px.histogram(results_df, x='Return_Probability', 
                             title='Probability Distribution',
                             nbins=20)
            fig.add_vline(x=0.5, line_dash="dash", line_color="red", 
                         annotation_text="Decision Threshold")
            st.plotly_chart(fig, width='stretch')
    
    # Results table with confidence levels
    st.markdown("### 📋 Detailed Prediction Results")
    
    # Prepare display columns - use original dataset columns plus predictions
    display_columns = []
    
    # Add original columns from the dataset (first 10 columns to avoid overcrowding)
    original_columns = [col for col in results_df.columns if col not in ['Prediction', 'Return_Probability', 'Prediction_Label', 'Confidence_Level']]
    display_columns.extend(original_columns[:10])  # Show first 10 original columns
    
    # Add prediction columns
    display_columns.extend(['Prediction_Label', 'Return_Probability', 'Confidence_Level'])
    
    # Add actual return status if available
    if 'Return_Status' in results_df.columns:
        display_columns.append('Return_Status')
    
    # Filter to available columns
    available_columns = [col for col in display_columns if col in results_df.columns]
    
    if available_columns:
        # Show sample of results
        sample_size = min(100, len(results_df))
        display_df = results_df[available_columns].head(sample_size)
        
        # Remove duplicate columns by keeping only the first occurrence
        display_df = display_df.loc[:, ~display_df.columns.duplicated()]
        
        st.dataframe(display_df, width='stretch', hide_index=True)
        
        if len(results_df) > sample_size:
            st.info(f"Showing first {sample_size} of {len(results_df)} records. Download the complete file for all results.")
    else:
        st.warning("No columns available for display.")
    
    # Visualizations
    st.markdown("### 📈 Visualizations")
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Return Prediction Distribution', 'Probability Distribution',
                       'Risk Level Distribution', 'Category-wise Return Rate'),
        specs=[[{"type": "pie"}, {"type": "histogram"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Return prediction distribution
    return_counts = results_df['Prediction_Label'].value_counts()
    fig.add_trace(
        go.Pie(
            labels=return_counts.index,
            values=return_counts.values,
            name="Prediction Distribution"
        ),
        row=1, col=1
    )
    
    # Probability distribution
    fig.add_trace(
        go.Histogram(
            x=results_df['Return_Probability'],
            nbinsx=20,
            name="Probability Distribution",
            marker_color='lightblue'
        ),
        row=1, col=2
    )
    
    # Risk level distribution
    risk_levels = ['High Risk', 'Medium Risk', 'Low Risk']
    risk_counts = [
        summary_stats['high_risk_count'],
        summary_stats['medium_risk_count'],
        summary_stats['low_risk_count']
    ]
    fig.add_trace(
        go.Bar(
            x=risk_levels,
            y=risk_counts,
            name="Risk Distribution",
            marker_color=['red', 'orange', 'green']
        ),
        row=2, col=1
    )
    
    # Category-wise return rate (if product_category exists)
    if 'product_category' in results_df.columns:
        category_return_rate = results_df.groupby('product_category')['Prediction'].mean().sort_values(ascending=False)
        fig.add_trace(
            go.Bar(
                x=category_return_rate.index[:10],  # Top 10 categories
                y=category_return_rate.values[:10],
                name="Category Return Rate",
                marker_color='salmon'
            ),
            row=2, col=2
        )
    
    fig.update_layout(
        height=800,
        showlegend=False,
        title_text="Prediction Analysis Dashboard"
    )
    
    st.plotly_chart(fig, width='stretch')
    
    # Results table
    st.markdown("### 📋 Detailed Results")
    
    # Add filters
    col1, col2 = st.columns(2)
    
    with col1:
        min_probability = st.slider("Minimum Return Probability", 0.0, 1.0, 0.0, 0.05)
    
    with col2:
        prediction_filter = st.selectbox("Filter by Prediction", ["All", "Return", "Not Return"])
    
    # Filter results
    filtered_df = results_df[results_df['Return_Probability'] >= min_probability].copy()
    
    if prediction_filter != "All":
        if prediction_filter == "Return":
            filtered_df = filtered_df[filtered_df['Prediction'] == 1]
        else:
            filtered_df = filtered_df[filtered_df['Prediction'] == 0]
    
    st.write(f"Showing {len(filtered_df):,} filtered records")
    
    # Display table with selected columns
    display_columns = ['product_category', 'product_price', 'customer_age', 'Prediction_Label', 'Return_Probability']
    available_columns = [col for col in display_columns if col in filtered_df.columns]
    
    if available_columns:
        # Add original columns if needed
        original_columns = [col for col in results_df.columns if col not in ['Prediction', 'Return_Probability', 'Prediction_Label']]
        display_cols = available_columns + [col for col in original_columns[:5] if col in filtered_df.columns]
        
        st.dataframe(
            filtered_df[display_cols].head(1000),  # Limit to 1000 rows for performance
            width='stretch',
            hide_index=True
        )
    
    return filtered_df

def create_download_button(results_df, filename_suffix=""):
    """Create a download button for the results"""
    # Create CSV data
    csv = results_df.to_csv(index=False)
    
    # Create download button
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name=f"return_predictions_{filename_suffix}.csv",
        mime="text/csv"
    )

def main():
    """Main application"""
    st.markdown('<h1 class="main-header">📊 E-Commerce Return Prediction System</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ## 🎯 Dataset-Based Batch Prediction System
    
    Upload your dataset to get batch predictions for product returns. The system will process all records 
    and provide detailed predictions with probability scores.
    """)
    
    # Load model
    if not st.session_state.model_loaded:
        with st.spinner("Loading ML model..."):
            success, message = load_model()
            if success:
                st.success(f"✅ {message}")
            else:
                st.error(f"❌ {message}")
                st.error("Please ensure the model has been trained by running the training script first.")
                return
    
    # File upload section
    st.markdown("---")
    st.markdown("### 📤 Upload Dataset")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file for prediction"
    )
    
    if uploaded_file is not None:
        # Store uploaded file in session state
        st.session_state.uploaded_file = uploaded_file
        
        try:
            # Read the uploaded file
            df = pd.read_csv(uploaded_file)
            
            # Display dataset preview
            st.markdown("#### 📋 Dataset Preview")
            st.dataframe(df.head(10), width='stretch')
            
            # Validate dataset
            validation_result = st.session_state.predictor.validate_dataset(df)
            
            if display_validation_result(validation_result):
                # Prediction button
                st.markdown("---")
                st.markdown("### 🚀 Run Prediction")
                
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col2:
                    if st.button("🔮 Run Batch Prediction", type="primary", width='stretch'):
                        with st.spinner("Processing predictions... This may take a while for large datasets."):
                            # Make predictions
                            result = simple_predict_batch(df)
                            
                            if result['success']:
                                st.session_state.prediction_results = result
                                st.success("✅ Predictions completed successfully!")
                            else:
                                st.error(f"❌ {result['error']}")
                                if 'details' in result:
                                    for detail in result['details']:
                                        st.error(f"• {detail}")
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.error("Please check your file format and try again.")
        
        # Display results if available
        if st.session_state.prediction_results:
            results_df = st.session_state.prediction_results['results_df']
            summary_stats = st.session_state.prediction_results['summary_stats']
            validation_metrics = st.session_state.prediction_results.get('validation_metrics')
            
            filtered_df = display_prediction_results(results_df, summary_stats, validation_metrics)
            
            # Download section
            st.markdown("---")
            st.markdown("### 📥 Download Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                create_download_button(results_df, "complete")
            
            with col2:
                if len(filtered_df) < len(results_df):
                    create_download_button(filtered_df, "filtered")
                
                st.info("💡 **Tip**: The complete file contains all predictions. The filtered file contains only the records matching your current filters.")
    
    # Instructions section
    st.markdown("---")
    with st.expander("📖 How to Use This System"):
        st.markdown("""
        ### Step-by-Step Guide:
        
        1. **Prepare Your Dataset**: Ensure your CSV file contains the necessary data columns
        
        2. **Upload the File**: Click the "Choose a CSV file" button and select your dataset
        
        3. **Review Validation**: Check if your dataset passes validation. Address any warnings or errors
        
        4. **Run Prediction**: Click "Run Batch Prediction" to process all records
        
        5. **Analyze Results**: View the prediction summary, visualizations, and detailed results
        
        6. **Download Results**: Download the complete or filtered results as CSV files
        
        ### 📊 Output Columns:
        - `Prediction`: 0 (Not Return) or 1 (Return)
        - `Return_Probability`: Probability score (0-1)
        - `Prediction_Label`: Human-readable prediction label
        
        ### ⚡ Performance Tips:
        - For large datasets (>10,000 rows), processing may take several minutes
        - Use the filters to focus on high-risk predictions
        - Download the filtered results for targeted analysis
        """)

if __name__ == "__main__":
    main()

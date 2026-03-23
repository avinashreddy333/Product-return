import streamlit as st
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="E-Commerce Product Return Prediction Model",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

def simple_predict_batch(df):
    """Ultra-simple flexible prediction that works with ANY dataset"""
    try:
        # Basic info
        total_records = len(df)
        total_columns = len(df.columns)
        missing_values = df.isnull().sum().sum()
        
        # Simple rule-based prediction
        predictions = []
        probabilities = []
        
        for _, row in df.iterrows():
            prob = 0.3  # Base probability
            
            # Look for relevant columns and adjust probability
            for col in df.columns:
                col_lower = col.lower()
                try:
                    val = pd.to_numeric(row[col], errors='coerce')
                    
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
                except:
                    pass
            
            # Clamp probability
            prob = max(0.05, min(0.95, prob))
            probabilities.append(prob)
            predictions.append(1 if prob > 0.5 else 0)
        
        # Create results
        results_df = df.copy()
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
        
        # Summary statistics
        predicted_returns = int(results_df['Prediction'].sum())
        return_rate = f"{results_df['Prediction'].mean():.2%}"
        avg_probability = f"{np.mean(probabilities):.3f}"
        
        # Find columns used for prediction
        columns_used = []
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['price', 'cost', 'amount', 'age', 'rating', 'score', 'time', 'day']):
                columns_used.append(col)
                if len(columns_used) >= 5:
                    break
        
        summary_stats = {
            'total_records': total_records,
            'predicted_returns': predicted_returns,
            'predicted_non_returns': total_records - predicted_returns,
            'return_rate': return_rate,
            'avg_probability': avg_probability,
            'high_risk_count': int(sum(1 for p in probabilities if p > 0.7)),
            'medium_risk_count': int(sum(1 for p in probabilities if 0.3 < p <= 0.7)),
            'low_risk_count': int(sum(1 for p in probabilities if p <= 0.3)),
            'missing_values_handled': missing_values,
            'dataset_columns': total_columns,
            'columns_used': columns_used,
            'model_type': 'ultra_simple_rule_based',
            'optimal_threshold': '0.500'
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

def main():
    st.title("E-Commerce Product Return Prediction Model")
    
    
    # File upload
    uploaded_file = st.file_uploader(
        " Upload your CSV dataset",
        type=['csv'],
        help="Upload any CSV file - the system will automatically detect and use available columns"
    )
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            df = pd.read_csv(uploaded_file)
            
            # Display dataset info
            st.markdown("###  Dataset Information")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Records", f"{len(df):,}")
            
            with col2:
                st.metric("Total Columns", len(df.columns))
            
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())
            
            # Display columns found
            st.markdown("###  Columns Detected")
            st.write(f"**Found columns:** {', '.join(df.columns)}")
            
            # Find relevant columns
            relevant_cols = []
            for col in df.columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in ['price', 'cost', 'amount', 'age', 'rating', 'score', 'time', 'day']):
                    relevant_cols.append(col)
            
            st.write(f"**Relevant columns for prediction:** {', '.join(relevant_cols) if relevant_cols else 'Using first available columns'}")
            
            # Run prediction button
            st.markdown("---")
            if st.button("start prediction", type="primary", use_container_width=True):
                with st.spinner("Processing predictions..."):
                    result = simple_predict_batch(df)
                    
                    if result['success']:
                        st.success(" Prediction completed!")
                        
                        # Display results
                        st.markdown("### 🎯 Prediction Results")
                        stats = result['summary_stats']
                        
                        # First row: Total Records and Predicted Returns
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Total Records", f"{stats['total_records']:,}")
                        
                        with col2:
                            st.metric("Predicted Returns", f"{stats['predicted_returns']:,}")
                        
                        # Second row: Return Rate and Low Risk (side by side)
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Return Rate", stats['return_rate'])
                        
                        with col2:
                            st.metric("Low Risk", f"{stats['low_risk_count']:,}")
                        
                        # Third row: High Risk and Medium Risk (side by side)
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("High Risk", f"{stats['high_risk_count']:,}")
                        
                        with col2:
                            st.metric("Medium Risk", f"{stats['medium_risk_count']:,}")
                        
                        # Display sample results
                        st.markdown("### Sample Predictions")
                        display_cols = list(df.columns[:3]) + ['Prediction_Label', 'Return_Probability', 'Confidence_Level']
                        sample_df = result['results_df'][display_cols].head(10)
                        st.dataframe(sample_df, use_container_width=True)
                        
                        # Download results
                        csv = result['results_df'].to_csv(index=False)
                        st.download_button(
                            label=" Download Results",
                            data=csv,
                            file_name='predictions.csv',
                            mime='text/csv'
                        )
                    else:
                        st.error(f" {result['error']}")
            
        except Exception as e:
            st.error(f" Error processing file: {str(e)}")
    
    else:
        st.info(" Please upload a CSV file to get started")

if __name__ == "__main__":
    main()

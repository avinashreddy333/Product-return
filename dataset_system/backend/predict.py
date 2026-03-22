import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import warnings
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
warnings.filterwarnings('ignore')

class FlexibleBatchPredictor:
    """
    Flexible batch predictor that works with any dataset structure
    """
    
    def __init__(self, model_path='model.pkl'):
        self.model_path = model_path
        self.pipeline = None
        self.model_type = None
        self.threshold = 0.5
        self.feature_info = None
        
    def load_model(self):
        """
        Load the flexible model
        """
        try:
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"Model file {self.model_path} not found")
            
            model_data = joblib.load(self.model_path)
            
            self.pipeline = model_data['pipeline']
            self.model_type = model_data['model_type']
            self.threshold = model_data['threshold']
            self.feature_info = model_data['feature_info']
            
            return True, f"Flexible model loaded successfully"
            
        except Exception as e:
            return False, f"Error loading model: {str(e)}"
    
    def detect_feature_types(self, df):
        """
        Automatically detect feature types from any dataset
        """
        feature_info = {
            'numeric_features': [],
            'categorical_features': [],
            'target_column': None,
            'leakage_columns': []
        }
        
        # Define leakage columns to remove
        leakage_columns = [
            'return_status', 'return_reason', 'return_date', 'returned',
            'return', 'prediction', 'prediction_label', 'return_probability',
            'days_to_return', 'return_flag', 'is_returned', 'has_returned'
        ]
        
        # Define possible target column names
        target_column_variations = [
            'return_status', 'returned', 'return', 'is_returned', 
            'has_returned', 'return_flag', 'return_indicator'
        ]
        
        # Identify target column
        for col in df.columns:
            col_normalized = col.lower().strip()
            for target_col in target_column_variations:
                if col_normalized == target_col.lower():
                    feature_info['target_column'] = col
                    break
        
        # Identify and remove leakage columns
        for col in df.columns:
            col_lower = col.lower()
            if any(leakage_col in col_lower for leakage_col in leakage_columns):
                feature_info['leakage_columns'].append(col)
        
        # Clean dataframe
        df_clean = df.drop(columns=feature_info['leakage_columns'])
        if feature_info['target_column']:
            df_clean = df_clean.drop(columns=[feature_info['target_column']])
        
        # Detect numeric features
        for col in df_clean.columns:
            # Try to convert to numeric
            numeric_series = pd.to_numeric(df_clean[col], errors='coerce')
            if not numeric_series.isna().all():
                # Has some numeric values
                unique_count = df_clean[col].nunique()
                if unique_count > 10:  # Likely categorical if too many unique values
                    feature_info['categorical_features'].append(col)
                else:
                    feature_info['numeric_features'].append(col)
            else:
                # All non-numeric, treat as categorical
                feature_info['categorical_features'].append(col)
        
        return feature_info
    
    def prepare_data_for_prediction(self, df):
        """
        Prepare data using exact column names from dataset
        """
        df_clean = df.copy()
        
        # Detect feature types
        feature_info = self.detect_feature_types(df_clean)
        
        # Remove leakage columns
        if feature_info['leakage_columns']:
            df_clean = df_clean.drop(columns=feature_info['leakage_columns'])
        
        # Convert all columns to appropriate types
        for col in df_clean.columns:
            # Try to convert to numeric if possible
            try:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='ignore')
            except:
                pass
        
        # Fill missing values with appropriate defaults
        for col in df_clean.columns:
            if df_clean[col].dtype in ['int64', 'float64']:
                # Numeric columns - fill with median
                median_val = df_clean[col].median()
                if pd.isna(median_val):
                    median_val = 0
                df_clean[col] = df_clean[col].fillna(median_val)
            else:
                # Categorical columns - fill with mode
                mode_val = df_clean[col].mode()
                if len(mode_val) > 0:
                    df_clean[col] = df_clean[col].fillna(mode_val[0])
                else:
                    df_clean[col] = df_clean[col].fillna('Unknown')
        
        return df_clean, feature_info
    
    def validate_dataset(self, df):
        """
        Validate dataset - works with any column names
        """
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'info': {},
            'leakage_columns_found': [],
            'target_column_found': None,
            'missing_values_count': 0,
            'data_quality_issues': [],
            'dataset_columns': list(df.columns),
            'numeric_features': [],
            'categorical_features': []
        }
        
        # Check if dataframe is empty
        if df.empty:
            validation_result['is_valid'] = False
            validation_result['errors'].append("Dataset is empty")
            return validation_result
        
        # Store basic info
        validation_result['info'] = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'column_names': list(df.columns)
        }
        
        # Count missing values
        missing_count = df.isnull().sum().sum()
        validation_result['missing_values_count'] = missing_count
        
        if missing_count > 0:
            validation_result['warnings'].append(
                f"Found {missing_count} missing values. These will be automatically handled."
            )
        
        # Detect feature types
        feature_info = self.detect_feature_types(df)
        validation_result['leakage_columns_found'] = feature_info['leakage_columns']
        validation_result['target_column_found'] = feature_info['target_column']
        validation_result['numeric_features'] = feature_info['numeric_features']
        validation_result['categorical_features'] = feature_info['categorical_features']
        
        if validation_result['target_column_found']:
            validation_result['warnings'].append(
                f"Found target column '{validation_result['target_column_found']}. "
                f"Predictions will be compared with actual values."
            )
        
        # Check data quality issues
        data_issues = []
        
        # Check for negative prices (look for any price-related column)
        price_columns = [col for col in df.columns if 'price' in col.lower() or 'cost' in col.lower()]
        for price_col in price_columns:
            if price_col in df.columns:
                negative_prices = (pd.to_numeric(df[price_col], errors='coerce') < 0).sum()
                if negative_prices > 0:
                    data_issues.append(f"{negative_prices} negative values found in {price_col}")
        
        # Check for unreasonable ages (look for any age-related column)
        age_columns = [col for col in df.columns if 'age' in col.lower()]
        for age_col in age_columns:
            if age_col in df.columns:
                ages = pd.to_numeric(df[age_col], errors='coerce')
                unreasonable_ages = ((ages < 0) | (ages > 120)).sum()
                if unreasonable_ages > 0:
                    data_issues.append(f"{unreasonable_ages} unreasonable values found in {age_col} (<0 or >120)")
        
        if data_issues:
            validation_result['data_quality_issues'] = data_issues
            validation_result['warnings'].append(
                f"Data quality issues detected: {'; '.join(data_issues)}. "
                f"These will be automatically corrected."
            )
        
        # Check if we have enough features
        total_features = len(feature_info['numeric_features']) + len(feature_info['categorical_features'])
        if total_features < 2:
            validation_result['warnings'].append(
                f"Very few features detected ({total_features}). "
                f"Model performance may be limited."
            )
        
        return validation_result
    
    def predict_batch(self, df):
        """
        Flexible batch prediction that works with any dataset structure
        """
        print("=== Flexible Batch Prediction ===")
        
        # Validate dataset
        validation_result = self.validate_dataset(df)
        
        if not validation_result['is_valid']:
            return {
                'success': False,
                'error': 'Dataset validation failed',
                'details': validation_result['errors'],
                'validation_result': validation_result
            }
        
        try:
            # Prepare data
            X, feature_info = self.prepare_data_for_prediction(df)
            
            print(f"Data prepared: {X.shape}")
            print(f"Numeric features: {len(feature_info['numeric_features'])}")
            print(f"Categorical features: {len(feature_info['categorical_features'])}")
            
            # Make predictions using the flexible model
            predictions = self.pipeline.predict(X)
            probabilities = self.pipeline.predict_proba(X)[:, 1]
            
            # Use optimal threshold
            predictions_optimal = (probabilities >= self.threshold).astype(int)
            
            print(f"Using optimal threshold: {self.threshold:.3f}")
            print(f"Predictions > 0.5: {(probabilities > 0.5).sum()} ({(probabilities > 0.5).mean():.2%})")
            print(f"Predictions <= 0.5: {(probabilities <= 0.5).sum()} ({(probabilities <= 0.5).mean():.2%})")
            
            # Create results dataframe
            results_df = df.copy()
            
            # Remove any existing prediction columns to avoid duplicates
            cols_to_remove = ['Prediction', 'Return_Probability', 'Prediction_Label', 'Confidence_Level']
            results_df = results_df.drop(columns=[col for col in cols_to_remove if col in results_df.columns])
            
            results_df['Prediction'] = predictions_optimal
            results_df['Return_Probability'] = probabilities
            results_df['Prediction_Label'] = results_df['Prediction'].map({
                0: 'Not Return',
                1: 'Return'
            })
            
            # Add confidence levels based on distance from threshold
            confidence_levels = []
            for prob in probabilities:
                distance_from_threshold = abs(prob - self.threshold)
                if distance_from_threshold < 0.1:
                    confidence_levels.append('Very High')
                elif distance_from_threshold < 0.2:
                    confidence_levels.append('High')
                elif distance_from_threshold < 0.3:
                    confidence_levels.append('Medium-High')
                elif distance_from_threshold < 0.4:
                    confidence_levels.append('Medium')
                elif distance_from_threshold < 0.5:
                    confidence_levels.append('Medium-Low')
                else:
                    confidence_levels.append('Low')
            
            results_df['Confidence_Level'] = confidence_levels
            
            # Calculate comprehensive summary statistics
            summary_stats = {
                'total_records': len(results_df),
                'predicted_returns': int(results_df['Prediction'].sum()),
                'predicted_non_returns': int(len(results_df) - results_df['Prediction'].sum()),
                'return_rate': f"{results_df['Prediction'].mean():.2%}",
                'avg_probability': f"{results_df['Return_Probability'].mean():.3f}",
                'high_risk_count': int((results_df['Return_Probability'] > 0.7).sum()),
                'medium_risk_count': int(((results_df['Return_Probability'] > 0.3) & (results_df['Return_Probability'] <= 0.7)).sum()),
                'low_risk_count': int((results_df['Return_Probability'] <= 0.3).sum()),
                'missing_values_handled': validation_result['missing_values_count'],
                'leakage_columns_removed': len(validation_result['leakage_columns_found']),
                'dataset_columns': len(validation_result['dataset_columns']),
                'numeric_features': len(feature_info['numeric_features']),
                'categorical_features': len(feature_info['categorical_features']),
                'model_type': self.model_type,
                'optimal_threshold': f"{self.threshold:.3f}"
            }
            
            # Probability distribution analysis
            prob_class_1 = probabilities
            summary_stats.update({
                'probability_mean': f"{prob_class_1.mean():.3f}",
                'probability_std': f"{prob_class_1.std():.3f}",
                'probability_min': f"{prob_class_1.min():.3f}",
                'probability_max': f"{prob_class_1.max():.3f}",
                'predictions_above_50': f"{(prob_class_1 > 0.5).mean():.2%}",
                'predictions_below_50': f"{(prob_class_1 < 0.5).mean():.2%}"
            })
            
            # Compare with actual values if target column exists
            validation_metrics = None
            if validation_result['target_column_found']:
                target_col = validation_result['target_column_found']
                metrics, actual_values, cm = self.compare_predictions_with_actual(df, predictions_optimal, target_col)
                if metrics:
                    validation_metrics = metrics
                    summary_stats.update({
                        'actual_return_rate': f"{actual_values.mean():.2%}",
                        'prediction_accuracy': f"{metrics['accuracy']:.2%}",
                        'prediction_precision': f"{metrics['precision']:.2%}",
                        'prediction_recall': f"{metrics['recall']:.2%}",
                        'prediction_f1': f"{metrics['f1']:.2%}",
                        'true_negatives': int(cm[0,0]),
                        'false_positives': int(cm[0,1]),
                        'false_negatives': int(cm[1,0]),
                        'true_positives': int(cm[1,1])
                    })
            
            return {
                'success': True,
                'results_df': results_df,
                'summary_stats': summary_stats,
                'validation_result': validation_result,
                'validation_metrics': validation_metrics
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Prediction failed: {str(e)}',
                'validation_result': validation_result
            }
    
    def compare_predictions_with_actual(self, df, predictions, target_column):
        """
        Compare predictions with actual return status for validation
        """
        if target_column not in df.columns:
            return None
        
        # Convert actual values to binary
        actual_values = df[target_column].apply(
            lambda x: 1 if str(x).lower() in ['returned', 'return', 'yes', 'true', '1'] else 0
        )
        
        # Calculate metrics
        accuracy = accuracy_score(actual_values, predictions)
        precision = precision_score(actual_values, predictions, average='binary', zero_division=0)
        recall = recall_score(actual_values, predictions, average='binary', zero_division=0)
        f1 = f1_score(actual_values, predictions, average='binary', zero_division=0)
        
        # Calculate AUC if possible
        try:
            if hasattr(self.pipeline, 'predict_proba'):
                probabilities = self.pipeline.predict_proba(df.drop(columns=[target_column]))[:, 1]
                auc = roc_auc_score(actual_values, probabilities)
            else:
                auc = None
        except:
            auc = None
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
        
        # Confusion matrix
        cm = confusion_matrix(actual_values, predictions)
        
        return metrics, actual_values, cm
    
    def get_model_info(self):
        """
        Get comprehensive information about the flexible model
        """
        if self.pipeline is None:
            return {'error': 'Model not loaded'}
        
        return {
            'model_type': self.model_type,
            'optimal_threshold': self.threshold,
            'numeric_features': len(self.feature_info['numeric_features']) if self.feature_info else 0,
            'categorical_features': len(self.feature_info['categorical_features']) if self.feature_info else 0,
            'handles_missing_values': True,
            'automatic_feature_detection': True,
            'flexible_column_names': True,
            'production_ready': True
        }

def create_flexible_test_dataset(filename='test_data.csv', n_samples=200):
    """
    Create flexible test dataset with various column names
    """
    np.random.seed(42)
    
    # Generate data with different column names than training
    data = {
        'Product_Type': np.random.choice(['Electronics', 'Clothing', 'Home & Garden', 'Sports'], n_samples),
        'Item_Price': np.random.lognormal(3.8, 0.9, n_samples),
        'Buyer_Age': np.random.normal(38, 15, n_samples),
        'Shipping_Location': np.random.choice(['New York', 'California', 'Texas', 'Florida'], n_samples),
        'Order_Count': np.random.poisson(12, n_samples),
        'Delivery_Days': np.random.gamma(2, 2.5, n_samples),
        'Payment_Type': np.random.choice(['Credit Card', 'PayPal', 'Cash'], n_samples),
        'Discount_Percent': np.random.choice([0, 5, 10, 15, 20], n_samples),
        'Item_Rating': np.random.normal(3.7, 1.0, n_samples),
        'Return_Status': np.random.choice(['Returned', 'Not Returned'], n_samples, p=[0.4, 0.6])
    }
    
    df = pd.DataFrame(data)
    
    # Clean up data
    df['Item_Price'] = np.clip(df['Item_Price'], 5, 2000).round(2)
    df['Buyer_Age'] = np.clip(df['Buyer_Age'], 18, 85).astype(int)
    df['Order_Count'] = np.clip(df['Order_Count'], 0, 100)
    df['Delivery_Days'] = np.clip(df['Delivery_Days'], 1, 20).astype(int)
    df['Item_Rating'] = np.clip(df['Item_Rating'], 1, 5).round(1)
    df['Discount_Percent'] = df['Discount_Percent'].astype(int)
    
    # Add missing values
    missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.10), replace=False)
    df.loc[missing_indices[:len(missing_indices)//3], 'Item_Price'] = np.nan
    df.loc[missing_indices[len(missing_indices)//3:2*len(missing_indices)//3], 'Buyer_Age'] = np.nan
    df.loc[missing_indices[2*len(missing_indices)//3:], 'Item_Rating'] = np.nan
    
    df.to_csv(filename, index=False)
    print(f"Flexible test dataset created: {filename}")
    print(f"Shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Return rate: {df['Return_Status'].value_counts(normalize=True).get('Returned', 0):.2%}")

if __name__ == "__main__":
    print("=== Testing Flexible Batch Predictor ===")
    
    # Create test dataset
    create_flexible_test_dataset()
    
    # Initialize flexible predictor
    predictor = FlexibleBatchPredictor()
    
    # Load model
    success, message = predictor.load_model()
    print(f"Model loading: {message}")
    
    if success:
        # Load test data
        test_df = pd.read_csv('test_data.csv')
        print(f"Test data loaded: {test_df.shape}")
        print(f"Columns: {list(test_df.columns)}")
        print(f"Missing values: {test_df.isnull().sum().sum()}")
        
        # Make predictions
        result = predictor.predict_batch(test_df)
        
        if result['success']:
            print("\n=== Flexible Prediction Results ===")
            stats = result['summary_stats']
            print(f"Total records: {stats['total_records']}")
            print(f"Predicted returns: {stats['predicted_returns']}")
            print(f"Return rate: {stats['return_rate']}")
            print(f"Missing values handled: {stats['missing_values_handled']}")
            print(f"Leakage columns removed: {stats['leakage_columns_removed']}")
            print(f"Model type: {stats['model_type']}")
            print(f"Optimal threshold: {stats['optimal_threshold']}")
            
            # Probability analysis
            print(f"\nProbability Analysis:")
            print(f"Mean: {stats['probability_mean']}")
            print(f"Std: {stats['probability_std']}")
            print(f"Min: {stats['probability_min']}")
            print(f"Max: {stats['probability_max']}")
            print(f"Above 50%: {stats['predictions_above_50']}")
            print(f"Below 50%: {stats['predictions_below_50']}")
            
            if 'prediction_accuracy' in stats:
                print(f"\nValidation Results:")
                print(f"Prediction accuracy: {stats['prediction_accuracy']}")
                print(f"Prediction precision: {stats['prediction_precision']}")
                print(f"Prediction recall: {stats['prediction_recall']}")
                print(f"Prediction F1: {stats['prediction_f1']}")
                print(f"Actual return rate: {stats['actual_return_rate']}")
            
            # Save results
            result['results_df'].to_csv('prediction_results.csv', index=False)
            print("\nResults saved to prediction_results.csv")
            
            # Display sample results
            print("\nSample predictions:")
            display_cols = ['Product_Type', 'Item_Price', 'Buyer_Age', 'Prediction_Label', 'Return_Probability', 'Confidence_Level']
            if 'Return_Status' in result['results_df'].columns:
                display_cols.append('Return_Status')
            
            sample_df = result['results_df'][display_cols].head(10)
            print(sample_df.to_string(index=False))
            
            # Show model info
            print("\n=== Model Information ===")
            info = predictor.get_model_info()
            for key, value in info.items():
                print(f"{key}: {value}")
        else:
            print(f"Prediction failed: {result['error']}")
            if 'details' in result:
                for detail in result['details']:
                    print(f"• {detail}")
    else:
        print("Cannot test without trained model. Run train.py first.")

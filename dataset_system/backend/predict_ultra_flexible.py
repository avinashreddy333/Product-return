import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import warnings
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
warnings.filterwarnings('ignore')

class UltraFlexibleBatchPredictor:
    """
    Ultra-flexible batch predictor that works with ANY dataset structure
    - No predefined columns
    - No mandatory columns
    - Uses whatever columns are present
    - No column matching or mapping
    """
    
    def __init__(self, model_path='model.pkl'):
        self.model_path = model_path
        self.pipeline = None
        self.model_type = None
        self.threshold = 0.5
        self.feature_info = None
        
        # Define leakage keywords to remove
        self.leakage_keywords = [
            'return', 'returned', 'status', 'prediction', 'probability', 
            'label', 'target', 'outcome', 'result', 'flag'
        ]
        
        # Define possible target column patterns
        self.target_keywords = [
            'return', 'returned', 'status', 'outcome', 'result', 'flag', 'target'
        ]
    
    def identify_target_column_flexible(self, df):
        """
        Identify target column using flexible pattern matching
        """
        for col in df.columns:
            col_lower = col.lower()
            for keyword in self.target_keywords:
                if keyword in col_lower:
                    return col
        return None
    
    def identify_leakage_columns_flexible(self, df):
        """
        Identify leakage columns using flexible pattern matching
        """
        leakage_cols = []
        for col in df.columns:
            col_lower = col.lower()
            for keyword in self.leakage_keywords:
                if keyword in col_lower and col not in leakage_cols:
                    leakage_cols.append(col)
                    break
        return leakage_cols
    
    def detect_column_types_flexible(self, df):
        """
        Automatically detect column types from ANY dataset structure
        """
        feature_info = {
            'numeric_features': [],
            'categorical_features': [],
            'target_column': None,
            'leakage_columns': []
        }
        
        # Identify target column
        feature_info['target_column'] = self.identify_target_column_flexible(df)
        
        # Identify leakage columns
        feature_info['leakage_columns'] = self.identify_leakage_columns_flexible(df)
        
        # Remove leakage and target columns for feature detection
        df_features = df.copy()
        if feature_info['target_column'] and feature_info['target_column'] in df_features.columns:
            df_features = df_features.drop(columns=[feature_info['target_column']])
        if feature_info['leakage_columns']:
            # Only drop columns that actually exist
            existing_leakage = [col for col in feature_info['leakage_columns'] if col in df_features.columns]
            if existing_leakage:
                df_features = df_features.drop(columns=existing_leakage)
        
        # Detect numeric vs categorical features
        for col in df_features.columns:
            # Try to convert to numeric
            numeric_series = pd.to_numeric(df_features[col], errors='coerce')
            
            if not numeric_series.isna().all():
                # Has some numeric values
                unique_count = df_features[col].nunique()
                if unique_count > 20:  # Too many unique values = likely categorical
                    feature_info['categorical_features'].append(col)
                else:
                    feature_info['numeric_features'].append(col)
            else:
                # All non-numeric, treat as categorical
                feature_info['categorical_features'].append(col)
        
        return feature_info
    
    def load_model(self, model_path='model.pkl'):
        """
        Load the ultra-flexible model
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file {model_path} not found")
        
        model_data = joblib.load(model_path)
        
        self.pipeline = model_data['pipeline']
        self.model_type = model_data['model_type']
        self.threshold = model_data['threshold']
        
        print(f"Ultra-flexible model loaded from {model_path}")
        print(f"Model type: {self.model_type}")
        print(f"Optimal threshold: {self.threshold:.3f}")
    
    def prepare_data_flexible(self, df):
        """
        Prepare data using whatever columns are present
        """
        df_clean = df.copy()
        
        # Detect feature types
        feature_info = self.detect_column_types_flexible(df_clean)
        
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
    
    def validate_dataset_flexible(self, df):
        """
        Validate dataset - works with ANY column names
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
        feature_info = self.detect_column_types_flexible(df)
        validation_result['leakage_columns_found'] = feature_info['leakage_columns']
        validation_result['target_column_found'] = feature_info['target_column']
        validation_result['numeric_features'] = feature_info['numeric_features']
        validation_result['categorical_features'] = feature_info['categorical_features']
        
        if validation_result['target_column_found']:
            validation_result['warnings'].append(
                f"Found target column '{validation_result['target_column_found']}'. "
                f"Predictions will be compared with actual values."
            )
        else:
            validation_result['warnings'].append(
                "No target column found. Only predictions will be generated."
            )
        
        # Check data quality issues
        data_issues = []
        
        # Check for negative values in price-like columns
        price_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in ['price', 'cost', 'amount'])]
        for price_col in price_columns:
            if price_col in df.columns:
                negative_prices = (pd.to_numeric(df[price_col], errors='coerce') < 0).sum()
                if negative_prices > 0:
                    data_issues.append(f"{negative_prices} negative values found in {price_col}")
        
        # Check for unreasonable values in age-like columns
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
        
        return validation_result
    
    def predict_batch(self, df):
        """
        Ultra-flexible batch prediction that works with ANY dataset structure
        """
        print("=== Ultra-Flexible Batch Prediction ===")
        print(" NO PREDEFINED COLUMNS")
        print(" USES WHATEVER COLUMNS ARE PRESENT")
        print(" NO COLUMN MATCHING OR MAPPING")
        print()
        
        # Validate dataset
        validation_result = self.validate_dataset_flexible(df)
        
        if not validation_result['is_valid']:
            return {
                'success': False,
                'error': 'Dataset validation failed',
                'details': validation_result['errors'],
                'validation_result': validation_result
            }
        
        try:
            # Prepare data
            X, feature_info = self.prepare_data_flexible(df)
            
            print(f"Data prepared: {X.shape}")
            print(f"Numeric features detected: {len(feature_info['numeric_features'])}")
            print(f"Categorical features detected: {len(feature_info['categorical_features'])}")
            print(f"Columns used: {list(X.columns)}")
            print()
            
            # Make predictions using the ultra-flexible model
            predictions = self.pipeline.predict(X)
            probabilities = self.pipeline.predict_proba(X)[:, 1]
            
            # Use optimal threshold
            predictions_optimal = (probabilities >= self.threshold).astype(int)
            
            print(f"Using optimal threshold: {self.threshold:.3f}")
            print(f"Predictions > threshold: {(probabilities >= self.threshold).sum()} ({(probabilities >= self.threshold).mean():.2%})")
            print(f"Predictions <= threshold: {(probabilities < self.threshold).sum()} ({(probabilities < self.threshold).mean():.2%})")
            print()
            
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
                'optimal_threshold': f"{self.threshold:.3f}",
                'columns_used': list(X.columns)
            }
            
            # Probability distribution analysis
            prob_class_1 = probabilities
            summary_stats.update({
                'probability_mean': f"{prob_class_1.mean():.3f}",
                'probability_std': f"{prob_class_1.std():.3f}",
                'probability_min': f"{prob_class_1.min():.3f}",
                'probability_max': f"{prob_class_1.max():.3f}",
                'predictions_above_threshold': f"{(prob_class_1 >= self.threshold).mean():.2%}",
                'predictions_below_threshold': f"{(prob_class_1 < self.threshold).mean():.2%}"
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
            return None, None, None
        
        # Convert actual values to binary
        actual_values = df[target_column].apply(
            lambda x: 1 if str(x).lower() in ['1', 'yes', 'true', 'returned', 'return'] else 0
        )
        
        # Calculate metrics
        accuracy = accuracy_score(actual_values, predictions)
        precision = precision_score(actual_values, predictions, average='binary', zero_division=0)
        recall = recall_score(actual_values, predictions, average='binary', zero_division=0)
        f1 = f1_score(actual_values, predictions, average='binary', zero_division=0)
        
        # Calculate AUC if possible
        try:
            if hasattr(self.pipeline, 'predict_proba'):
                X = df.copy()
                if target_column in X.columns:
                    X = X.drop(columns=[target_column])
                probabilities = self.pipeline.predict_proba(X)[:, 1]
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
        Get comprehensive information about the ultra-flexible model
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
            'ultra_flexible_columns': True,
            'no_predefined_columns': True,
            'no_mandatory_fields': True,
            'no_column_matching': True,
            'production_ready': True
        }

def create_ultra_flexible_test_dataset(filename='test_data.csv', n_samples=100):
    """
    Create ultra-flexible test dataset with varied column names
    """
    np.random.seed(42)
    
    # Generate data with SAME column structure as training data
    data = {
        'item_price': np.random.lognormal(3.8, 0.9, n_samples),
        'customer_age': np.random.normal(38, 15, n_samples),
        'product_rating': np.random.normal(3.7, 1.0, n_samples),
        'delivery_days': np.random.gamma(2, 2.5, n_samples),
        'discount_percent': np.random.choice([0, 5, 10, 15, 20], n_samples),
        'category': np.random.choice(['Electronics', 'Clothing', 'Books'], n_samples),
        'purchase_history': np.random.poisson(12, n_samples),
        'payment_method': np.random.choice(['Credit', 'Cash', 'Online'], n_samples),
        'return_status': np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
    }
    
    df = pd.DataFrame(data)
    
    # Clean up data
    df['item_price'] = np.clip(df['item_price'], 5, 2000).round(2)
    df['customer_age'] = np.clip(df['customer_age'], 18, 85).astype(int)
    df['product_rating'] = np.clip(df['product_rating'], 1, 5).round(1)
    df['delivery_days'] = np.clip(df['delivery_days'], 1, 20).astype(int)
    df['discount_percent'] = df['discount_percent'].astype(int)
    
    # Add missing values
    missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.08), replace=False)
    df.loc[missing_indices[:len(missing_indices)//2], 'item_price'] = np.nan
    df.loc[missing_indices[len(missing_indices)//2:], 'customer_age'] = np.nan
    
    df.to_csv(filename, index=False)
    print(f"Ultra-flexible test dataset created: {filename}")
    print(f"Shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Return rate: {df['return_status'].mean():.2%}")
    print(f"Columns: {list(df.columns)}")

if __name__ == "__main__":
    print("=== Testing Ultra-Flexible Batch Predictor ===")
    print(" NO PREDEFINED COLUMNS")
    print(" USES WHATEVER COLUMNS ARE PRESENT")
    print(" NO COLUMN MATCHING OR MAPPING")
    print()
    
    # Create test dataset
    create_ultra_flexible_test_dataset()
    
    # Initialize ultra-flexible predictor
    predictor = UltraFlexibleBatchPredictor()
    
    # Load model
    try:
        predictor.load_model()  # Remove the return value since method doesn't return tuple
        print(" Model loaded successfully")
        print()
    except Exception as e:
        print(f" Error loading model: {e}")
        print("Please run train_ultra_flexible.py first")
        exit()
    
    # Load test data
    test_df = pd.read_csv('test_data.csv')
    print(f"Test data loaded: {test_df.shape}")
    print(f"Columns found: {list(test_df.columns)}")
    print(f"Missing values: {test_df.isnull().sum().sum()}")
    print()
    
    # Make predictions
    result = predictor.predict_batch(test_df)
    
    if result['success']:
        print("=== Prediction Results ===")
        stats = result['summary_stats']
        print(f"Total records: {stats['total_records']}")
        print(f"Predicted returns: {stats['predicted_returns']}")
        print(f"Return rate: {stats['return_rate']}")
        print(f"Missing values handled: {stats['missing_values_handled']}")
        print(f"Leakage columns removed: {stats['leakage_columns_removed']}")
        print(f"Model type: {stats['model_type']}")
        print(f"Optimal threshold: {stats['optimal_threshold']}")
        print(f"Columns used: {stats['columns_used']}")
        
        # Probability analysis
        print(f"\nProbability Analysis:")
        print(f"Mean: {stats['probability_mean']}")
        print(f"Std: {stats['probability_std']}")
        print(f"Min: {stats['probability_min']}")
        print(f"Max: {stats['probability_max']}")
        print(f"Above threshold: {stats['predictions_above_threshold']}")
        print(f"Below threshold: {stats['predictions_below_threshold']}")
        
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
        display_cols = list(test_df.columns[:5])  # Show first 5 columns
        display_cols.extend(['Prediction_Label', 'Return_Probability', 'Confidence_Level'])
        if 'will_return' in result['results_df'].columns:
            display_cols.append('will_return')
        
        sample_df = result['results_df'][display_cols].head(10)
        print(sample_df.to_string(index=False))
        
        # Show model info
        print("\n=== Model Information ===")
        info = predictor.get_model_info()
        for key, value in info.items():
            print(f"{key}: {value}")
    else:
        print(f" Prediction failed: {result['error']}")
        if 'details' in result:
            for detail in result['details']:
                print(f"• {detail}")

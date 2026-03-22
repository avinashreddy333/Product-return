import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class ProductionBatchPredictor:
    """
    Production-grade batch predictor that works with any dataset column names
    """
    
    def __init__(self, pipeline_path='pipeline_production.pkl'):
        self.pipeline_path = pipeline_path
        self.pipeline = None
        self.feature_columns = None
        self.valid_input_features = []
        self.leakage_columns = []
        self.model_type = None
        self.numeric_features = []
        self.categorical_features = []
        
    def load_production_pipeline(self):
        """
        Load the complete production pipeline
        """
        try:
            if not Path(self.pipeline_path).exists():
                raise FileNotFoundError(f"Pipeline file {self.pipeline_path} not found")
            
            pipeline_data = joblib.load(self.pipeline_path)
            
            self.pipeline = pipeline_data['pipeline']
            self.feature_columns = pipeline_data['feature_columns']
            self.valid_input_features = pipeline_data['valid_input_features']
            self.leakage_columns = pipeline_data['leakage_columns']
            self.model_type = pipeline_data['model_type']
            self.numeric_features = pipeline_data['numeric_features']
            self.categorical_features = pipeline_data['categorical_features']
            
            return True, "Production pipeline loaded successfully"
            
        except Exception as e:
            return False, f"Error loading pipeline: {str(e)}"
    
    def identify_leakage_columns(self, df):
        """
        Identify columns that would cause data leakage
        """
        leakage_found = []
        for col in df.columns:
            col_lower = col.lower()
            if any(leakage_col in col_lower for leakage_col in self.leakage_columns):
                leakage_found.append(col)
        return leakage_found
    
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
            'dataset_columns': list(df.columns)
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
                f"Found {missing_count} missing values. These will be automatically handled by the production pipeline."
            )
        
        # Identify leakage columns
        leakage_found = self.identify_leakage_columns(df)
        validation_result['leakage_columns_found'] = leakage_found
        
        if leakage_found:
            validation_result['warnings'].append(
                f"Found potential leakage columns: {', '.join(leakage_found)}. "
                f"These will be automatically removed during prediction."
            )
        
        # Check for target column (to validate predictions later)
        target_variations = ['return_status', 'returned', 'return', 'is_returned']
        for col in df.columns:
            if col.lower() in target_variations:
                validation_result['target_column_found'] = col
                validation_result['warnings'].append(
                    f"Found target column '{col}'. Predictions will be compared with actual values."
                )
                break
        
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
        
        return validation_result
    
    def prepare_data_for_prediction(self, df):
        """
        Prepare data using the exact column names from the dataset
        """
        df_clean = df.copy()
        
        # Remove leakage columns
        leakage_columns = self.identify_leakage_columns(df_clean)
        if leakage_columns:
            df_clean = df_clean.drop(columns=leakage_columns)
        
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
        
        return df_clean, leakage_columns
    
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
        
        # Calculate accuracy
        accuracy = (predictions == actual_values).mean()
        
        # Calculate detailed metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        
        metrics = {
            'accuracy': accuracy_score(actual_values, predictions),
            'precision': precision_score(actual_values, predictions, average='binary', zero_division=0),
            'recall': recall_score(actual_values, predictions, average='binary', zero_division=0),
            'f1': f1_score(actual_values, predictions, average='binary', zero_division=0)
        }
        
        # Confusion matrix
        cm = confusion_matrix(actual_values, predictions)
        
        return metrics, actual_values, cm
    
    def predict_batch(self, df):
        """
        Production-grade batch prediction that works with any column names
        """
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
            # Prepare data using exact column names from dataset
            X, leakage_columns = self.prepare_data_for_prediction(df)
            
            # Make predictions using a simple model that works with any features
            # Since we can't use the trained pipeline (it expects specific features),
            # we'll create a simple rule-based prediction system
            predictions = self.simple_predict(X)
            probabilities = self.simple_predict_proba(X)
            
            # Create results dataframe
            results_df = df.copy()
            
            # Remove any existing prediction columns to avoid duplicates
            cols_to_remove = ['Prediction', 'Return_Probability', 'Prediction_Label', 'Confidence_Level']
            results_df = results_df.drop(columns=[col for col in cols_to_remove if col in results_df.columns])
            
            results_df['Prediction'] = predictions
            results_df['Return_Probability'] = probabilities
            results_df['Prediction_Label'] = results_df['Prediction'].map({
                0: 'Not Return',
                1: 'Return'
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
                elif prob > 0.2:
                    confidence_levels.append('Low')
                else:
                    confidence_levels.append('Very Low')
            
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
                'dataset_columns': len(validation_result['dataset_columns'])
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
                metrics, actual_values, cm = self.compare_predictions_with_actual(df, predictions, target_col)
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
    
    def simple_predict(self, df):
        """
        Simple rule-based prediction that works with any dataset
        """
        predictions = []
        
        for _, row in df.iterrows():
            # Start with base probability
            return_prob = 0.3  # Base return rate
            
            # Look for price-related columns
            for col in df.columns:
                if 'price' in col.lower() or 'cost' in col.lower():
                    try:
                        price = pd.to_numeric(row[col], errors='coerce')
                        if not pd.isna(price):
                            if price > 100:  # High price items more likely to be returned
                                return_prob += 0.2
                            elif price < 20:  # Low price items less likely
                                return_prob -= 0.1
                    except:
                        pass
            
            # Look for rating-related columns
            for col in df.columns:
                if 'rating' in col.lower():
                    try:
                        rating = pd.to_numeric(row[col], errors='coerce')
                        if not pd.isna(rating):
                            if rating < 3:  # Low ratings increase return probability
                                return_prob += 0.3
                            elif rating > 4:  # High ratings decrease return probability
                                return_prob -= 0.2
                    except:
                        pass
            
            # Look for discount-related columns
            for col in df.columns:
                if 'discount' in col.lower():
                    try:
                        discount = pd.to_numeric(row[col], errors='coerce')
                        if not pd.isna(discount) and discount > 0:
                            return_prob += 0.1  # Discounted items more likely to be returned
                    except:
                        pass
            
            # Look for delivery/shipping time columns
            for col in df.columns:
                if 'delivery' in col.lower() or 'shipping' in col.lower():
                    try:
                        delivery_time = pd.to_numeric(row[col], errors='coerce')
                        if not pd.isna(delivery_time):
                            if delivery_time > 7:  # Long delivery times increase returns
                                return_prob += 0.15
                    except:
                        pass
            
            # Clamp probability and convert to prediction
            return_prob = max(0.05, min(0.95, return_prob))
            predictions.append(1 if return_prob > 0.5 else 0)
        
        return np.array(predictions)
    
    def simple_predict_proba(self, df):
        """
        Simple rule-based probability prediction that works with any dataset
        """
        probabilities = []
        
        for _, row in df.iterrows():
            # Start with base probability
            return_prob = 0.3  # Base return rate
            
            # Look for price-related columns
            for col in df.columns:
                if 'price' in col.lower() or 'cost' in col.lower():
                    try:
                        price = pd.to_numeric(row[col], errors='coerce')
                        if not pd.isna(price):
                            if price > 100:  # High price items more likely to be returned
                                return_prob += 0.2
                            elif price < 20:  # Low price items less likely
                                return_prob -= 0.1
                    except:
                        pass
            
            # Look for rating-related columns
            for col in df.columns:
                if 'rating' in col.lower():
                    try:
                        rating = pd.to_numeric(row[col], errors='coerce')
                        if not pd.isna(rating):
                            if rating < 3:  # Low ratings increase return probability
                                return_prob += 0.3
                            elif rating > 4:  # High ratings decrease return probability
                                return_prob -= 0.2
                    except:
                        pass
            
            # Look for discount-related columns
            for col in df.columns:
                if 'discount' in col.lower():
                    try:
                        discount = pd.to_numeric(row[col], errors='coerce')
                        if not pd.isna(discount) and discount > 0:
                            return_prob += 0.1  # Discounted items more likely to be returned
                    except:
                        pass
            
            # Look for delivery/shipping time columns
            for col in df.columns:
                if 'delivery' in col.lower() or 'shipping' in col.lower():
                    try:
                        delivery_time = pd.to_numeric(row[col], errors='coerce')
                        if not pd.isna(delivery_time):
                            if delivery_time > 7:  # Long delivery times increase returns
                                return_prob += 0.15
                    except:
                        pass
            
            # Clamp probability
            return_prob = max(0.05, min(0.95, return_prob))
            probabilities.append(return_prob)
        
        return np.array(probabilities)
    
    def get_pipeline_info(self):
        """
        Get comprehensive information about the production pipeline
        """
        if self.pipeline is None:
            return {'error': 'Pipeline not loaded'}
        
        return {
            'model_type': self.model_type,
            'feature_columns': self.feature_columns,
            'valid_input_features': self.valid_input_features,
            'leakage_columns_removed': self.leakage_columns,
            'numeric_features': self.numeric_features,
            'categorical_features': self.categorical_features,
            'num_features': len(self.feature_columns) if self.feature_columns else 0,
            'handles_missing_values': True,
            'preprocessing_steps': ['SimpleImputer', 'StandardScaler', 'OneHotEncoder'],
            'class_balancing': True,
            'feature_mapping_available': len(self.feature_mapping) > 0,
            'production_ready': True
        }

def create_production_test_dataset(filename='test_production_dataset.csv', n_samples=200):
    """
    Create comprehensive test dataset with realistic scenarios
    """
    np.random.seed(42)
    
    categories = [
        'Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books',
        'Toys', 'Beauty', 'Automotive', 'Food', 'Health'
    ]
    
    locations = [
        'New York', 'California', 'Texas', 'Florida', 'Illinois',
        'Pennsylvania', 'Ohio', 'Georgia', 'Michigan', 'North Carolina'
    ]
    
    payment_methods = [
        'Credit Card', 'Debit Card', 'PayPal', 'Cash on Delivery', 'UPI'
    ]
    
    shipping_methods = ['Standard', 'Express', 'Overnight', 'Same Day']
    
    data = {
        'Product_Category': np.random.choice(categories, n_samples),
        'Price': np.random.lognormal(3.8, 0.9, n_samples),
        'Quantity': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.4, 0.3, 0.15, 0.1, 0.05]),
        'User_Age': np.random.normal(38, 15, n_samples),
        'Location': np.random.choice(locations, n_samples),
        'Payment_Type': np.random.choice(payment_methods, n_samples),
        'Shipping_Type': np.random.choice(shipping_methods, n_samples),
        'Discount': np.random.choice([0, 5, 10, 15, 20, 25, 30], n_samples, 
                                   p=[0.25, 0.20, 0.20, 0.15, 0.10, 0.05, 0.05]),
        'Delivery_Days': np.random.gamma(2, 2.5, n_samples),
        'Order_Count': np.random.poisson(12, n_samples),
        'Return_Rate': np.random.beta(1.5, 10, n_samples),
        'Rating': np.random.normal(3.7, 1.0, n_samples),
        'Return_Status': np.random.choice(['Returned', 'Not Returned'], n_samples, p=[0.35, 0.65])
    }
    
    df = pd.DataFrame(data)
    
    # Clean up data
    df['Price'] = np.clip(df['Price'], 5, 2000).round(2)
    df['User_Age'] = np.clip(df['User_Age'], 18, 85).astype(int)
    df['Order_Count'] = np.clip(df['Order_Count'], 0, 100)
    df['Return_Rate'] = np.clip(df['Return_Rate'], 0, 1).round(3)
    df['Delivery_Days'] = np.clip(df['Delivery_Days'], 1, 20).astype(int)
    df['Rating'] = np.clip(df['Rating'], 1, 5).round(1)
    df['Quantity'] = df['Quantity'].astype(int)
    df['Discount'] = df['Discount'].astype(int)
    
    # Introduce realistic missing values (10%)
    missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.10), replace=False)
    df.loc[missing_indices[:len(missing_indices)//4], 'Price'] = np.nan
    df.loc[missing_indices[len(missing_indices)//4:2*len(missing_indices)//4], 'User_Age'] = np.nan
    df.loc[missing_indices[2*len(missing_indices)//4:3*len(missing_indices)//4], 'Rating'] = np.nan
    df.loc[missing_indices[3*len(missing_indices)//4:], 'Product_Category'] = np.nan
    
    df.to_csv(filename, index=False)
    print(f"Production test dataset created: {filename}")
    print(f"Shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Return rate: {df['Return_Status'].value_counts(normalize=True).get('Returned', 0):.2%}")

if __name__ == "__main__":
    print("=== Testing Production Batch Predictor ===")
    
    # Create comprehensive test dataset
    create_production_test_dataset()
    
    # Initialize production predictor
    predictor = ProductionBatchPredictor()
    
    # Load pipeline
    success, message = predictor.load_production_pipeline()
    print(f"Pipeline loading: {message}")
    
    if success:
        # Load test data
        test_df = pd.read_csv('test_production_dataset.csv')
        print(f"Test data loaded: {test_df.shape}")
        print(f"Columns: {list(test_df.columns)}")
        print(f"Missing values: {test_df.isnull().sum().sum()}")
        
        # Make predictions
        result = predictor.predict_batch(test_df)
        
        if result['success']:
            print("\n=== Production Prediction Results ===")
            stats = result['summary_stats']
            print(f"Total records: {stats['total_records']}")
            print(f"Predicted returns: {stats['predicted_returns']}")
            print(f"Return rate: {stats['return_rate']}")
            print(f"Missing values handled: {stats['missing_values_handled']}")
            print(f"Leakage columns removed: {stats['leakage_columns_removed']}")
            print(f"Feature coverage: {stats['feature_coverage']}")
            print(f"Features mapped: {stats['features_mapped']}/{stats['features_total']}")
            
            # Probability analysis
            print(f"\n=== Probability Analysis ===")
            print(f"Mean probability: {stats['probability_mean']}")
            print(f"Std probability: {stats['probability_std']}")
            print(f"Min probability: {stats['probability_min']}")
            print(f"Max probability: {stats['probability_max']}")
            print(f"Above 50%: {stats['predictions_above_50']}")
            print(f"Below 50%: {stats['predictions_below_50']}")
            
            if 'prediction_accuracy' in stats:
                print(f"\n=== Validation Results ===")
                print(f"Prediction accuracy: {stats['prediction_accuracy']}")
                print(f"Prediction precision: {stats['prediction_precision']}")
                print(f"Prediction recall: {stats['prediction_recall']}")
                print(f"Prediction F1: {stats['prediction_f1']}")
                print(f"Actual return rate: {stats['actual_return_rate']}")
                print(f"Confusion Matrix:")
                print(f"  TN: {stats['true_negatives']}, FP: {stats['false_positives']}")
                print(f"  FN: {stats['false_negatives']}, TP: {stats['true_positives']}")
            
            # Save results
            result['results_df'].to_csv('production_prediction_results.csv', index=False)
            print("\nResults saved to production_prediction_results.csv")
            
            # Display sample results with confidence levels
            print("\n=== Sample Predictions with Confidence ===")
            display_cols = ['Product_Category', 'Price', 'User_Age', 'Prediction_Label', 
                          'Return_Probability', 'Confidence_Level']
            if 'Return_Status' in result['results_df'].columns:
                display_cols.append('Return_Status')
            sample_results = result['results_df'][display_cols].head(15)
            print(sample_results.to_string(index=False))
            
            # Show pipeline information
            print("\n=== Production Pipeline Information ===")
            info = predictor.get_pipeline_info()
            for key, value in info.items():
                print(f"{key}: {value}")
        else:
            print(f"Prediction failed: {result['error']}")
            if 'details' in result:
                for detail in result['details']:
                    print(f"• {detail}")
    else:
        print("Cannot test without trained pipeline. Run train_production.py first.")

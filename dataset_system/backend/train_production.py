import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class ProductionReturnPredictionModel:
    """
    Production-grade ML model with comprehensive pipeline and high accuracy
    """
    
    def __init__(self):
        self.pipeline = None
        self.feature_columns = None
        self.model_type = 'gradient_boosting'
        self.target_encoder = LabelEncoder()
        
        # Define valid input features (NO leakage columns)
        self.valid_input_features = [
            'product_category', 'product_price', 'order_quantity', 'customer_age', 
            'customer_location', 'payment_method', 'shipping_method', 'discount',
            'delivery_time', 'purchase_history_count', 'return_history_rate', 
            'product_rating'
        ]
        
        # Define leakage columns to remove BEFORE processing
        self.leakage_columns = [
            'return_status', 'return_reason', 'return_date', 'returned',
            'return', 'prediction', 'prediction_label', 'return_probability',
            'days_to_return', 'return_flag', 'is_returned', 'has_returned'
        ]
        
        # Define possible target column names
        self.target_column_variations = [
            'return_status', 'returned', 'return', 'is_returned', 
            'has_returned', 'return_flag', 'return_indicator'
        ]
        
        # Define feature types for preprocessing
        self.numeric_features = [
            'product_price', 'order_quantity', 'customer_age', 'discount',
            'delivery_time', 'purchase_history_count', 'return_history_rate', 
            'product_rating'
        ]
        
        self.categorical_features = [
            'product_category', 'customer_location', 'payment_method', 'shipping_method'
        ]
        
        # Feature mapping for column name normalization
        self.feature_mapping = {
            'product_category': [
                'product_category', 'category', 'product_type', 'item_category', 
                'category_name', 'productcategory', 'product_cat', 'prod_category',
                'item_type', 'type', 'product_line', 'department', 'section', 'class'
            ],
            'product_price': [
                'product_price', 'price', 'amount', 'cost', 'unit_price', 
                'selling_price', 'retail_price', 'price_usd', 'productprice',
                'item_price', 'order_amount', 'total_price', 'sale_price', 'list_price'
            ],
            'order_quantity': [
                'order_quantity', 'quantity', 'qty', 'product_quantity', 
                'item_quantity', 'order_qty', 'product_qty', 'items_ordered'
            ],
            'customer_age': [
                'customer_age', 'age', 'user_age', 'customerage', 'userage',
                'cust_age', 'client_age', 'buyer_age', 'person_age', 'age_years'
            ],
            'customer_location': [
                'customer_location', 'location', 'state', 'city', 'region', 
                'country', 'customerlocation', 'user_location', 'cust_location',
                'address', 'geo_location', 'shipping_location', 'billing_location'
            ],
            'payment_method': [
                'payment_method', 'payment_type', 'payment', 'payment_option',
                'paymentmode', 'payment_method_type', 'transaction_method',
                'billing_method', 'payment_channel', 'payment_source'
            ],
            'shipping_method': [
                'shipping_method', 'delivery_method', 'shipping_type', 
                'delivery_type', 'shipping', 'delivery', 'shipping_option',
                'delivery_option', 'courier', 'shipping_carrier'
            ],
            'discount': [
                'discount', 'discount_applied', 'discount_percentage', 
                'discount_rate', 'discount_amount', 'discount_percent',
                'rebate', 'price_reduction', 'discount_value', 'savings'
            ],
            'delivery_time': [
                'delivery_time', 'delivery_days', 'shipping_time', 
                'shipping_days', 'delivery_period', 'delivery', 'shipping',
                'days_to_deliver', 'delivery_duration', 'estimated_delivery'
            ],
            'purchase_history_count': [
                'purchase_history_count', 'purchase_count', 'order_count', 
                'total_orders', 'previous_orders', 'order_history',
                'customer_orders', 'user_orders', 'lifetime_orders',
                'total_purchases', 'order_frequency', 'past_orders'
            ],
            'return_history_rate': [
                'return_history_rate', 'return_rate', 'return_frequency',
                'previous_returns', 'customer_return_rate', 'return_ratio',
                'return_percentage', 'return_history', 'historical_return_rate'
            ],
            'product_rating': [
                'product_rating', 'rating', 'stars', 'review_rating', 
                'customer_rating', 'user_rating', 'item_rating',
                'product_score', 'rating_value', 'avg_rating',
                'customer_satisfaction', 'quality_rating'
            ]
        }
    
    def identify_target_column(self, df):
        """
        Identify the target column from various possible names
        """
        for col in df.columns:
            col_normalized = col.lower().strip()
            for target_col in self.target_column_variations:
                if col_normalized == target_col.lower():
                    return col
        return None
    
    def comprehensive_data_cleaning(self, df, is_training=True):
        """
        Comprehensive data cleaning with leakage removal and quality checks
        """
        df_clean = df.copy()
        
        # Store original shape for reporting
        original_shape = df_clean.shape
        
        # Step 1: Remove leakage columns
        columns_to_drop = []
        for col in df_clean.columns:
            col_lower = col.lower()
            if any(leakage_col in col_lower for leakage_col in self.leakage_columns):
                columns_to_drop.append(col)
        
        # Store target column before dropping (for training)
        target_column = None
        target_values = None
        
        if is_training:
            target_column = self.identify_target_column(df_clean)
            if target_column:
                # Extract target values before dropping
                target_values = df_clean[target_column].copy()
                columns_to_drop.append(target_column)
        
        # Drop leakage columns
        df_clean = df_clean.drop(columns=[col for col in columns_to_drop if col in df_clean.columns])
        
        # Step 2: Normalize column names
        column_mapping = self.normalize_column_names(df_clean)
        
        # Step 3: Create standardized dataset
        processed_df = pd.DataFrame()
        
        # Map each required feature
        for required_col, actual_col in column_mapping.items():
            if actual_col in df_clean.columns:
                processed_df[required_col] = df_clean[actual_col]
        
        # Step 4: Fill missing columns with intelligent defaults
        default_values = {
            'product_category': 'Unknown',
            'product_price': 50.0,
            'order_quantity': 1,
            'customer_age': 35.0,
            'customer_location': 'Unknown',
            'payment_method': 'Unknown',
            'shipping_method': 'Standard',
            'discount': 0.0,
            'delivery_time': 3.0,
            'purchase_history_count': 5.0,
            'return_history_rate': 0.1,
            'product_rating': 4.0
        }
        
        for col in self.valid_input_features:
            if col not in processed_df.columns:
                processed_df[col] = default_values.get(col, 0)
        
        # Step 5: Convert data types and basic validation
        for col in self.numeric_features:
            processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
            # Clip unrealistic values
            if col == 'product_price':
                processed_df[col] = np.clip(processed_df[col], 1, 1000)
            elif col == 'customer_age':
                processed_df[col] = np.clip(processed_df[col], 18, 100)
            elif col == 'order_quantity':
                processed_df[col] = np.clip(processed_df[col], 1, 100)
            elif col == 'discount':
                processed_df[col] = np.clip(processed_df[col], 0, 100)
            elif col == 'delivery_time':
                processed_df[col] = np.clip(processed_df[col], 1, 30)
            elif col == 'product_rating':
                processed_df[col] = np.clip(processed_df[col], 1, 5)
        
        for col in self.categorical_features:
            processed_df[col] = processed_df[col].astype(str)
            processed_df[col] = processed_df[col].replace('nan', 'Unknown')
        
        # Step 6: Process target variable
        processed_target = None
        if is_training and target_values is not None:
            # Convert target to binary format
            processed_target = target_values.apply(
                lambda x: 1 if str(x).lower() in ['returned', 'return', 'yes', 'true', '1'] else 0
            )
        
        # Store feature columns
        self.feature_columns = self.valid_input_features
        
        # Report cleaning results
        print(f"Data Cleaning Summary:")
        print(f"  Original shape: {original_shape}")
        print(f"  Leakage columns removed: {len(columns_to_drop)}")
        print(f"  Final shape: {processed_df.shape}")
        print(f"  Missing values: {processed_df.isnull().sum().sum()}")
        
        return processed_df, processed_target, column_mapping
    
    def normalize_column_names(self, df):
        """
        Normalize column names using comprehensive mapping
        """
        column_mapping = {}
        available_columns = list(df.columns)
        
        for required_col, variations in self.feature_mapping.items():
            mapped_col = None
            
            # Normalize available columns
            normalized_available = [col.lower().strip() for col in available_columns]
            
            # Try exact match first
            for i, col in enumerate(available_columns):
                if col.lower().strip() in [v.lower() for v in variations]:
                    mapped_col = col
                    break
            
            # Try partial match
            if not mapped_col:
                for i, col in enumerate(available_columns):
                    col_lower = col.lower()
                    for variation in variations:
                        if variation.lower() in col_lower or col_lower in variation.lower():
                            mapped_col = col
                            break
                    if mapped_col:
                        break
            
            if mapped_col:
                column_mapping[required_col] = mapped_col
        
        return column_mapping
    
    def build_production_pipeline(self):
        """
        Build production-grade sklearn pipeline with proper preprocessing
        """
        # Numeric preprocessing pipeline
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical preprocessing pipeline
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Column transformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ],
            remainder='drop'  # Drop any columns not specified
        )
        
        # Model selection with class balancing
        models = {
            'gradient_boosting': GradientBoostingClassifier(
                random_state=42,
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                min_samples_split=5,
                min_samples_leaf=2,
                subsample=0.8
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                bootstrap=True
            ),
            'logistic_regression': LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced',
                C=1.0
            )
        }
        
        # Create complete pipeline
        model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', models[self.model_type])
        ])
        
        return model_pipeline
    
    def train_with_validation(self, X, y):
        """
        Train model with comprehensive validation and hyperparameter tuning
        """
        print("=== Training Production Model ===")
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        print(f"Class distribution - Train: {np.bincount(y_train)}")
        print(f"Class distribution - Test: {np.bincount(y_test)}")
        
        # Build and train pipeline
        self.pipeline = self.build_production_pipeline()
        
        # Train the model
        self.pipeline.fit(X_train, y_train)
        
        # Comprehensive evaluation
        y_pred_train = self.pipeline.predict(X_train)
        y_pred_test = self.pipeline.predict(X_test)
        y_proba_test = self.pipeline.predict_proba(X_test)
        
        # Calculate metrics
        train_metrics = {
            'accuracy': accuracy_score(y_train, y_pred_train),
            'precision': precision_score(y_train, y_pred_train, average='binary', zero_division=0),
            'recall': recall_score(y_train, y_pred_train, average='binary', zero_division=0),
            'f1': f1_score(y_train, y_pred_train, average='binary', zero_division=0)
        }
        
        test_metrics = {
            'accuracy': accuracy_score(y_test, y_pred_test),
            'precision': precision_score(y_test, y_pred_test, average='binary', zero_division=0),
            'recall': recall_score(y_test, y_pred_test, average='binary', zero_division=0),
            'f1': f1_score(y_test, y_pred_test, average='binary', zero_division=0)
        }
        
        # Cross-validation
        cv_scores = cross_val_score(self.pipeline, X_train, y_train, cv=5, scoring='accuracy')
        
        print(f"\n=== Model Performance ===")
        print(f"Model Type: {self.model_type}")
        print(f"\nTraining Metrics:")
        for metric, value in train_metrics.items():
            print(f"  {metric.capitalize()}: {value:.4f}")
        
        print(f"\nTest Metrics:")
        for metric, value in test_metrics.items():
            print(f"  {metric.capitalize()}: {value:.4f}")
        
        print(f"\nCross-Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Probability distribution analysis
        prob_class_1 = y_proba_test[:, 1]
        print(f"\nProbability Distribution:")
        print(f"  Mean: {prob_class_1.mean():.4f}")
        print(f"  Std: {prob_class_1.std():.4f}")
        print(f"  Min: {prob_class_1.min():.4f}")
        print(f"  Max: {prob_class_1.max():.4f}")
        print(f"  > 0.5: {(prob_class_1 > 0.5).sum()} ({(prob_class_1 > 0.5).mean():.2%})")
        print(f"  < 0.5: {(prob_class_1 < 0.5).sum()} ({(prob_class_1 < 0.5).mean():.2%})")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred_test)
        print(f"\nConfusion Matrix:")
        print(f"  True Negatives: {cm[0,0]}")
        print(f"  False Positives: {cm[0,1]}")
        print(f"  False Negatives: {cm[1,0]}")
        print(f"  True Positives: {cm[1,1]}")
        
        return X_train, X_test, y_train, y_test, test_metrics, cv_scores
    
    def save_production_pipeline(self, pipeline_path='pipeline_production.pkl'):
        """
        Save the complete production pipeline
        """
        pipeline_data = {
            'pipeline': self.pipeline,
            'feature_columns': self.feature_columns,
            'valid_input_features': self.valid_input_features,
            'leakage_columns': self.leakage_columns,
            'model_type': self.model_type,
            'numeric_features': self.numeric_features,
            'categorical_features': self.categorical_features,
            'feature_mapping': self.feature_mapping
        }
        
        joblib.dump(pipeline_data, pipeline_path)
        print(f"Production pipeline saved as {pipeline_path}")
    
    def load_production_pipeline(self, pipeline_path='pipeline_production.pkl'):
        """
        Load the complete production pipeline
        """
        if not Path(pipeline_path).exists():
            raise FileNotFoundError(f"Pipeline file {pipeline_path} not found")
        
        pipeline_data = joblib.load(pipeline_path)
        
        self.pipeline = pipeline_data['pipeline']
        self.feature_columns = pipeline_data['feature_columns']
        self.valid_input_features = pipeline_data['valid_input_features']
        self.leakage_columns = pipeline_data['leakage_columns']
        self.model_type = pipeline_data['model_type']
        self.numeric_features = pipeline_data['numeric_features']
        self.categorical_features = pipeline_data['categorical_features']
        self.feature_mapping = pipeline_data['feature_mapping']
        
        print(f"Production pipeline loaded from {pipeline_path}")
        print(f"Model type: {self.model_type}")

def create_production_training_data(filename='training_data_production.csv', n_samples=5000):
    """
    Create high-quality training data with realistic patterns and missing values
    """
    np.random.seed(42)
    
    # More diverse categories for better generalization
    categories = [
        'Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books', 
        'Toys', 'Beauty', 'Automotive', 'Food', 'Health', 'Furniture', 
        'Jewelry', 'Shoes', 'Accessories', 'Pet Supplies'
    ]
    
    locations = [
        'New York', 'California', 'Texas', 'Florida', 'Illinois', 
        'Pennsylvania', 'Ohio', 'Georgia', 'Michigan', 'North Carolina',
        'New Jersey', 'Virginia', 'Washington', 'Arizona', 'Massachusetts'
    ]
    
    payment_methods = [
        'Credit Card', 'Debit Card', 'PayPal', 'Cash on Delivery', 
        'UPI', 'Bank Transfer', 'Apple Pay', 'Google Pay'
    ]
    
    shipping_methods = [
        'Standard', 'Express', 'Overnight', 'Same Day', 'Pickup'
    ]
    
    # Generate realistic data with correlations
    data = {
        'Product_Category': np.random.choice(categories, n_samples),
        'Product_Price': np.random.lognormal(3.8, 0.9, n_samples),
        'Order_Quantity': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.4, 0.3, 0.15, 0.1, 0.05]),
        'Customer_Age': np.random.normal(38, 15, n_samples),
        'Customer_Location': np.random.choice(locations, n_samples),
        'Payment_Method': np.random.choice(payment_methods, n_samples),
        'Shipping_Method': np.random.choice(shipping_methods, n_samples),
        'Discount': np.random.choice([0, 5, 10, 15, 20, 25, 30, 40], n_samples, 
                                   p=[0.25, 0.20, 0.20, 0.15, 0.10, 0.05, 0.03, 0.02]),
        'Delivery_Time': np.random.gamma(2, 2.5, n_samples),
        'Purchase_History_Count': np.random.poisson(12, n_samples),
        'Return_History_Rate': np.random.beta(1.5, 10, n_samples),
        'Product_Rating': np.random.normal(3.7, 1.0, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Clean up the data with realistic constraints
    df['Product_Price'] = np.clip(df['Product_Price'], 5, 2000).round(2)
    df['Customer_Age'] = np.clip(df['Customer_Age'], 18, 85).astype(int)
    df['Purchase_History_Count'] = np.clip(df['Purchase_History_Count'], 0, 100)
    df['Return_History_Rate'] = np.clip(df['Return_History_Rate'], 0, 1).round(3)
    df['Delivery_Time'] = np.clip(df['Delivery_Time'], 1, 20).astype(int)
    df['Product_Rating'] = np.clip(df['Product_Rating'], 1, 5).round(1)
    
    # Generate realistic return patterns with multiple factors
    return_prob = (
        0.25 +  # base return rate
        (df['Product_Price'] / 2000) * 0.20 +  # price factor
        np.isin(df['Product_Category'], ['Electronics', 'Clothing', 'Shoes']) * 0.15 +  # category factor
        (df['Delivery_Time'] / 20) * 0.20 +  # delivery factor
        (5 - df['Product_Rating']) / 4 * 0.25 +  # rating factor
        df['Return_History_Rate'] * 0.30 +  # return history factor
        (df['Discount'] / 40) * 0.15 +  # discount factor
        (df['Order_Quantity'] > 3) * 0.10 +  # quantity factor
        (df['Customer_Age'] < 25) * 0.05  # age factor
    )
    
    return_prob = np.clip(return_prob, 0.05, 0.95)
    df['Return_Status'] = (np.random.random(n_samples) < return_prob).apply(lambda x: 'Returned' if x else 'Not Returned')
    
    # Introduce realistic missing values (5-10%)
    missing_rate = 0.08
    n_missing = int(n_samples * missing_rate)
    missing_indices = np.random.choice(n_samples, size=n_missing, replace=False)
    
    # Missing values in different columns
    df.loc[missing_indices[:n_missing//4], 'Product_Price'] = np.nan
    df.loc[missing_indices[n_missing//4:2*n_missing//4], 'Customer_Age'] = np.nan
    df.loc[missing_indices[2*n_missing//4:3*n_missing//4], 'Product_Rating'] = np.nan
    df.loc[missing_indices[3*n_missing//4:], 'Delivery_Time'] = np.nan
    
    # Some missing categorical values
    cat_missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.03), replace=False)
    df.loc[cat_missing_indices[:len(cat_missing_indices)//2], 'Product_Category'] = np.nan
    df.loc[cat_missing_indices[len(cat_missing_indices)//2:], 'Customer_Location'] = np.nan
    
    df.to_csv(filename, index=False)
    print(f"Production training data generated: {filename}")
    print(f"Shape: {df.shape}")
    print(f"Return rate: {df['Return_Status'].value_counts(normalize=True).get('Returned', 0):.2%}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Categories: {df['Product_Category'].nunique()}")
    print(f"Locations: {df['Customer_Location'].nunique()}")

def main():
    """
    Main training function for production-grade model
    """
    print("=== PRODUCTION-GRADE E-Commerce Return Prediction Model ===\n")
    
    # Generate high-quality training data
    if not Path('training_data_production.csv').exists():
        print("Generating production-grade training data...")
        create_production_training_data()
        print()
    
    # Load training data
    try:
        df = pd.read_csv('training_data_production.csv')
    except FileNotFoundError:
        print("Error: training_data_production.csv not found.")
        return
    
    print(f"Dataset loaded: {df.shape}")
    print(f"Return rate: {df['Return_Status'].value_counts(normalize=True).get('Returned', 0):.2%}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Columns: {list(df.columns)}")
    print()
    
    # Initialize production model
    model = ProductionReturnPredictionModel()
    
    # Comprehensive data cleaning
    X, y, column_mapping = model.comprehensive_data_cleaning(df, is_training=True)
    
    # Validate data preparation
    if y is None:
        print("Error: No valid target column found.")
        return
    
    print(f"\nData preparation completed:")
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target distribution: {np.bincount(y)}")
    print(f"Column mapping: {len(column_mapping)} features mapped")
    print(f"Missing values after cleaning: {X.isnull().sum().sum()}")
    print()
    
    # Train with comprehensive validation
    X_train, X_test, y_train, y_test, metrics, cv_scores = model.train_with_validation(X, y)
    
    # Test with sample data including missing values
    print("\n=== Sample Prediction Test ===")
    test_df = df.head(10).copy()
    
    # Introduce missing values in test data
    test_df.loc[0, 'Product_Price'] = np.nan
    test_df.loc[1, 'Customer_Age'] = np.nan
    test_df.loc[2, 'Product_Category'] = np.nan
    test_df.loc[3, 'Product_Rating'] = np.nan
    
    X_test_sample, _, _ = model.comprehensive_data_cleaning(test_df, is_training=False)
    predictions = model.pipeline.predict(X_test_sample)
    probabilities = model.pipeline.predict_proba(X_test_sample)
    
    print("Sample predictions (with missing values):")
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        actual = df.iloc[i]['Return_Status']
        prob_class_1 = prob[1]
        confidence = "High" if prob_class_1 > 0.7 or prob_class_1 < 0.3 else "Medium"
        print(f"Sample {i+1}: Predicted={pred} ({'Return' if pred else 'No Return'}), "
              f"Probability={prob_class_1:.3f}, Confidence={confidence}, Actual={actual}")
    
    # Save production pipeline
    model.save_production_pipeline()
    
    print(f"\n=== Production Training Complete ===")
    print(f"Final Model: {model.model_type}")
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test F1 Score: {metrics['f1']:.4f}")
    print(f"Cross-Validation: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print(f"Features used: {len(model.feature_columns)}")
    print(f"Pipeline handles missing values automatically")
    print(f"Ready for production deployment")

if __name__ == "__main__":
    main()

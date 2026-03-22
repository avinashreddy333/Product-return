import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score, roc_curve
from sklearn.utils.class_weight import compute_class_weight
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class FlexibleReturnPredictionModel:
    """
    Flexible ML model that works with any dataset structure
    """
    
    def __init__(self):
        self.pipeline = None
        self.feature_columns = None
        self.model_type = 'random_forest'
        self.threshold = 0.5  # Will be optimized
        self.scaler = None
        self.label_encoders = {}
        
        # Define leakage columns to remove
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
        
        # Identify target column
        feature_info['target_column'] = self.identify_target_column(df)
        
        # Identify and remove leakage columns
        for col in df.columns:
            col_lower = col.lower()
            if any(leakage_col in col_lower for leakage_col in self.leakage_columns):
                feature_info['leakage_columns'].append(col)
        
        # Clean dataframe
        df_clean = df.drop(columns=feature_info['leakage_columns'])
        if feature_info['target_column'] and feature_info['target_column'] in df_clean.columns:
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
    
    def create_flexible_preprocessor(self, numeric_features, categorical_features):
        """
        Create flexible preprocessor for any feature set
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
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='drop'  # Drop any columns not specified
        )
        
        return preprocessor
    
    def find_optimal_threshold(self, y_true, y_proba):
        """
        Find optimal threshold using ROC curve
        """
        if len(np.unique(y_true)) < 2:
            return 0.5  # Default if only one class
        
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        
        # Calculate Youden's J statistic
        youden_j = tpr - fpr
        optimal_idx = np.argmax(youden_j)
        optimal_threshold = thresholds[optimal_idx]
        
        return optimal_threshold
    
    def train_model(self, X, y):
        """
        Train model with proper hyperparameter tuning and class balancing
        """
        print("=== Training Flexible ML Model ===")
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        print(f"Class distribution - Train: {np.bincount(y_train)}")
        print(f"Class distribution - Test: {np.bincount(y_test)}")
        
        # Calculate class weights for imbalance
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        
        # Create preprocessor
        feature_info = self.detect_feature_types(X)
        preprocessor = self.create_flexible_preprocessor(
            feature_info['numeric_features'], 
            feature_info['categorical_features']
        )
        
        # Try different models with hyperparameter tuning
        models = {
            'random_forest': RandomForestClassifier(
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced'
            )
        }
        
        best_model = None
        best_score = 0
        best_name = None
        
        for name, model in models.items():
            print(f"\n--- Training {name} ---")
            
            # Create pipeline
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])
            
            # Hyperparameter grids
            param_grids = {
                'random_forest': {
                    'classifier__n_estimators': [100, 200],
                    'classifier__max_depth': [10, 15, None],
                    'classifier__min_samples_split': [2, 5],
                    'classifier__min_samples_leaf': [1, 2]
                },
                'gradient_boosting': {
                    'classifier__n_estimators': [100, 200],
                    'classifier__learning_rate': [0.01, 0.1],
                    'classifier__max_depth': [3, 6, None]
                },
                'logistic_regression': {
                    'classifier__C': [0.1, 1.0, 10.0],
                    'classifier__penalty': ['l1', 'l2']
                }
            }
            
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                pipeline, 
                param_grids[name], 
                cv=5, 
                scoring='f1',
                n_jobs=-1,
                verbose=0
            )
            
            # Fit grid search
            grid_search.fit(X_train, y_train)
            
            # Get best model
            best_pipeline = grid_search.best_estimator_
            train_score = grid_search.best_score_
            
            # Evaluate on test set
            y_pred = best_pipeline.predict(X_test)
            y_proba = best_pipeline.predict_proba(X_test)[:, 1]
            
            # Find optimal threshold
            optimal_threshold = self.find_optimal_threshold(y_test, y_proba)
            
            # Calculate metrics with optimal threshold
            y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
            
            test_accuracy = accuracy_score(y_test, y_pred_optimal)
            test_precision = precision_score(y_test, y_pred_optimal, average='binary', zero_division=0)
            test_recall = recall_score(y_test, y_pred_optimal, average='binary', zero_division=0)
            test_f1 = f1_score(y_test, y_pred_optimal, average='binary', zero_division=0)
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"CV F1 Score: {train_score:.4f}")
            print(f"Optimal Threshold: {optimal_threshold:.3f}")
            print(f"Test Metrics:")
            print(f"  Accuracy: {test_accuracy:.4f}")
            print(f"  Precision: {test_precision:.4f}")
            print(f"  Recall: {test_recall:.4f}")
            print(f"  F1: {test_f1:.4f}")
            
            # Select best model
            if test_f1 > best_score:
                best_score = test_f1
                best_model = best_pipeline
                best_name = name
                self.threshold = optimal_threshold
        
        if best_model is None:
            raise ValueError("No model could be trained successfully")
        
        self.pipeline = best_model
        self.model_type = best_name
        
        print(f"\n=== Best Model Selected: {best_name} ===")
        print(f"F1 Score: {best_score:.4f}")
        print(f"Optimal Threshold: {self.threshold:.3f}")
        
        return X_train, X_test, y_train, y_test, feature_info
    
    def save_model(self, model_path='model.pkl'):
        """
        Save the complete flexible model
        """
        model_data = {
            'pipeline': self.pipeline,
            'model_type': self.model_type,
            'threshold': self.threshold,
            'feature_info': self.detect_feature_types(pd.DataFrame())  # Empty for structure
        }
        
        joblib.dump(model_data, model_path)
        print(f"Flexible model saved as {model_path}")
    
    def load_model(self, model_path='model.pkl'):
        """
        Load the flexible model
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file {model_path} not found")
        
        model_data = joblib.load(model_path)
        
        self.pipeline = model_data['pipeline']
        self.model_type = model_data['model_type']
        self.threshold = model_data['threshold']
        
        print(f"Flexible model loaded from {model_path}")
        print(f"Model type: {self.model_type}")
        print(f"Optimal threshold: {self.threshold:.3f}")

def create_flexible_training_data(filename='training_data.csv', n_samples=3000):
    """
    Create flexible training data with various column names and realistic patterns
    """
    np.random.seed(42)
    
    # Generate diverse data with realistic patterns
    categories = [
        'Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books', 
        'Toys', 'Beauty', 'Automotive', 'Food', 'Health'
    ]
    
    locations = [
        'New York', 'California', 'Texas', 'Florida', 'Illinois', 
        'Pennsylvania', 'Ohio', 'Georgia', 'Michigan'
    ]
    
    payment_methods = [
        'Credit Card', 'Debit Card', 'PayPal', 'Cash on Delivery', 'UPI'
    ]
    
    # Generate realistic data
    data = {
        'Product_Category': np.random.choice(categories, n_samples),
        'Product_Price': np.random.lognormal(3.8, 0.9, n_samples),
        'Customer_Age': np.random.normal(38, 15, n_samples),
        'Customer_Location': np.random.choice(locations, n_samples),
        'Purchase_History_Count': np.random.poisson(12, n_samples),
        'Delivery_Time': np.random.gamma(2, 2.5, n_samples),
        'Payment_Method': np.random.choice(payment_methods, n_samples),
        'Discount_Applied': np.random.choice([0, 5, 10, 15, 20, 25, 30], n_samples, 
                                   p=[0.25, 0.20, 0.20, 0.15, 0.10, 0.05, 0.05]),
        'Product_Rating': np.random.normal(3.7, 1.0, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Clean up data
    df['Product_Price'] = np.clip(df['Product_Price'], 5, 2000).round(2)
    df['Customer_Age'] = np.clip(df['Customer_Age'], 18, 85).astype(int)
    df['Purchase_History_Count'] = np.clip(df['Purchase_History_Count'], 0, 100)
    df['Delivery_Time'] = np.clip(df['Delivery_Time'], 1, 20).astype(int)
    df['Product_Rating'] = np.clip(df['Product_Rating'], 1, 5).round(1)
    df['Discount_Applied'] = df['Discount_Applied'].astype(int)
    
    # Generate realistic return patterns
    return_prob = (
        0.35 +  # base return rate
        (df['Product_Price'] / 2000) * 0.25 +  # price factor
        np.isin(df['Product_Category'], ['Electronics', 'Clothing']) * 0.15 +  # category factor
        (df['Delivery_Time'] / 20) * 0.20 +  # delivery factor
        (5 - df['Product_Rating']) / 4 * 0.25 +  # rating factor
        (df['Discount_Applied'] / 30) * 0.15  # discount factor
    )
    
    return_prob = np.clip(return_prob, 0.05, 0.95)
    df['Return_Status'] = (np.random.random(n_samples) < return_prob).apply(lambda x: 'Returned' if x else 'Not Returned')
    
    # Add some missing values to test robustness
    missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.08), replace=False)
    df.loc[missing_indices[:len(missing_indices)//3], 'Product_Price'] = np.nan
    df.loc[missing_indices[len(missing_indices)//3:2*len(missing_indices)//3], 'Customer_Age'] = np.nan
    df.loc[missing_indices[2*len(missing_indices)//3:], 'Product_Rating'] = np.nan
    
    df.to_csv(filename, index=False)
    print(f"Flexible training data generated: {filename}")
    print(f"Shape: {df.shape}")
    print(f"Return rate: {df['Return_Status'].value_counts(normalize=True).get('Returned', 0):.2%}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Columns: {list(df.columns)}")

def main():
    """
    Main training function for flexible model
    """
    print("=== FLEXIBLE E-Commerce Return Prediction Model ===\n")
    
    # Generate flexible training data
    if not Path('training_data.csv').exists():
        print("Generating flexible training data...")
        create_flexible_training_data()
        print()
    
    # Load training data
    try:
        df = pd.read_csv('training_data.csv')
    except FileNotFoundError:
        print("Error: training_data.csv not found.")
        return
    
    print(f"Dataset loaded: {df.shape}")
    print(f"Return rate: {df['Return_Status'].value_counts(normalize=True).get('Returned', 0):.2%}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Columns: {list(df.columns)}")
    print()
    
    # Initialize flexible model
    model = FlexibleReturnPredictionModel()
    
    # Detect feature types
    feature_info = model.detect_feature_types(df)
    print(f"Detected numeric features: {feature_info['numeric_features']}")
    print(f"Detected categorical features: {feature_info['categorical_features']}")
    print(f"Target column: {feature_info['target_column']}")
    print(f"Leakage columns removed: {feature_info['leakage_columns']}")
    print()
    
    # Prepare data
    # Remove leakage columns and target
    X = df.drop(columns=feature_info['leakage_columns'])
    if feature_info['target_column'] and feature_info['target_column'] in X.columns:
        X = X.drop(columns=[feature_info['target_column']])
        y = df[feature_info['target_column']].apply(
            lambda x: 1 if str(x).lower() in ['returned', 'return', 'yes', 'true', '1'] else 0
        )
    else:
        print("Warning: No target column found, creating dummy target for testing")
        y = np.random.choice([0, 1], size=len(X))
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target distribution: {np.bincount(y)}")
    print()
    
    # Train model
    X_train, X_test, y_train, y_test, feature_info = model.train_model(X, y)
    
    # Test with sample data
    print("\n=== Sample Prediction Test ===")
    test_df = df.head(10).copy()
    
    # Introduce missing values in test data
    test_df.loc[0, 'Product_Price'] = np.nan
    test_df.loc[1, 'Customer_Age'] = np.nan
    test_df.loc[2, 'Product_Rating'] = np.nan
    
    # Remove target and leakage for prediction
    test_X = test_df.drop(columns=feature_info['leakage_columns'])
    if feature_info['target_column'] and feature_info['target_column'] in test_X.columns:
        test_X = test_X.drop(columns=[feature_info['target_column']])
    
    predictions = model.pipeline.predict(test_X)
    probabilities = model.pipeline.predict_proba(test_X)[:, 1]
    predictions_optimal = (probabilities >= model.threshold).astype(int)
    
    print("Sample predictions (with missing values):")
    for i, (pred, prob) in enumerate(zip(predictions_optimal, probabilities)):
        actual = df.iloc[i]['Return_Status']
        confidence = "High" if prob > 0.7 or prob < 0.3 else "Medium"
        print(f"Sample {i+1}: Predicted={pred} ({'Return' if pred else 'No Return'}), "
              f"Probability={prob:.3f}, Confidence={confidence}, Actual={actual}")
    
    # Save model
    model.save_model()
    
    print(f"\n=== Flexible Training Complete ===")
    print(f"Final Model: {model.model_type}")
    print(f"Optimal Threshold: {model.threshold:.3f}")
    print(f"Ready for flexible prediction")

if __name__ == "__main__":
    main()

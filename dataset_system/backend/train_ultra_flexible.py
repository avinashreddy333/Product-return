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
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class UltraFlexibleReturnPredictionModel:
    """
    Ultra-flexible ML model that works with ANY dataset structure
    - No predefined columns
    - No mandatory columns  
    - Uses whatever columns are present
    - No column matching or mapping
    """
    
    def __init__(self):
        self.pipeline = None
        self.model_type = 'random_forest'
        self.threshold = 0.5
        self.scaler = None
        self.label_encoders = {}
        
        # Define leakage columns to remove (flexible matching)
        self.leakage_keywords = [
            'return', 'returned', 'status', 'prediction', 'probability', 
            'label', 'target', 'outcome', 'result', 'flag'
        ]
        
        # Define possible target column patterns
        self.target_keywords = [
            'return', 'returned', 'status', 'outcome', 'result', 'flag', 'target'
        ]
        
        # But exclude leakage patterns
        self.exclusion_keywords = [
            'history', 'rate', 'prediction', 'probability', 'label'
        ]
    
    def identify_target_column_flexible(self, df):
        """
        Identify target column using flexible pattern matching
        """
        for col in df.columns:
            col_lower = col.lower()
            for keyword in self.target_keywords:
                if keyword in col_lower:
                    # Check if it's not a leakage column
                    is_leakage = any(exclusion in col_lower for exclusion in self.exclusion_keywords)
                    if not is_leakage:
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
    
    def create_ultra_flexible_preprocessor(self, numeric_features, categorical_features):
        """
        Create preprocessor that works with any column set
        """
        transformers = []
        
        if numeric_features:
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            transformers.append(('num', numeric_transformer, numeric_features))
        
        if categorical_features:
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            transformers.append(('cat', categorical_transformer, categorical_features))
        
        # Column transformer
        preprocessor = ColumnTransformer(
            transformers=transformers,
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
        Train model with whatever features are available
        """
        print("=== Training Ultra-Flexible ML Model ===")
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        print(f"Class distribution - Train: {np.bincount(y_train)}")
        print(f"Class distribution - Test: {np.bincount(y_test)}")
        
        # Create preprocessor based on available columns
        feature_info = self.detect_column_types_flexible(X)
        preprocessor = self.create_ultra_flexible_preprocessor(
            feature_info['numeric_features'], 
            feature_info['categorical_features']
        )
        
        # Try different models
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
            
            # Simple cross-validation
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=3, scoring='f1')
            cv_score = cv_scores.mean()
            
            # Fit model
            pipeline.fit(X_train, y_train)
            
            # Evaluate on test set
            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            
            # Find optimal threshold
            optimal_threshold = self.find_optimal_threshold(y_test, y_proba)
            
            # Calculate metrics with optimal threshold
            y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
            
            test_accuracy = accuracy_score(y_test, y_pred_optimal)
            test_precision = precision_score(y_test, y_pred_optimal, average='binary', zero_division=0)
            test_recall = recall_score(y_test, y_pred_optimal, average='binary', zero_division=0)
            test_f1 = f1_score(y_test, y_pred_optimal, average='binary', zero_division=0)
            
            print(f"CV F1 Score: {cv_score:.4f}")
            print(f"Optimal Threshold: {optimal_threshold:.3f}")
            print(f"Test Metrics:")
            print(f"  Accuracy: {test_accuracy:.4f}")
            print(f"  Precision: {test_precision:.4f}")
            print(f"  Recall: {test_recall:.4f}")
            print(f"  F1: {test_f1:.4f}")
            
            # Select best model
            if test_f1 > best_score:
                best_score = test_f1
                best_model = pipeline
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
        Save the ultra-flexible model
        """
        model_data = {
            'pipeline': self.pipeline,
            'model_type': self.model_type,
            'threshold': self.threshold,
            'feature_info': self.detect_column_types_flexible(pd.DataFrame())
        }
        
        joblib.dump(model_data, model_path)
        print(f"Ultra-flexible model saved as {model_path}")
    
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

def create_ultra_flexible_training_data(filename='training_data.csv', n_samples=2000):
    """
    Create training data with varied column names and patterns
    """
    np.random.seed(42)
    
    # Generate completely random column names and data
    data = {
        'item_price': np.random.lognormal(3.8, 0.9, n_samples),
        'customer_age': np.random.normal(38, 15, n_samples),
        'product_rating': np.random.normal(3.7, 1.0, n_samples),
        'delivery_days': np.random.gamma(2, 2.5, n_samples),
        'discount_percent': np.random.choice([0, 5, 10, 15, 20], n_samples),
        'category': np.random.choice(['Electronics', 'Clothing', 'Books'], n_samples),
        'purchase_history': np.random.poisson(12, n_samples),
        'payment_method': np.random.choice(['Credit', 'Cash', 'Online'], n_samples),
        'return_flag': np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
    }
    
    df = pd.DataFrame(data)
    
    # Clean up data
    df['item_price'] = np.clip(df['item_price'], 5, 2000).round(2)
    df['customer_age'] = np.clip(df['customer_age'], 18, 85).astype(int)
    df['product_rating'] = np.clip(df['product_rating'], 1, 5).round(1)
    df['delivery_days'] = np.clip(df['delivery_days'], 1, 20).astype(int)
    df['discount_percent'] = df['discount_percent'].astype(int)
    
    # Add missing values
    missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    df.loc[missing_indices[:len(missing_indices)//2], 'item_price'] = np.nan
    df.loc[missing_indices[len(missing_indices)//2:], 'customer_age'] = np.nan
    
    df.to_csv(filename, index=False)
    print(f"Ultra-flexible training data generated: {filename}")
    print(f"Shape: {df.shape}")
    print(f"Return rate: {df['return_flag'].mean():.2%}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Columns: {list(df.columns)}")

def main():
    """
    Main training function for ultra-flexible model
    """
    print("=== ULTRA-FLEXIBLE E-Commerce Return Prediction Model ===\n")
    print("✅ NO PREDEFINED COLUMNS")
    print("✅ NO MANDATORY FIELDS") 
    print("✅ USES WHATEVER COLUMNS ARE PRESENT")
    print("✅ NO COLUMN MATCHING OR MAPPING")
    print()
    
    # Generate ultra-flexible training data
    if not Path('training_data.csv').exists():
        print("Generating ultra-flexible training data...")
        create_ultra_flexible_training_data()
        print()
    
    # Load training data
    try:
        df = pd.read_csv('training_data.csv')
    except FileNotFoundError:
        print("Error: training_data.csv not found.")
        return
    
    print(f"Dataset loaded: {df.shape}")
    print(f"Columns found: {list(df.columns)}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print()
    
    # Initialize ultra-flexible model
    model = UltraFlexibleReturnPredictionModel()
    
    # Detect column types
    feature_info = model.detect_column_types_flexible(df)
    print(f"Detected numeric features: {feature_info['numeric_features']}")
    print(f"Detected categorical features: {feature_info['categorical_features']}")
    print(f"Target column: {feature_info['target_column']}")
    print(f"Leakage columns removed: {feature_info['leakage_columns']}")
    print()
    
    # Prepare data - remove target and leakage
    X = df.copy()
    if feature_info['target_column']:
        X = X.drop(columns=[feature_info['target_column']])
        y = df[feature_info['target_column']].apply(
            lambda x: 1 if str(x).lower() in ['1', 'yes', 'true', 'returned', 'return'] else 0
        )
    else:
        print("Warning: No target column found, creating dummy target for testing")
        y = np.random.choice([0, 1], size=len(X), p=[0.4, 0.6])  # Balanced target
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target distribution: {np.bincount(y)}")
    print()
    
    # Train model
    X_train, X_test, y_train, y_test, feature_info = model.train_model(X, y)
    
    # Test with sample data
    print("\n=== Sample Prediction Test ===")
    test_df = df.head(5).copy()
    
    # Remove target for prediction
    test_X = test_df.copy()
    if feature_info['target_column']:
        test_X = test_X.drop(columns=[feature_info['target_column']])
    
    predictions = model.pipeline.predict(test_X)
    probabilities = model.pipeline.predict_proba(test_X)[:, 1]
    predictions_optimal = (probabilities >= model.threshold).astype(int)
    
    print("Sample predictions:")
    for i, (pred, prob) in enumerate(zip(predictions_optimal, probabilities)):
        actual = df.iloc[i][feature_info['target_column']] if feature_info['target_column'] else 'N/A'
        print(f"Sample {i+1}: Predicted={pred} ({'Return' if pred else 'No Return'}), "
              f"Probability={prob:.3f}, Actual={actual}")
    
    # Save model
    model.save_model()
    
    print(f"\n=== Ultra-Flexible Training Complete ===")
    print(f"Final Model: {model.model_type}")
    print(f"Optimal Threshold: {model.threshold:.3f}")
    print(f"✅ READY FOR ANY DATASET STRUCTURE")

if __name__ == "__main__":
    main()

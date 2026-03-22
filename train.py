import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class ReturnPredictionModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        self.model = None
        self.model_type = 'random_forest'
        
    def preprocess_data(self, df, is_training=True):
        """
        Preprocess the data for training or prediction
        """
        df_processed = df.copy()
        
        # Drop ID columns
        id_columns = ['Order_ID', 'Customer_ID']
        df_processed = df_processed.drop(columns=[col for col in id_columns if col in df_processed.columns])
        
        # Handle categorical variables
        categorical_columns = ['Product_Category', 'Customer_Location', 'Payment_Method']
        
        for col in categorical_columns:
            if col in df_processed.columns:
                if is_training:
                    # Fit and transform for training data
                    self.label_encoders[col] = LabelEncoder()
                    df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col])
                else:
                    # Only transform for test/prediction data
                    if col in self.label_encoders:
                        df_processed[col] = self.label_encoders[col].transform(df_processed[col])
        
        # Separate features and target (only for training)
        if is_training:
            if 'Returned' in df_processed.columns:
                y = df_processed['Returned']
                X = df_processed.drop('Returned', axis=1)
                self.feature_columns = X.columns.tolist()
                
                # Scale numerical features
                numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns
                X[numerical_columns] = self.scaler.fit_transform(X[numerical_columns])
                
                return X, y
            else:
                raise ValueError("Target column 'Returned' not found in training data")
        else:
            # For prediction
            X = df_processed[self.feature_columns]
            
            # Scale numerical features
            numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns
            X[numerical_columns] = self.scaler.transform(X[numerical_columns])
            
            return X
    
    def train_models(self, X, y):
        """
        Train both Logistic Regression and Random Forest models
        """
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train Logistic Regression (baseline)
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_model.fit(X_train, y_train)
        
        # Evaluate Logistic Regression
        lr_pred = lr_model.predict(X_test)
        lr_metrics = {
            'accuracy': accuracy_score(y_test, lr_pred),
            'precision': precision_score(y_test, lr_pred),
            'recall': recall_score(y_test, lr_pred),
            'f1': f1_score(y_test, lr_pred)
        }
        
        # Train Random Forest (main model)
        rf_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )
        rf_model.fit(X_train, y_train)
        
        # Evaluate Random Forest
        rf_pred = rf_model.predict(X_test)
        rf_metrics = {
            'accuracy': accuracy_score(y_test, rf_pred),
            'precision': precision_score(y_test, rf_pred),
            'recall': recall_score(y_test, rf_pred),
            'f1': f1_score(y_test, rf_pred)
        }
        
        # Select the best model (Random Forest usually performs better)
        if rf_metrics['f1'] >= lr_metrics['f1']:
            self.model = rf_model
            self.model_type = 'random_forest'
            best_metrics = rf_metrics
            print("Selected Random Forest as the best model")
        else:
            self.model = lr_model
            self.model_type = 'logistic_regression'
            best_metrics = lr_metrics
            print("Selected Logistic Regression as the best model")
        
        # Print model performance
        print("\n=== Model Performance ===")
        print(f"Logistic Regression: {lr_metrics}")
        print(f"Random Forest: {rf_metrics}")
        print(f"\nBest Model ({self.model_type}): {best_metrics}")
        
        # Feature importance (only for Random Forest)
        if self.model_type == 'random_forest':
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\n=== Top 10 Feature Importance ===")
            print(feature_importance.head(10))
            
            # Save feature importance plot
            plt.figure(figsize=(10, 6))
            sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
            plt.title('Top 10 Feature Importance')
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        return X_train, X_test, y_train, y_test, best_metrics
    
    def predict(self, input_data):
        """
        Make predictions on new data
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_models first.")
        
        X = self.preprocess_data(input_data, is_training=False)
        
        # Get prediction probabilities
        probabilities = self.model.predict_proba(X)
        predictions = self.model.predict(X)
        
        return predictions, probabilities
    
    def save_model(self, filename='model.pkl'):
        """
        Save the trained model and preprocessing objects
        """
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'model_type': self.model_type
        }
        
        joblib.dump(model_data, filename)
        print(f"Model saved as {filename}")
    
    def load_model(self, filename='model.pkl'):
        """
        Load a trained model and preprocessing objects
        """
        model_data = joblib.load(filename)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_columns = model_data['feature_columns']
        self.model_type = model_data['model_type']
        
        print(f"Model loaded from {filename}")
        print(f"Model type: {self.model_type}")

def main():
    """
    Main training function
    """
    print("=== E-Commerce Product Return Prediction Model Training ===\n")
    
    # Load data
    if not Path('data.csv').exists():
        print("Error: data.csv not found. Please run generate_data.py first.")
        return
    
    df = pd.read_csv('data.csv')
    print(f"Dataset loaded: {df.shape}")
    print(f"Return rate: {df['Returned'].mean():.2%}\n")
    
    # Initialize model
    model = ReturnPredictionModel()
    
    # Preprocess and train
    X, y = model.preprocess_data(df, is_training=True)
    X_train, X_test, y_train, y_test, metrics = model.train_models(X, y)
    
    # Save model
    model.save_model('model.pkl')
    
    # Test prediction on sample data
    print("\n=== Sample Prediction Test ===")
    sample_data = df.head(5).drop('Returned', axis=1)
    predictions, probabilities = model.predict(sample_data)
    
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        actual_return = df.iloc[i]['Returned']
        print(f"Sample {i+1}: Predicted={pred}, Probability={prob[1]:.3f}, Actual={actual_return}")
    
    print(f"\n=== Training Complete ===")
    print(f"Final Model: {model.model_type}")
    print(f"Performance: {metrics}")

if __name__ == "__main__":
    main()

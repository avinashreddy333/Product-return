# 🛍️ E-Commerce Product Return Prediction System

A machine learning-powered web application that predicts whether a product is likely to be returned based on various factors including product characteristics, customer information, and transaction details.

## 🎯 Features

- **Intelligent Prediction**: Uses Random Forest classifier to predict product returns
- **Interactive Dashboard**: User-friendly Streamlit interface with real-time predictions
- **Data Visualizations**: Interactive charts and graphs for insights
- **Feature Importance**: Understand which factors influence return decisions
- **Dataset Explorer**: Preview and analyze the training dataset

## 📁 Project Structure

```
project-root/
├── app.py                 # Main Streamlit application
├── train.py              # Model training script
├── generate_data.py      # Synthetic data generator
├── utils.py              # Helper functions and utilities
├── model.pkl             # Trained model (generated after training)
├── data.csv              # Dataset (generated after data generation)
├── feature_importance.png # Feature importance visualization
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Dataset

```bash
python generate_data.py
```

This will create a synthetic dataset with 2000 samples of e-commerce transactions.

### 3. Train the Model

```bash
python train.py
```

This will:
- Preprocess the data
- Train both Logistic Regression and Random Forest models
- Select the best performing model
- Save the trained model as `model.pkl`
- Generate feature importance visualization

### 4. Run the Application

```bash
streamlit run app.py
```

The application will open in your web browser at `http://localhost:8501`

## 📊 Dataset Features

The system uses the following features for prediction:

### Product Features
- **Product_Category**: Type of product (Electronics, Clothing, etc.)
- **Product_Price**: Price in USD
- **Product_Rating**: Customer rating (1-5)
- **Discount_Applied**: Discount percentage applied

### Customer Features
- **Customer_Age**: Age of the customer
- **Customer_Location**: Geographic location
- **Purchase_History_Count**: Number of previous purchases
- **Return_History_Rate**: Historical return rate (0-1)

### Transaction Features
- **Delivery_Time**: Expected delivery time in days
- **Payment_Method**: Payment method used

### Target Variable
- **Returned**: Whether the product was returned (0 or 1)

## 🎨 Application Pages

### 1. Home
- Project overview and introduction
- Model performance metrics
- Usage instructions

### 2. Predict
- Interactive form for inputting product and customer details
- Real-time prediction with probability scores
- Visual indicators and recommendations

### 3. Model Insights
- Feature importance visualization
- Model information and statistics
- Detailed feature rankings

### 4. Dataset Preview
- Dataset summary statistics
- Interactive data overview charts
- Sample data exploration

## 🤖 Model Performance

The trained Random Forest model achieves:
- **Accuracy**: 59.5%
- **Precision**: 45.7%
- **Recall**: 13.3%
- **F1 Score**: 20.6%

### Top Predictive Features
1. Return_History_Rate (17.3%)
2. Product_Price (16.5%)
3. Product_Rating (11.9%)
4. Customer_Age (11.7%)
5. Purchase_History_Count (8.5%)

## 🛠️ Technologies Used

- **Python**: Core programming language
- **Streamlit**: Web application framework
- **Scikit-learn**: Machine learning library
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Plotly**: Interactive visualizations
- **Matplotlib/Seaborn**: Static visualizations
- **Joblib**: Model serialization

## 📝 Usage Examples

### Example 1: High-Risk Return
```
Product Category: Electronics
Price: $500
Rating: 2.5
Customer Age: 25
Return History Rate: 0.8
Delivery Time: 10 days
```
**Result**: Likely to be Returned (78% probability)

### Example 2: Low-Risk Return
```
Product Category: Books
Price: $25
Rating: 4.8
Customer Age: 45
Return History Rate: 0.05
Delivery Time: 2 days
```
**Result**: Not Likely to be Returned (12% probability)

## 🔧 Customization

### Adding New Features
1. Update the `generate_data.py` to include new features
2. Modify the preprocessing pipeline in `train.py`
3. Update the input form in `app.py`
4. Retrain the model

### Improving Model Performance
- Increase dataset size
- Feature engineering
- Hyperparameter tuning
- Try different algorithms (XGBoost, LightGBM)
- Cross-validation

## 🐛 Troubleshooting

### Common Issues

1. **Model not found**: Ensure you've run `train.py` before starting the app
2. **Dataset not found**: Run `generate_data.py` first
3. **Import errors**: Install all dependencies from `requirements.txt`
4. **Port already in use**: Streamlit will automatically suggest an alternative port

### Performance Tips
- For large datasets, consider using a subset for faster training
- Adjust model hyperparameters in `train.py`
- Use GPU acceleration if available

## 📄 License

This project is for educational purposes. Feel free to use, modify, and distribute.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📞 Support

For issues and questions:
- Check the troubleshooting section
- Review the code comments
- Open an issue on the repository

---

*Built with ❤️ using Streamlit and Scikit-learn*

# 🎯 Customer Churn Prediction System

A comprehensive Streamlit web application for predicting customer churn with advanced revenue impact simulation capabilities.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Revenue Simulation](#revenue-simulation)
- [Screenshots](#screenshots)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

## 🔍 Overview

This application provides a complete solution for customer churn prediction, including:
- Machine learning model training and evaluation
- Interactive prediction interface
- Comprehensive revenue impact simulation
- Business insights and recommendations

The system helps businesses understand which customers are likely to churn and calculates the financial impact of retention strategies.

## ✨ Features

### 🤖 Machine Learning
- **Multiple Models**: Support for Random Forest and Logistic Regression
- **Automated Preprocessing**: Feature scaling and encoding
- **Performance Metrics**: Accuracy, confusion matrix, and feature importance

### 🔮 Prediction Interface
- **Real-time Predictions**: Input customer data and get instant churn predictions
- **Probability Visualization**: Interactive gauge showing churn probability
- **Risk Assessment**: Clear classification of high-risk vs low-risk customers

### 💰 Revenue Simulation
- **Threshold Analysis**: Optimize churn probability thresholds
- **ROI Calculation**: Calculate return on investment for retention campaigns
- **Scenario Modeling**: Compare different business scenarios
- **Cost-Benefit Analysis**: Detailed financial impact breakdown

### 📊 Data Analytics
- **Customer Insights**: Age, tenure, and purchase amount distributions
- **Churn Patterns**: Visualize churn behavior across different segments
- **Interactive Charts**: Plotly-powered visualizations

## 📁 Project Structure

```
churn-prediction-app/
├── data/                   # Data folder (for storing datasets)
├── model/                  # Model folder (for saving trained models)
├── app.py                  # Main Streamlit application (legacy)
├── churn_app.py           # Enhanced Streamlit application with simulation
├── requirements.txt        # Python dependencies
└── README.md              # Project documentation
```

## 🚀 Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/abdulghif/churn-apps-simulation.git
   cd churn-apps-simulation
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run churn_app.py
   ```

4. **Open your browser**
   Navigate to `http://localhost:8501`

## 📦 Dependencies

```txt
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.3.0
plotly>=5.15.0
pickle-mixin>=1.0.0
```

## 💻 Usage

### 1. Model Training
1. Open the application in your browser
2. Navigate to the sidebar "Model Configuration"
3. Select your preferred model (Random Forest or Logistic Regression)
4. Click "🚀 Train Model" to train the model on sample data

### 2. Making Predictions
1. Go to the "🔮 Prediction" tab
2. Input customer information:
   - **Age**: Customer's age (default: 30)
   - **Gender**: Male or Female (default: Female)
   - **Purchase Amount**: Total purchase in IDR (default: 10,000,000)
   - **Tenure**: Months as customer (default: 12)
3. Click "🎯 Predict Churn" to get results

### 3. Revenue Simulation
1. Navigate to the "💰 Revenue Simulation" tab
2. Configure simulation parameters:
   - **Churn Threshold**: Probability threshold for churn classification
   - **Retention Cost**: Campaign cost as percentage of purchase amount
   - **Success Rate**: Retention campaign success rate
3. Click "🚀 Run Revenue Simulation"
4. Analyze the financial impact and ROI

## 📈 Model Performance

The application provides comprehensive model evaluation:

- **Accuracy Score**: Overall prediction accuracy
- **Feature Importance**: Which features matter most for churn prediction
- **Confusion Matrix**: Detailed breakdown of prediction categories

### Sample Performance Metrics
- **Random Forest**: ~85% accuracy on test data
- **Logistic Regression**: ~82% accuracy on test data

## 💼 Revenue Simulation Logic

### Business Scenarios

| Prediction Type | Description | Financial Impact |
|----------------|-------------|------------------|
| **True Positive** | Correctly predict churn | Revenue saved through retention |
| **False Positive** | Incorrectly predict churn | Unnecessary campaign costs |
| **True Negative** | Correctly predict stay | No action needed |
| **False Negative** | Miss actual churn | Revenue lost completely |

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Abdullah Ghifari**
- GitHub: [@abdulghif](https://github.com/abdulghif)
- Project: [Churn Apps Simulation](https://github.com/abdulghif/churn-apps-simulation)

## 📞 Support

If you have any questions or issues, please:
1. Check the [Issues](https://github.com/abdulghif/churn-apps-simulation/issues) page
2. Create a new issue if your problem isn't already listed
3. Provide detailed information about your environment and the issue

## ⭐ Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Machine learning powered by [Scikit-learn](https://scikit-learn.org/)
- Visualizations created with [Plotly](https://plotly.com/)
- Inspired by real-world customer retention challenges

---

**Made with ❤️ for better customer retention strategies**

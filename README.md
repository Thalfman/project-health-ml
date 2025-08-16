# Project Health Prediction System

## Overview
A machine learning application that predicts project health status using Naive Bayes classification. This system analyzes various project metrics to provide early warning signs of potential project risks.

## Features
- **Predictive Analytics**: Uses Naive Bayes algorithm to classify project health
- **Real-time Monitoring**: Tracks project metrics and provides instant health assessments
- **Data Generation**: Includes synthetic data generation for testing and demonstration
- **Logging System**: Comprehensive logging for predictions and monitoring
- **Web Application**: User-friendly interface built with Streamlit

## Project Structure
```
Capstone Repo/
│
├── src/
│   ├── app.py                      # Main application file
│   ├── generate_project_data.py    # Synthetic data generator
│   ├── train_naive_bayes_model.py  # Model training script
│   ├── data/
│   │   └── project_health_data.csv # Training dataset
│   ├── models/
│   │   ├── naive_bayes_model.pkl   # Trained model
│   │   └── scaler.pkl              # Feature scaler
│   └── logs/
│       ├── monitoring.json         # System monitoring logs
│       └── predictions.log         # Prediction history
│
├── models/
│   └── project_health_model.pkl    # Production model
│
├── requirements.txt                 # Python dependencies
└── .gitignore                      # Git ignore rules
```

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/Thalfman/Capstone.git
cd Capstone
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
# or
source venv/bin/activate  # On Mac/Linux
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model
```bash
python src/train_naive_bayes_model.py
```

### Generating Sample Data
```bash
python src/generate_project_data.py
```

### Running the Application
```bash
streamlit run src/app.py
```

## Model Details
The system uses a Naive Bayes classifier to predict project health based on various metrics:
- Project timeline adherence
- Resource utilization
- Budget compliance
- Risk indicators
- Team performance metrics

## Technologies Used
- **Python 3.11**
- **Machine Learning**: scikit-learn
- **Data Processing**: pandas, numpy
- **Web Framework**: Streamlit
- **Visualization**: Plotly

## Author
Thomas Halfman

## License
This project is part of a Capstone course requirement.

import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
from pathlib import Path

def train_model():
    """
    Train the Naive Bayes classifier for project health prediction.
    I chose Naive Bayes because it's simple and works well for classification.
    """
    
    # Same seed as data generation for consistency
    np.random.seed(832)
    
    # Load the data
    data_path = Path('data') / 'project_health_data.csv'
    if not data_path.exists():
        print(f"Error: Can't find data file at {data_path}")
        print("Run generate_project_data.py first!")
        return
    
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} project records")
    
    # Set up features and target
    features = ['budget_variance', 'schedule_variance', 'resource_utilization', 
                'risk_score', 'team_size', 'project_duration']
    
    X = df[features]
    y = df['health_status']
    
    # Split the data 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=742, stratify=y
    )
    
    print(f"Training on {len(X_train)} samples")
    print(f"Testing on {len(X_test)} samples")
    
    # Scale the features - important for Naive Bayes to work properly
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\nTraining the model...")
    
    # Train Gaussian Naive Bayes
    # I tried different values for var_smoothing but this seemed to work best
    model = GaussianNB(var_smoothing=1e-8)
    model.fit(X_train_scaled, y_train)
    
    # Test the model
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nModel Accuracy: {accuracy:.3f}")
    
    # Note: High accuracy suggests potential overfitting on synthetic data
    # Real accuracy would likely be lower
    
    if accuracy < 0.75:
        print("Warning: Accuracy is lower than expected")
        print("May require parameter tuning or improved data quality")
    elif accuracy > 0.95:
        print("Alert: Accuracy exceeds expected range - reviewing for overfitting")
    else:
        print("Accuracy looks reasonable!")
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Show confusion matrix in a readable format
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred, labels=['Healthy', 'At Risk', 'Critical'])
    print("              Predicted:")
    print("              Healthy | At Risk | Critical")
    print("-" * 45)
    for i, actual in enumerate(['Healthy', 'At Risk', 'Critical']):
        print(f"{actual:10}:  {cm[i][0]:7} | {cm[i][1]:7} | {cm[i][2]:8}")
    
    # Save the model and scaler
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    # Using pickle to save - not the most secure but works for this project
    with open(models_dir / 'naive_bayes_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open(models_dir / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"\nModel saved to {models_dir}/naive_bayes_model.pkl")
    print(f"Scaler saved to {models_dir}/scaler.pkl")
    
    # Quick test to make sure it saved correctly
    try:
        with open(models_dir / 'naive_bayes_model.pkl', 'rb') as f:
            test_load = pickle.load(f)
        print("✓ Model file saved successfully")
    except:
        print("✗ Problem saving model file")
    
    return accuracy

if __name__ == "__main__":
    print("Starting model training...")
    print("=" * 50)
    
    accuracy = train_model()
    
    print("\n" + "=" * 50)
    if accuracy and accuracy >= 0.85:
        print("✓ Model is ready to use!")
        print(f"  Final accuracy: {accuracy:.1%}")
    elif accuracy and accuracy >= 0.75:
        print("○ Model trained but accuracy could be better")
        print(f"  Final accuracy: {accuracy:.1%}")
    else:
        print("✗ Model accuracy too low - needs work")
    
    print("\nNext step: Run app.py to use the model")

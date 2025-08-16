import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime, timedelta
import json
import hashlib
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Project Health Prediction System",
    page_icon="📊",
    layout="wide"
)

# Get the project root path
PROJECT_ROOT = Path(__file__).resolve().parent

# Initialize session state for security
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'login_attempts' not in st.session_state:
    st.session_state.login_attempts = 0

# Security Configuration
ADMIN_PASSWORD_HASH = hashlib.sha256("wgu".encode()).hexdigest()
MAX_LOGIN_ATTEMPTS = 5
LOG_FILE = PROJECT_ROOT / "logs" / "predictions.log"
MONITORING_FILE = PROJECT_ROOT / "logs" / "monitoring.json"

def init_logging():
    """Initialize logging directories and files"""
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    
    if not LOG_FILE.exists():
        LOG_FILE.touch()
    
    if not MONITORING_FILE.exists():
        with open(MONITORING_FILE, 'w') as f:
            json.dump({
                "total_predictions": 0,
                "daily_predictions": {},
                "confidence_scores": [],
                "predictions_by_status": {"Healthy": 0, "At Risk": 0, "Critical": 0},
                "system_health_checks": []
            }, f)

def sanitize_input(value, input_type="text"):
    """Sanitize user input to prevent security issues"""
    if value is None:
        return None
    
    if input_type == "text":
        # Remove any HTML/script tags
        value = re.sub(r'<[^>]*>', '', str(value))
        # Remove any SQL-like keywords
        sql_keywords = ['DROP', 'DELETE', 'INSERT', 'UPDATE', 'SELECT', 'UNION']
        for keyword in sql_keywords:
            value = re.sub(rf'\b{keyword}\b', '', value, flags=re.IGNORECASE)
        return value.strip()
    
    elif input_type == "number":
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    return value

def log_prediction(prediction, confidence, input_data, username="anonymous"):
    """Log prediction to file for monitoring"""
    init_logging()
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "username": sanitize_input(username),
        "prediction": prediction,
        "confidence": float(confidence),
        "input_data": input_data.to_dict('records')[0]
    }
    
    # Write to log file
    with open(LOG_FILE, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')
    
    # Update monitoring stats
    update_monitoring_stats(prediction, confidence)

def update_monitoring_stats(prediction, confidence):
    """Update monitoring statistics"""
    try:
        with open(MONITORING_FILE, 'r') as f:
            stats = json.load(f)
        
        # Update total predictions
        stats["total_predictions"] += 1
        
        # Update daily predictions
        today = datetime.now().strftime("%Y-%m-%d")
        if today not in stats["daily_predictions"]:
            stats["daily_predictions"][today] = 0
        stats["daily_predictions"][today] += 1
        
        # Keep only last 30 days
        cutoff_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        stats["daily_predictions"] = {k: v for k, v in stats["daily_predictions"].items() if k >= cutoff_date}
        
        # Update confidence scores (keep last 100)
        stats["confidence_scores"].append(float(confidence))
        if len(stats["confidence_scores"]) > 100:
            stats["confidence_scores"] = stats["confidence_scores"][-100:]
        
        # Update predictions by status
        stats["predictions_by_status"][prediction] += 1
        
        with open(MONITORING_FILE, 'w') as f:
            json.dump(stats, f)
    except Exception as e:
        st.error(f"Error updating monitoring stats: {str(e)}")

def perform_system_health_check():
    """Check system health and log results"""
    health_status = {
        "timestamp": datetime.now().isoformat(),
        "checks": {}
    }
    
    # Check model files
    model_path = PROJECT_ROOT / "models" / "naive_bayes_model.pkl"
    scaler_path = PROJECT_ROOT / "models" / "scaler.pkl"
    
    health_status["checks"]["model_files"] = model_path.exists() and scaler_path.exists()
    
    # Check data file
    data_path = PROJECT_ROOT / "data" / "project_health_data.csv"
    health_status["checks"]["data_file"] = data_path.exists()
    
    # Check log directory
    log_dir = PROJECT_ROOT / "logs"
    health_status["checks"]["log_directory"] = log_dir.exists()
    
    # Overall status
    health_status["overall"] = all(health_status["checks"].values())
    
    # Log the health check
    try:
        with open(MONITORING_FILE, 'r') as f:
            stats = json.load(f)
        
        stats["system_health_checks"].append(health_status)
        # Keep only last 10 health checks
        if len(stats["system_health_checks"]) > 10:
            stats["system_health_checks"] = stats["system_health_checks"][-10:]
        
        with open(MONITORING_FILE, 'w') as f:
            json.dump(stats, f)
    except:
        pass
    
    return health_status

def check_admin_authentication(password):
    """Check if admin password is correct"""
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    return password_hash == ADMIN_PASSWORD_HASH

def check_data_quality(df):
    """Check for issues in the data"""
    issues = []
    
    # check for missing values
    missing = df.isnull().sum()
    if missing.any():
        issues.append(f"Missing values found: {missing[missing > 0].to_dict()}")
    
    # Check for outliers using IQR method
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
        if outliers > 0:
            issues.append(f"{col}: {outliers} outliers detected")
    
    return issues

def clean_data(df):
    """Clean the dataframe - handle missing values and outliers"""
    df_clean = df.copy()
    
    # Fill missing values with mean - Uses mean imputation for missing values
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_clean[col].isnull().any():
            df_clean[col].fillna(df_clean[col].mean(), inplace=True)
    
    # Remove duplicates if any
    df_clean.drop_duplicates(inplace=True)
    
    # Cap outliers instead of removing them
    for col in numeric_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
    
    return df_clean

def validate_input(value, min_val, max_val, field_name):
    """Make sure the input is within valid range with security checks"""
    try:
        # Sanitize the input first
        sanitized_value = sanitize_input(value, "number")
        if sanitized_value is None:
            return False, f"Invalid {field_name} value - must be a number"
        
        if min_val <= sanitized_value <= max_val:
            return True, sanitized_value
        else:
            return False, f"{field_name} must be between {min_val} and {max_val}"
    except:
        return False, f"Invalid {field_name} value"

def load_model():
    """Load the trained model and scaler from pickle files"""
    model_path = PROJECT_ROOT / "models" / "naive_bayes_model.pkl"
    scaler_path = PROJECT_ROOT / "models" / "scaler.pkl"
    
    if not model_path.exists() or not scaler_path.exists():
        st.error(f"Model files not found. Please run train_naive_bayes_model.py first.")
        return None, None
    
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        with open(scaler_path, 'rb') as file:
            scaler = pickle.load(file)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def load_data():
    """Load the project data CSV file"""
    data_path = PROJECT_ROOT / "data" / "project_health_data.csv"
    
    try:
        df = pd.read_csv(data_path)
        # Clean it up
        df = clean_data(df)
        return df
    except FileNotFoundError:
        st.error("Data file not found. Please run generate_project_data.py first.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def evaluate_model_performance(model, scaler, df):
    """Check how well the model performs on the data"""
    try:
        feature_cols = ['budget_variance', 'schedule_variance', 'resource_utilization',
                       'risk_score', 'team_size', 'project_duration']
        
        X = df[feature_cols]
        y = df['health_status']
        
        # Scale the features
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y, predictions)
        precision = precision_score(y, predictions, average='weighted')
        recall = recall_score(y, predictions, average='weighted')
        f1 = f1_score(y, predictions, average='weighted')
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    except Exception as e:
        return None

def generate_prescriptive_recommendations(prediction, input_data):
    """
    PRESCRIPTIVE METHOD: Generate specific actionable recommendations with quantified changes
    This is the key prescriptive analytics component that tells users exactly what to do
    """
    prescriptions = []
    
    # Extract current values
    budget_var = input_data['budget_variance'][0]
    schedule_var = input_data['schedule_variance'][0]
    resource_util = input_data['resource_utilization'][0]
    risk_score = input_data['risk_score'][0]
    team_size = input_data['team_size'][0]
    duration = input_data['project_duration'][0]
    
    prescriptions.append("## 🎯 Prescriptive Actions to Improve Project Health\n")
    
    if prediction == 'Critical':
        prescriptions.append("### ⚠️ IMMEDIATE INTERVENTIONS REQUIRED:\n")
        
        # Budget optimization
        if budget_var > 20:
            reduction_needed = budget_var - 10
            prescriptions.append(f"**💰 Budget Action:**")
            prescriptions.append(f"   • Reduce budget overrun by {reduction_needed:.1f}%")
            prescriptions.append(f"   • Target: Bring budget variance below 10%")
            prescriptions.append(f"   • Method: Cut non-essential features or negotiate additional funding of ${reduction_needed * 1000:.0f}")
        
        # Schedule optimization
        if schedule_var > 20:
            days_to_recover = (schedule_var - 10) * duration / 100
            prescriptions.append(f"\n**📅 Schedule Action:**")
            prescriptions.append(f"   • Compress schedule by {days_to_recover:.0f} days")
            prescriptions.append(f"   • Target: Reduce delay to under 10%")
            prescriptions.append(f"   • Method: Add {int(days_to_recover/5)} resources or reduce scope by 15%")
        
        # Resource optimization
        if resource_util > 100:
            reduction = resource_util - 85
            additional_team = int(team_size * (reduction / 100))
            prescriptions.append(f"\n**👥 Resource Action:**")
            prescriptions.append(f"   • Reduce utilization by {reduction:.0f}%")
            prescriptions.append(f"   • Add {additional_team} team members immediately")
            prescriptions.append(f"   • Or redistribute {reduction:.0f}% of workload")
        
        # Risk mitigation
        if risk_score > 7:
            risk_reduction = risk_score - 5
            prescriptions.append(f"\n**⚡ Risk Action:**")
            prescriptions.append(f"   • Reduce risk score by {risk_reduction:.1f} points")
            prescriptions.append(f"   • Implement daily risk reviews")
            prescriptions.append(f"   • Create contingency plans for top 3 risks")
    
    elif prediction == 'At Risk':
        prescriptions.append("### ⚠️ PREVENTIVE ACTIONS RECOMMENDED:\n")
        
        if budget_var > 10:
            prescriptions.append(f"**💰 Budget Action:**")
            prescriptions.append(f"   • Monitor weekly - variance at {budget_var:.1f}%")
            prescriptions.append(f"   • Set threshold alerts at 15%")
            prescriptions.append(f"   • Freeze non-critical purchases")
        
        if schedule_var > 10:
            buffer_days = int(duration * 0.1)
            prescriptions.append(f"\n**📅 Schedule Action:**")
            prescriptions.append(f"   • Add {buffer_days} day buffer to critical path")
            prescriptions.append(f"   • Increase check-ins to twice weekly")
        
        if resource_util > 85:
            prescriptions.append(f"\n**👥 Resource Action:**")
            prescriptions.append(f"   • Consider adding 1-2 team members")
            prescriptions.append(f"   • Implement workload balancing")
    
    else:  # Healthy
        prescriptions.append("### ✅ OPTIMIZATION OPPORTUNITIES:\n")
        
        # Even healthy projects can be optimized
        if budget_var < -10:
            prescriptions.append(f"**💰 Opportunity:** Reinvest {abs(budget_var):.1f}% budget surplus into quality improvements")
        
        if resource_util < 70:
            prescriptions.append(f"**👥 Opportunity:** Team has {70-resource_util:.0f}% spare capacity - consider taking on additional features")
        
        prescriptions.append(f"\n**📊 Best Practices to Maintain:**")
        prescriptions.append(f"   • Continue current monitoring frequency")
        prescriptions.append(f"   • Document successful practices")
        prescriptions.append(f"   • Share learnings with other projects")
    
    # Add ROI calculation
    prescriptions.append("\n### 💵 Expected ROI of Actions:")
    if prediction == 'Critical':
        prescriptions.append(f"   • Implementing these changes could save ${abs(budget_var) * 5000:.0f}")
        prescriptions.append(f"   • Time to positive ROI: 2-3 weeks")
    elif prediction == 'At Risk':
        prescriptions.append(f"   • Preventive actions could avoid ${abs(budget_var) * 2000:.0f} in overruns")
        prescriptions.append(f"   • Time to positive ROI: 4-6 weeks")
    
    return prescriptions

def generate_recommendations(prediction, probabilities, input_data):
    """Generate recommendations based on the prediction (original descriptive method)"""
    recommendations = []
    
    if prediction == 'Critical':
        recommendations.append("**Immediate Actions Required:**")
        
        # Check each metric and suggest fixes
        if input_data['budget_variance'][0] > 20:
            recommendations.append("• Budget is significantly over - need immediate review")
            recommendations.append("• Consider reducing scope or getting more funding")
        
        if input_data['schedule_variance'][0] > 20:
            recommendations.append("• Schedule is significantly delayed - critical path review needed")
            recommendations.append("• May need to add resources or reduce scope")
        
        if input_data['resource_utilization'][0] > 100:
            recommendations.append("• Team capacity exceeded - burnout risk high")
        
        if input_data['risk_score'][0] > 7:
            recommendations.append("• Critical risk level - escalate to senior management")
    
    elif prediction == 'At Risk':
        recommendations.append("**Preventive Actions Recommended:**")
        
        if input_data['budget_variance'][0] > 10:
            recommendations.append("• Keep close eye on budget")
        
        if input_data['schedule_variance'][0] > 10:
            recommendations.append("• Review schedule and identify bottlenecks")
        
        if input_data['resource_utilization'][0] > 85:
            recommendations.append("• Monitor team workload")
    
    else:  # Healthy
        recommendations.append("**Project is on track:**")
        recommendations.append("• Maintain current practices")
        recommendations.append("• Document what's working well")
    
    # Add note if confidence is low
    max_prob = probabilities.max()
    if max_prob < 0.6:
        recommendations.append("\n⚠️ Note: Prediction confidence below threshold. Manual review recommended.")
    
    return recommendations

def main():
    # Initialize logging
    init_logging()
    
    # Perform system health check on startup
    perform_system_health_check()
    
    st.title("Project Health Prediction System")
    st.markdown("---")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        ["Home", "Make Prediction", "View Analytics", "Model Performance", "System Monitoring"]
    )
    
    # Check if admin features require authentication
    if page == "System Monitoring" and not st.session_state.authenticated:
        show_admin_login()
    elif page == "Home":
        show_home()
    elif page == "Make Prediction":
        show_prediction()
    elif page == "View Analytics":
        show_analytics()
    elif page == "Model Performance":
        show_model_performance()
    elif page == "System Monitoring":
        show_monitoring()

def show_admin_login():
    """Show admin login page for secure features"""
    st.header("🔒 Admin Authentication Required")
    
    st.info("For Testing/Evaluation: Use password 'wgu' to access monitoring features")
    
    if st.session_state.login_attempts >= MAX_LOGIN_ATTEMPTS:
        st.error("Maximum login attempts exceeded. Please restart the application.")
        return
    
    with st.form("admin_login"):
        password = st.text_input("Admin Password", type="password")
        submitted = st.form_submit_button("Login")
        
        if submitted:
            if check_admin_authentication(password):
                st.session_state.authenticated = True
                st.session_state.login_attempts = 0
                st.success("Authentication successful!")
                st.rerun()
            else:
                st.session_state.login_attempts += 1
                remaining = MAX_LOGIN_ATTEMPTS - st.session_state.login_attempts
                st.error(f"Invalid password. {remaining} attempts remaining.")

def show_home():
    st.header("Welcome to Project Health Prediction System")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("About This System")
        st.write("""
        This system uses machine learning to predict project health status based on:
        - Budget Variance
        - Schedule Variance  
        - Resource Utilization
        - Risk Score
        - Team Size
        - Project Duration
        
        **Key Features:**
        - **Predictive Analytics**: ML-based health predictions
        - **Prescriptive Analytics**: Specific action recommendations
        - **Security**: Input validation and admin authentication
        - **Monitoring**: Track system usage and performance
        """)
        
        # Check if model is loaded
        model, scaler = load_model()
        if model:
            st.success("✅ Model Status: Ready")
            
            df = load_data()
            if df is not None:
                metrics = evaluate_model_performance(model, scaler, df)
                if metrics:
                    st.info(f"Current Model Accuracy: {metrics['accuracy']:.1%}")
        else:
            st.error("❌ Model Status: Not Available")
    
    with col2:
        st.subheader("Health Categories")
        
        st.success("**Healthy** - Project is on track")
        st.warning("**At Risk** - Needs attention")  
        st.error("**Critical** - Immediate action required")
        
        st.subheader("Dataset Statistics")
        df = load_data()
        if df is not None:
            st.metric("Total Projects", len(df))
            health_dist = df['health_status'].value_counts()
            for status in ['Healthy', 'At Risk', 'Critical']:
                if status in health_dist.index:
                    percentage = health_dist[status]/len(df)*100
                    st.write(f"{status}: {health_dist[status]} ({percentage:.1f}%)")
        
        # System Health Status
        st.subheader("System Health")
        health = perform_system_health_check()
        if health["overall"]:
            st.success("✅ All systems operational")
        else:
            st.error("⚠️ System issues detected")

def show_prediction():
    st.header("Single Project Health Assessment")
    
    model, scaler = load_model()
    if model is None:
        return
    
    st.write("Enter project metrics to get a health prediction with prescriptive recommendations:")
    
    # Input form with security validation
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            budget_variance = st.number_input(
                "Budget Variance (%)",
                min_value=-50.0,
                max_value=100.0,
                value=0.0,
                help="Negative = under budget, Positive = over budget"
            )
            
            schedule_variance = st.number_input(
                "Schedule Variance (%)",
                min_value=-50.0,
                max_value=100.0,
                value=0.0,
                help="Negative = ahead, Positive = behind"
            )
        
        with col2:
            resource_utilization = st.slider(
                "Resource Utilization (%)",
                min_value=0,
                max_value=150,
                value=80
            )
            
            risk_score = st.slider(
                "Risk Score",
                min_value=0.0,
                max_value=10.0,
                value=5.0,
                step=0.1
            )
        
        with col3:
            team_size = st.number_input(
                "Team Size",
                min_value=1,
                max_value=100,
                value=10
            )
            
            project_duration = st.number_input(
                "Project Duration (days)",
                min_value=1,
                max_value=1000,
                value=90
            )
        
        submitted = st.form_submit_button("Predict Health Status", type="primary")
    
    if submitted:
        # Validate and sanitize inputs
        valid_budget, budget_msg = validate_input(budget_variance, -50, 100, "Budget Variance")
        valid_schedule, schedule_msg = validate_input(schedule_variance, -50, 100, "Schedule Variance")
        
        if not valid_budget:
            st.error(budget_msg)
            return
        if not valid_schedule:
            st.error(schedule_msg)
            return
        
        # Additional security checks
        if team_size > 100 or team_size < 1:
            st.error("Invalid team size. Please enter a value between 1 and 100.")
            return
        
        if project_duration > 1000 or project_duration < 1:
            st.error("Invalid project duration. Please enter a value between 1 and 1000 days.")
            return
        
        # Prepare data for prediction
        input_data = pd.DataFrame({
            'budget_variance': [budget_variance],
            'schedule_variance': [schedule_variance],
            'resource_utilization': [resource_utilization],
            'risk_score': [risk_score],
            'team_size': [team_size],
            'project_duration': [project_duration]
        })
        
        # Make prediction
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]
        
        # Get the class labels in the same order as the model
        class_labels = model.classes_
        
        # Log the prediction
        max_confidence = probabilities.max()
        log_prediction(prediction, max_confidence, input_data)
        
        st.markdown("---")
        st.subheader("Assessment Results")
        
        # Create tabs for different types of recommendations
        tab1, tab2, tab3 = st.tabs(["Prediction", "Prescriptive Actions", "General Recommendations"])
        
        with tab1:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Show the prediction with appropriate color
                if prediction == 'Healthy':
                    st.success(f"### Status: {prediction}")
                elif prediction == 'At Risk':
                    st.warning(f"### Status: {prediction}")
                else:
                    st.error(f"### Status: {prediction}")
                
                # Show confidence levels
                st.subheader("Confidence Levels")
                
                # Create a dictionary to map class names to their probabilities
                prob_dict = {}
                for i, label in enumerate(class_labels):
                    prob_dict[label] = probabilities[i]
                
                # Display in a consistent order
                for status in ['Healthy', 'At Risk', 'Critical']:
                    if status in prob_dict:
                        st.progress(prob_dict[status])
                        st.write(f"{status}: {prob_dict[status]:.1%}")
            
            with col2:
                # Visualization of the prediction
                fig = px.bar(
                    x=list(prob_dict.keys()),
                    y=list(prob_dict.values()),
                    labels={'x': 'Status', 'y': 'Probability'},
                    title='Prediction Probabilities',
                    color=list(prob_dict.values()),
                    color_continuous_scale='RdYlGn_r'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Prescriptive Analytics")
            prescriptions = generate_prescriptive_recommendations(prediction, input_data)
            for prescription in prescriptions:
                st.markdown(prescription)
        
        with tab3:
            st.subheader("General Recommendations")
            recommendations = generate_recommendations(prediction, probabilities, input_data)
            for rec in recommendations:
                st.write(rec)

def show_analytics():
    st.header("Analytics Dashboard")
    
    df = load_data()
    if df is None:
        return
    
    # Data quality section
    with st.expander("Data Quality Report"):
        issues = check_data_quality(df)
        if issues:
            for issue in issues:
                st.write(f"• {issue}")
        else:
            st.success("No data quality issues found")
    
    # Metrics row
    st.subheader("Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Projects", len(df))
    
    with col2:
        healthy_pct = (df['health_status'] == 'Healthy').mean() * 100
        st.metric("Healthy %", f"{healthy_pct:.1f}%")
    
    with col3:
        critical_count = (df['health_status'] == 'Critical').sum()
        st.metric("Critical Projects", critical_count)
    
    with col4:
        avg_risk = df['risk_score'].mean()
        st.metric("Avg Risk Score", f"{avg_risk:.2f}")
    
    # Charts
    st.subheader("Visual Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Project Health Distribution**")
        health_counts = df['health_status'].value_counts()
        fig1 = px.pie(
            values=health_counts.values,
            names=health_counts.index,
            color_discrete_map={
                'Healthy': '#2E7D32',
                'At Risk': '#F57C00', 
                'Critical': '#C62828'
            }
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.write("**Average Risk Factors by Status**")
        risk_data = df.groupby('health_status')[['budget_variance', 'schedule_variance', 'risk_score']].mean()
        fig2 = px.bar(
            risk_data.T,
            color_discrete_map={
                'Healthy': '#2E7D32',
                'At Risk': '#F57C00',
                'Critical': '#C62828'
            }
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Correlation matrix
    st.write("**Feature Correlation Matrix**")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()
    
    fig3 = px.imshow(
        corr,
        color_continuous_scale='RdBu',
        aspect="auto"
    )
    st.plotly_chart(fig3, use_container_width=True)
    
    # Stats table
    st.subheader("Descriptive Statistics")
    grouped_stats = df.groupby('health_status')[['budget_variance', 'schedule_variance', 
                                                  'resource_utilization', 'risk_score']].agg(['mean', 'std'])
    st.dataframe(grouped_stats, use_container_width=True)

def show_model_performance():
    st.header("Model Performance Evaluation")
    
    model, scaler = load_model()
    if model is None:
        return
    
    df = load_data()
    if df is None:
        return
    
    st.write("Evaluating model performance on the dataset...")
    
    # Get performance metrics
    metrics = evaluate_model_performance(model, scaler, df)
    
    if metrics:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        
        with col2:
            st.metric("Precision", f"{metrics['precision']:.2%}")
        
        with col3:
            st.metric("Recall", f"{metrics['recall']:.2%}")
        
        with col4:
            st.metric("F1 Score", f"{metrics['f1_score']:.2%}")
        
        # Confusion Matrix
        st.subheader("Confusion Matrix")
        
        feature_cols = ['budget_variance', 'schedule_variance', 'resource_utilization',
                       'risk_score', 'team_size', 'project_duration']
        
        X = df[feature_cols]
        y = df['health_status']
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)
        
        # Create confusion matrix
        cm = confusion_matrix(y, predictions, labels=['Healthy', 'At Risk', 'Critical'])
        
        fig = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['Healthy', 'At Risk', 'Critical'],
            y=['Healthy', 'At Risk', 'Critical'],
            color_continuous_scale='Blues',
            text_auto=True
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Classification report
        st.subheader("Performance by Category")
        
        from sklearn.metrics import classification_report
        report = classification_report(y, predictions, output_dict=True)
        
        report_df = pd.DataFrame(report).transpose()
        report_df = report_df[report_df.index.isin(['Healthy', 'At Risk', 'Critical'])]
        
        st.dataframe(report_df[['precision', 'recall', 'f1-score', 'support']], use_container_width=True)
        
        # Add a note about limitations
        st.warning("⚠️ Note: The 94% accuracy reflects synthetic data patterns. Production accuracy typically ranges 75-85%.")

def show_monitoring():
    """Show system monitoring dashboard"""
    st.header("📊 System Monitoring Dashboard")
    
    # Load monitoring stats
    try:
        with open(MONITORING_FILE, 'r') as f:
            stats = json.load(f)
    except:
        st.error("No monitoring data available yet.")
        return
    
    # System Health Check
    st.subheader("System Health Status")
    health = perform_system_health_check()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if health["checks"].get("model_files", False):
            st.success("✅ Model Files: OK")
        else:
            st.error("❌ Model Files: Missing")
    
    with col2:
        if health["checks"].get("data_file", False):
            st.success("✅ Data File: OK")
        else:
            st.error("❌ Data File: Missing")
    
    with col3:
        if health["checks"].get("log_directory", False):
            st.success("✅ Logs: OK")
        else:
            st.error("❌ Logs: Missing")
    
    # Usage Statistics
    st.subheader("Usage Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Predictions", stats.get("total_predictions", 0))
    
    with col2:
        today_count = stats.get("daily_predictions", {}).get(datetime.now().strftime("%Y-%m-%d"), 0)
        st.metric("Today's Predictions", today_count)
    
    with col3:
        if stats.get("confidence_scores"):
            avg_confidence = np.mean(stats["confidence_scores"])
            st.metric("Avg Confidence", f"{avg_confidence:.1%}")
        else:
            st.metric("Avg Confidence", "N/A")
    
    with col4:
        # Calculate trend
        if len(stats.get("daily_predictions", {})) > 1:
            dates = sorted(stats["daily_predictions"].keys())
            if len(dates) >= 2:
                yesterday = stats["daily_predictions"].get(dates[-2], 0)
                today = stats["daily_predictions"].get(dates[-1], 0)
                trend = today - yesterday
                st.metric("Daily Trend", today, delta=trend)
            else:
                st.metric("Daily Trend", "N/A")
        else:
            st.metric("Daily Trend", "N/A")
    
    # Daily Predictions Chart
    if stats.get("daily_predictions"):
        st.subheader("Daily Predictions (Last 30 Days)")
        
        dates = list(stats["daily_predictions"].keys())
        counts = list(stats["daily_predictions"].values())
        
        fig = px.line(
            x=dates,
            y=counts,
            labels={'x': 'Date', 'y': 'Number of Predictions'},
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Predictions by Status
    if stats.get("predictions_by_status"):
        st.subheader("Predictions by Health Status")
        
        col1, col2 = st.columns(2)
        
        with col1:
            status_data = stats["predictions_by_status"]
            fig = px.pie(
                values=list(status_data.values()),
                names=list(status_data.keys()),
                color_discrete_map={
                    'Healthy': '#2E7D32',
                    'At Risk': '#F57C00',
                    'Critical': '#C62828'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Show as metrics
            for status, count in status_data.items():
                if stats["total_predictions"] > 0:
                    pct = (count / stats["total_predictions"]) * 100
                    st.metric(f"{status} Predictions", f"{count} ({pct:.1f}%)")
    
    # Confidence Score Distribution
    if stats.get("confidence_scores"):
        st.subheader("Confidence Score Distribution")
        
        fig = px.histogram(
            stats["confidence_scores"],
            nbins=20,
            labels={'value': 'Confidence Score', 'count': 'Frequency'},
            title="Distribution of Prediction Confidence Scores"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent Predictions Log
    st.subheader("Recent Predictions Log")
    
    if LOG_FILE.exists():
        with open(LOG_FILE, 'r') as f:
            lines = f.readlines()
            recent_logs = []
            for line in lines[-10:]:  # Last 10 predictions
                try:
                    log = json.loads(line)
                    recent_logs.append({
                        'Timestamp': log['timestamp'],
                        'Prediction': log['prediction'],
                        'Confidence': f"{log['confidence']:.1%}",
                        'Budget Var': f"{log['input_data']['budget_variance']:.1f}%",
                        'Schedule Var': f"{log['input_data']['schedule_variance']:.1f}%"
                    })
                except:
                    continue
            
            if recent_logs:
                df_logs = pd.DataFrame(recent_logs)
                st.dataframe(df_logs, use_container_width=True)
            else:
                st.info("No recent predictions to display")
    
    # Admin Actions
    st.subheader("Admin Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Clear Logs"):
            try:
                with open(LOG_FILE, 'w') as f:
                    f.write("")
                st.success("Logs cleared successfully")
            except Exception as e:
                st.error(f"Error clearing logs: {str(e)}")
    
    with col2:
        if st.button("Reset Statistics"):
            try:
                with open(MONITORING_FILE, 'w') as f:
                    json.dump({
                        "total_predictions": 0,
                        "daily_predictions": {},
                        "confidence_scores": [],
                        "predictions_by_status": {"Healthy": 0, "At Risk": 0, "Critical": 0},
                        "system_health_checks": []
                    }, f)
                st.success("Statistics reset successfully")
                st.rerun()
            except Exception as e:
                st.error(f"Error resetting statistics: {str(e)}")

if __name__ == "__main__":
    main()
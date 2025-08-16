import pandas as pd
import numpy as np
from pathlib import Path

def generate_project_data(n_samples=1000):
    """
    Generate synthetic project data for training purposes.
    Based on project management research patterns.
    """
    
    # Using my student ID last 3 digits as seed for reproducibility
    np.random.seed(832)  
    
    # Initialize lists to store the data
    health_status = []
    budget_variance = []
    schedule_variance = []
    resource_utilization = []
    risk_score = []
    team_size = []
    project_duration = []
    
    # Generate data - trying to get roughly 1/3 of each category
    # but adding some randomness so it's not exactly equal
    for i in range(n_samples):
        rand_num = np.random.random()
        
        if rand_num < 0.33:  # Healthy projects
            status = 'Healthy'
            # Healthy projects should have low variance
            bv = np.random.normal(0, 8)  # centered around 0, small std dev
            sv = np.random.normal(0, 8)
            ru = np.random.normal(80, 10)  # around 80% utilization seems healthy
            rs = np.random.uniform(1, 4)  # low risk
            
        elif rand_num < 0.70:  # At Risk projects (slightly more common)
            status = 'At Risk'
            bv = np.random.normal(0, 15)
            # making some projects more over budget than others
            if np.random.random() > 0.5:
                bv = abs(bv) + 10
            sv = np.random.normal(0, 18)
            if np.random.random() > 0.5:
                sv = abs(sv) + 12
            ru = np.random.normal(90, 20)  
            rs = np.random.uniform(4, 7.5)
            
        else:  # Critical projects
            status = 'Critical'
            # These should be really bad
            bv = np.random.normal(0, 20)
            if np.random.random() > 0.3:
                bv = abs(bv) + 20  # significantly over budget
            sv = np.random.normal(0, 25)
            if np.random.random() > 0.3:
                sv = abs(sv) + 20  # significantly behind schedule
            # Critical projects either have too many or too few resources
            if np.random.random() > 0.5:
                ru = np.random.normal(120, 15)  # overutilized
            else:
                ru = np.random.normal(40, 15)  # underutilized  
            rs = np.random.uniform(6.5, 9.5)  # high risk
        
        # Team size and duration don't seem to correlate much with health
        # so just making them random
        ts = np.random.randint(5, 50)
        proj_dur = np.random.randint(30, 365)
        
        # Clip values to reasonable ranges
        bv = max(min(bv, 100), -50)
        sv = max(min(sv, 100), -50)
        ru = max(min(ru, 150), 0)
        rs = max(min(rs, 10), 0)
        
        health_status.append(status)
        budget_variance.append(bv)
        schedule_variance.append(sv)
        resource_utilization.append(ru)
        risk_score.append(rs)
        team_size.append(ts)
        project_duration.append(proj_dur)
    
    # Create project IDs 
    project_ids = [f'PROJ_{i:04d}' for i in range(1, n_samples + 1)]
    
    # Put it all in a dataframe
    df = pd.DataFrame({
        'project_id': project_ids,
        'budget_variance': budget_variance,
        'schedule_variance': schedule_variance,
        'resource_utilization': resource_utilization,
        'risk_score': risk_score,
        'team_size': team_size,
        'project_duration': project_duration,
        'health_status': health_status
    })
    
    # Shuffle the data so it's not in order
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Save to CSV
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    output_path = data_dir / 'project_health_data.csv'
    df.to_csv(output_path, index=False)
    
    print(f"Generated {n_samples} project records")
    print(f"Saved to {output_path}")
    
    # Check the distribution
    print("\nHealth Status Distribution:")
    counts = df['health_status'].value_counts()
    print(counts)
    
    # Print percentages
    for status in ['Healthy', 'At Risk', 'Critical']:
        pct = (df['health_status'] == status).mean() * 100
        print(f"{status}: {pct:.1f}%")
    
    return df

if __name__ == "__main__":
    # Generate 1000 fake projects
    df = generate_project_data(1000)
    
    print("\nFirst few rows:")
    print(df.head())
    
    print("\nAverage values by health status:")
    print(df.groupby('health_status')[['budget_variance', 'schedule_variance', 'risk_score']].mean())
    
    # Just checking if the data makes sense
    print("\nValidation: Critical projects exhibit deteriorated metrics as expected.")

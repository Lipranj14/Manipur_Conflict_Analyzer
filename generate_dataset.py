import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

def generate_data(num_records=4000):
    np.random.seed(42)
    random.seed(42)

    districts = ['Bishnupur', 'Chandel', 'Churachandpur', 'Imphal East', 'Imphal West', 
                 'Jiribam', 'Kakching', 'Kamjong', 'Kangpokpi', 'Noney', 'Pherzawl', 
                 'Senapati', 'Tamenglong', 'Tengnoupal', 'Thoubal', 'Ukhrul']

    event_types = ['Battles', 'Explosions/Remote violence', 'Protests', 'Riots', 
                   'Strategic developments', 'Violence against civilians']

    actors = ['State Forces', 'Rebel Group A', 'Rebel Group B', 'Protesters', 'Civilians', 'Unidentified Armed Group']

    start_date = datetime(2010, 1, 1)
    end_date = datetime.now()
    days_between = (end_date - start_date).days

    data = []
    
    # Introduce some temporal and spatial bias for realism
    for _ in range(num_records):
        # Time bias (more events recently)
        days_offset = int(np.random.beta(a=2, b=1) * days_between) 
        event_date = start_date + timedelta(days=days_offset)
        
        # Spatial bias (some districts more active)
        district_probs = [0.05]*16
        district_probs[2] = 0.15 # Churachandpur
        district_probs[3] = 0.15 # Imphal East
        district_probs[4] = 0.15 # Imphal West
        district_probs = [p / sum(district_probs) for p in district_probs]
        district = np.random.choice(districts, p=district_probs)
        
        event_type = np.random.choice(event_types, p=[0.2, 0.1, 0.3, 0.2, 0.1, 0.1])
        
        # Fatalities logic based on event type
        if event_type in ['Battles', 'Explosions/Remote violence']:
            fatalities = np.random.poisson(2)
        elif event_type in ['Violence against civilians', 'Riots']:
            fatalities = np.random.poisson(0.5)
        else:
            fatalities = 0
            
        actor1 = np.random.choice(actors)
        actor2 = np.random.choice(actors) if random.random() > 0.3 else 'None'
        
        # Approximate lat/lon for Manipur
        base_lat, base_lon = 24.8170, 93.9368
        lat = base_lat + np.random.normal(0, 0.3)
        lon = base_lon + np.random.normal(0, 0.3)
        
        data.append({
            'event_id_cnty': f'IND{np.random.randint(10000, 99999)}',
            'event_date': event_date.strftime('%Y-%m-%d'),
            'year': event_date.year,
            'event_type': event_type,
            'actor1': actor1,
            'actor2': actor2,
            'country': 'India',
            'admin1': 'Manipur',
            'admin2': district,
            'latitude': lat,
            'longitude': lon,
            'fatalities': fatalities
        })

    df = pd.DataFrame(data)
    
    # Sort by date
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df.sort_values('event_date').reset_index(drop=True)
    
    os.makedirs('data/raw', exist_ok=True)
    df.to_csv('data/raw/acled_manipur_synthetic.csv', index=False)
    print(f"Generated {num_records} synthetic records and saved to data/raw/acled_manipur_synthetic.csv.")

if __name__ == "__main__":
    generate_data()

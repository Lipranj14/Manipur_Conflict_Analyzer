import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
import pickle
import os

def prepare_model_data(df):
    """Aggregate data to district-month level and create target variable."""
    # We want to predict if a district-month has High/Low conflict
    # Aggregate by district and year_month
    agg_df = df.groupby(['district', 'year_month']).agg({
        'fatalities': 'sum',
        'event_count': 'sum',
        'rolling_event_count_30d': 'max' # Max rolling count in that month
    }).reset_index()
    
    # We also need features like season. Let's merge it back.
    # Get the last season used in that month for that district
    season_df = df.drop_duplicates(subset=['district', 'year_month'], keep='last')[['district', 'year_month', 'season', 'year']]
    agg_df = pd.merge(agg_df, season_df, on=['district', 'year_month'], how='left')
    
    # Define Target: High conflict if event_count > 3 or fatalities > 1, else Low
    agg_df['intensity_target'] = np.where((agg_df['event_count'] > 3) | (agg_df['fatalities'] > 1), 'High', 'Low')
    
    return agg_df

def train_models(data_path='data/processed/manipur_processed.csv'):
    print("Loading processed data...")
    df = pd.read_csv(data_path)
    
    print("Preparing data for modeling...")
    model_df = prepare_model_data(df)
    
    # Features & Target
    # Convert categorical to numeric
    le_district = LabelEncoder()
    le_season = LabelEncoder()
    le_target = LabelEncoder()
    
    model_df['district_encoded'] = le_district.fit_transform(model_df['district'])
    model_df['season_encoded'] = le_season.fit_transform(model_df['season'].astype(str))
    
    # We will use 'district', 'season', 'year' to predict risk
    # This is a bit simplistic, in reality, lag features would be used.
    # Let's add lag_event_count (events in previous month)
    model_df = model_df.sort_values(by=['district', 'year_month'])
    model_df['lag_event_count'] = model_df.groupby('district')['event_count'].shift(1).fillna(0)
    model_df['lag_fatalities'] = model_df.groupby('district')['fatalities'].shift(1).fillna(0)
    
    X = model_df[['district_encoded', 'season_encoded', 'lag_event_count', 'lag_fatalities', 'year']]
    y = le_target.fit_transform(model_df['intensity_target'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("\n--- Training Logistic Regression ---")
    lr_model = LogisticRegression(class_weight='balanced', max_iter=1000)
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)
    
    print(f"Accuracy: {accuracy_score(y_test, lr_preds):.4f}")
    print(f"F1 Score: {f1_score(y_test, lr_preds):.4f}")
    print(classification_report(y_test, lr_preds, target_names=le_target.classes_))
    
    print("\n--- Training Random Forest ---")
    rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    
    rf_acc = accuracy_score(y_test, rf_preds)
    rf_f1 = f1_score(y_test, rf_preds)
    print(f"Accuracy: {rf_acc:.4f}")
    print(f"F1 Score: {rf_f1:.4f}")
    print(classification_report(y_test, rf_preds, target_names=le_target.classes_))
    
    # Save the best model (RF typically performs better here)
    os.makedirs('models', exist_ok=True)
    with open('models/random_forest_conflict_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
        
    # Save encoders for the Streamlit app
    with open('models/label_encoders.pkl', 'wb') as f:
        pickle.dump({
            'district': le_district,
            'season': le_season,
            'target': le_target,
            'feature_names': list(X.columns)
        }, f)
        
    # Feature Importance for Random Forest
    importances = rf_model.feature_importances_
    features = X.columns
    fi_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    fi_df = fi_df.sort_values(by='Importance', ascending=False)
    
    print("\n--- Random Forest Feature Importance ---")
    print(fi_df)
    
    print("\nSaved Random Forest model and encoders to models/")

if __name__ == "__main__":
    train_models()

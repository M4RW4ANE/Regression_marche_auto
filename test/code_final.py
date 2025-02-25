import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder

def load_data(filepath):
    """Load dataset from CSV file."""
    data = pd.read_csv(filepath, sep=",")
    data.drop(columns=['New_Price'], inplace=True, errors='ignore')
    data.dropna(inplace=True)
    return data

def preprocess_data(data):
    """Clean and preprocess data."""
    features = data[['Year', 'Kilometers_Driven', 'Fuel_Type', 'Transmission', 'Owner_Type', 'Mileage', 'Engine', 'Power', 'Seats']].copy()
    target = data['Price']
    
    # Convert numerical strings to numbers
    def extract_numeric(value):
        if isinstance(value, str):
            value = ''.join([c for c in value if c.isdigit() or c == '.'])
            return float(value) if value else np.nan
        return value
    
    features.loc[:, 'Mileage'] = features['Mileage'].apply(extract_numeric)
    features.loc[:, 'Engine'] = features['Engine'].apply(extract_numeric)
    features.loc[:, 'Power'] = features['Power'].apply(extract_numeric)
    
    # One-hot encoding categorical variables
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    categorical_features = ['Fuel_Type', 'Transmission', 'Owner_Type']
    encoded_data = encoder.fit_transform(features[categorical_features])
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_features))
    
    features = features.drop(columns=categorical_features).reset_index(drop=True)
    processed_features = pd.concat([features, encoded_df], axis=1)
    
    return processed_features, target, encoder

def train_model(features, target):
    """Train a RandomForest model."""
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    print("R2 Score:", r2_score(y_test, predictions))
    print("Mean Squared Error:", mean_squared_error(y_test, predictions))
    
    return model

def save_model(model, encoder, filepath="model.joblib"):
    """Save the trained model and encoder."""
    joblib.dump({'model': model, 'encoder': encoder}, filepath)
    print(f"Model saved to {filepath}")

if __name__ == "__main__":
    data = load_data("Datas/train.csv")
    features, target, encoder = preprocess_data(data)
    model = train_model(features, target)
    save_model(model, encoder)

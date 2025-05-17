import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import os

# Load dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df[['Temperature', 'Relative Humidity', 'Wind Speed', 'Conditions']]
    df.dropna(inplace=True)
    return df

# Train and save the model
def train_model(file_path, model_path='weather_model.pkl'):
    df = load_data(file_path)
    
    X = df[['Temperature', 'Relative Humidity', 'Wind Speed']]
    y = df['Conditions']
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    # Model selection and training
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    }
    
    best_model = None
    best_score = 0
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        score = accuracy_score(y_test, model.predict(X_test))
        print(f'{name} Accuracy: {score:.4f}')
        
        if score > best_score:
            best_score = score
            best_model = model
    
    # Hyperparameter tuning for RandomForest
    param_grid = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]}
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3)
    grid_search.fit(X_train, y_train)
    
    final_model = grid_search.best_estimator_ if grid_search.best_score_ > best_score else best_model
    
    # Save model and label encoder
    with open(model_path, 'wb') as f:
        pickle.dump((final_model, label_encoder), f)
    
    print("Best model trained and saved successfully!")

# Load model once to prevent multiple loads
_model, _label_encoder = None, None

def load_model(model_path='weather_model.pkl'):
    global _model, _label_encoder
    if _model is None or _label_encoder is None:
        with open(model_path, 'rb') as f:
            _model, _label_encoder = pickle.load(f)

# Predict weather condition
def predict_condition(temp, humidity, wind_speed, model_path='weather_model.pkl'):
    load_model(model_path)
    input_data = np.array([[temp, humidity, wind_speed]])
    prediction = _model.predict(input_data)
    return _label_encoder.inverse_transform(prediction)[0]

# Train the model (Run once)
if __name__ == "__main__":
    train_model('beng_dataset.csv')
    print("Use predict_condition(temp, humidity, wind_speed) to get weather condition.")

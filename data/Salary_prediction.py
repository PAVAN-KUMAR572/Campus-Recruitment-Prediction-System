import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

# Load Dataset
df = pd.read_csv('data/Salary_prediction_data.csv')
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df.fillna(0, inplace=True)

# Keep only placed students for training
df = df[df['PlacementStatus'] == 'Placed']

# Feature Selection
x = df.drop(['StudentId', 'salary', 'PlacementStatus'], axis=1)
y = df['salary']

# Encode Categorical Columns
x['Internship'] = x['Internship'].map({'Yes': 1, 'No': 0})
x['Hackathon'] = x['Hackathon'].map({'Yes': 1, 'No': 0})

# Train Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Hyperparameter Tuning
param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'bootstrap': [True, False]
}

rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(x_train_scaled, y_train)

best_model = grid_search.best_estimator_

# Evaluate
ypred = best_model.predict(x_test_scaled)
print(f"Best Params: {grid_search.best_params_}")
print(f"Fine-tuned Salary R2 Score: {r2_score(y_test, ypred):.4f}")
print(f"Fine-tuned Salary MAE: {mean_absolute_error(y_test, ypred):.2f}")

# Save Model and Scaler
with open('salary_model.pkl', 'wb') as f:
    pickle.dump({
        'model': best_model,
        'scaler': scaler,
        'features': list(x.columns)
    }, f)

print("Fine-tuned salary model saved to salary_model.pkl")
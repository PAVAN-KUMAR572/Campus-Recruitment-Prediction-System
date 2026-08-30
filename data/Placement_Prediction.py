import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Load Dataset
df = pd.read_csv('data/Placement_Prediction_data.csv')
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df.fillna(0, inplace=True)

# Feature Selection
x = df.drop(['StudentId', 'PlacementStatus'], axis=1)
y = df['PlacementStatus']

# Encode Categorical Columns
x['Internship'] = x['Internship'].map({'Yes': 1, 'No': 0})
x['Hackathon'] = x['Hackathon'].map({'Yes': 1, 'No': 0})

# Train Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

# Scaling
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Hyperparameter Tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'criterion': ['gini', 'entropy']
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(x_train_scaled, y_train)

best_model = grid_search.best_estimator_

# Evaluate
ypred = best_model.predict(x_test_scaled)
print(f"Best Params: {grid_search.best_params_}")
print(f"Fine-tuned Placement Accuracy: {accuracy_score(y_test, ypred):.4f}")
print(classification_report(y_test, ypred))

# Save Model and Scaler
with open('placement_model.pkl', 'wb') as f:
    pickle.dump({
        'model': best_model,
        'scaler': scaler,
        'features': list(x.columns)
    }, f)

print("Fine-tuned placement model saved to placement_model.pkl")
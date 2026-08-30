import pandas as pd


df_p = pd.read_csv('data/Placement_Prediction_data.csv')
df_p = df_p.loc[:, ~df_p.columns.str.contains('^Unnamed')]
df_p['Internship'] = df_p['Internship'].map({'Yes': 1, 'No': 0})
df_p['Hackathon'] = df_p['Hackathon'].map({'Yes': 1, 'No': 0})
df_p['PlacementStatus'] = df_p['PlacementStatus'].map({'Placed': 1, 'NotPlaced': 0})

# Correlation for placement
corr_p = df_p.drop(['StudentId'], axis=1).corr()
print("Placement Correlations:")
print(corr_p['PlacementStatus'].sort_values(ascending=False))

df_s = pd.read_csv('data/Salary_prediction_data.csv')
df_s = df_s.loc[:, ~df_s.columns.str.contains('^Unnamed')]
df_s = df_s[df_s['PlacementStatus'] == 'Placed']
df_s['Internship'] = df_s['Internship'].map({'Yes': 1, 'No': 0})
df_s['Hackathon'] = df_s['Hackathon'].map({'Yes': 1, 'No': 0})

# Correlation for salary
corr_s = df_s.drop(['StudentId', 'PlacementStatus'], axis=1).corr()
print("\nSalary Correlations:")
print(corr_s['salary'].sort_values(ascending=False))

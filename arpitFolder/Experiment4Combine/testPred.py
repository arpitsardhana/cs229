import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

df1 = pd.read_csv(Path("heart_failure_clinical_records_dataset.csv"))
df2 = pd.read_csv(Path("heart_statlog_cleveland_hungary_final.csv"))

clf1 = LogisticRegression(penalty='l2', solver='liblinear', random_state=42)
y1 = df1["DEATH_EVENT"]
x1 = df1.drop(columns=['DEATH_EVENT'])
clf1.fit(x1, y1)

clf2 = LogisticRegression(penalty='l2', solver='liblinear', random_state=42)
y2 = df2["target"]
x2 = df2.drop(columns=['target'])
clf2.fit(x2, y2)

merged_df = pd.read_csv(Path("merged_data.csv"))

t1 = merged_df[['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction',
                               'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex',
                               'smoking', 'time']]

t2 = merged_df[['age', 'sex', 'chest pain type', 'resting bp s', 'cholesterol',
                               'fasting blood sugar', 'resting ecg', 'max heart rate', 'exercise angina',
                               'oldpeak', 'ST slope']]
y0 = clf1.predict(t1)
y1 = clf2.predict(t2)

print(y0)
print(y1)
y_pred = y0 + y1
print(y_pred)
print(merged_df['output'])
y_test = merged_df['output']
classification_rep = classification_report(y_test, y_pred, target_names=['low risk', 'medium risk', 'high risk'])
accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(classification_rep)

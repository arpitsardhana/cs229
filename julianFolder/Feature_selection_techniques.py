import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import seaborn as sns
import plotly.express as px
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt

from pandas import read_csv
#fatality   = read_csv('../datasets/fatalityDataset/heart_failure_clinical_records_dataset.csv')
disease     = read_csv('../datasets/diseaseDataset/heart_statlog_cleveland_hungary_final.csv')
disease.head()

#Substitute Missing/Zero values with Mean #cholestrol
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=0, strategy = 'mean')
imputer = imputer.fit(disease[['cholesterol']])
disease['cholesterol'] = imputer.transform(disease[['cholesterol']])

#Substitute Missing values with Mean #resting bp
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = 0, strategy = 'mean')
imputer = imputer.fit(disease[['resting bp s']])
disease['resting bp s'] = imputer.transform(disease[['resting bp s']])

target          = "target"
y_disease       = disease[target].values
x_disease       = disease.drop("target", axis = 1)
feature_col     = x_disease.columns.to_list()


#Feature importance (Plotting 10 important features)

model = ExtraTreesClassifier()
model.fit(x_disease,y_disease)
print(model.feature_importances_)
# Plot graph of feature importance for better visualization
feat_importance = pd.Series(model.feature_importances_, index=x_disease.columns)
feat_importance.nlargest(10).plot(kind='barh')
plt.show()



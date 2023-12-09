import os
import numpy as np
import pandas as pd
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
warnings.filterwarnings("ignore")
pd.set_option("display.max_rows",None)
from sklearn import preprocessing
import matplotlib
from sklearn.preprocessing import LabelEncoder
import streamlit as st
#import lime.lime_tabular as lt
from grid_search_utils import plot_grid_search, table_grid_search

from pandas import read_csv
#fatality   = read_csv('../datasets/fatalityDataset/heart_failure_clinical_records_dataset.csv')
disease     = read_csv('../datasets/diseaseDataset/heart_statlog_cleveland_hungary_final.csv')

#Plotting a few statistics

px.imshow(disease.corr(),title="Correlation Plot")

#Plotting
fig=px.histogram(disease,
                 x     = "target",
                 color = "sex",
                 hover_data = disease.columns,
                 title = "Distribution of Heart Diseases",
                 barmode = "group")
fig.show()


fig=px.histogram(disease,
                 x="chest pain type",
                 color      = "sex",
                 hover_data = disease.columns,
                 title      = "Types of Chest Pain"
                )
fig.show()

fig=px.histogram(disease,
                 x="sex",
                 hover_data=disease.columns,
                 title="Sex Ratio in the Data")
fig.show()

fig=px.histogram(disease,
                 x          =   "resting ecg",
                 hover_data =   disease.columns,
                 title      =   "Distribution of Resting ECG")
fig.show()

plt.figure(figsize=(15,10))
sns.pairplot(disease, hue="target")
plt.title("More insight into the data")
plt.legend("Heart Disease Dataset")
plt.tight_layout()
plt.plot()


plt.figure(figsize=(15,10))
for i,col in enumerate(disease.columns,1):
    plt.subplot(4,3,i)
    plt.title(f"Distribution of {col} Data")
    sns.histplot(disease[col],kde=True)
    plt.tight_layout()
    plt.plot()

fig = px.box(disease,y="age",x="target",title=f"Distribution of Age")
fig.show()


fig = px.box(disease,y="resting bp s",x="target",title=f"Distribution of Resting BP",color="sex")
fig.show()

fig = px.box(disease,y="cholesterol",x="target",title=f"Distribution of Cholesterol")
fig.show()

fig = px.box(disease,y="oldpeak",x="target",title=f"Distribution of Oldpeak")
fig.show()

fig = px.box(disease,y="max heart rate",x="target",title=f"Distribution of MaxHR")
fig.show()

disease.info()

disease.isnull().sum()

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

#Feature Scaling
#1) Distance Based Algorithms
#2) Tree-Based Algorithms :

for col in ['sex', 'chest pain type', 'fasting blood sugar', 'resting ecg','ST slope']:
    print(f"The distribution of categorical values in the {col} is : ")
    print(disease[col].value_counts())

## Creaeting one hot encoded features for working with non-tree based algorithms
x_nontree = pd.get_dummies(disease, columns=['sex', 'chest pain type', 'fasting blood sugar', 'resting ecg', 'ST slope'], drop_first =False)
x_nontree.head()


## Getting the target column at the end (disease non_tree)
target          = "target"
y_non_tree      = x_nontree[target].values
x_nontree.drop("target", axis=1, inplace=True)
#disease_nontree = pd.concat([disease_nontree,disease[target]],axis=1)
#print(disease_nontree.head())

## Getting column names
feature_col_nontree = x_nontree.columns.to_list()
#feature_col_nontree.remove(target)
print(feature_col_nontree)

#Separate X and Y for disease
target          = "target"
y_disease       = disease[target].values
x_disease       = disease.drop("target", axis = 1)
feature_col     = x_disease.columns.to_list()

print(x_nontree.head())
print(y_non_tree)


# Imports for ROC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

# Imports for PR Curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn import metrics


def ROC(model, X, Y, name= "LogisticRegression"):
    print(Y)
    trainX, testX, trainy, testy = train_test_split(X, Y, test_size=0.5, random_state=2)

    ns_probs = [0 for _ in range(len(testy))]
    model.fit(trainX, trainy)

    # predict probabilities
    lr_probs = model.predict_proba(testX)
    # keep probabilities for the positive outcome only
    lr_probs = lr_probs[:, 1]
    # calculate scores
    ns_auc = roc_auc_score(testy, ns_probs)
    lr_auc = roc_auc_score(testy, lr_probs)
    # summarize scores
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('Logistic: ROC AUC=%.3f' % (lr_auc))
    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(testy, lr_probs)
    # plot the roc curve for the model
    plt.plot(ns_fpr, ns_tpr, linestyle='--') # label='No Skill'
    plt.plot(lr_fpr, lr_tpr, marker='.', label=name)
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # show the plot
    #plt.show()


def plot_precision_recall_curve(model, X, Y, name= "LogisticRegression"):

    trainX, testX, trainy, testy = train_test_split(X, Y, test_size=0.5, random_state=2)

    ns_probs = [0 for _ in range(len(testy))]
    model.fit(trainX, trainy)

    # Predict probabilities
    lr_probs = model.predict_proba(testX)
    # keep probabilities for the positive outcome only
    lr_probs = lr_probs[:, 1]
    # Predict class values
    yhat = model.predict(testX)
    lr_precision, lr_recall, _ = precision_recall_curve(testy, lr_probs)
    lr_f1, lr_auc = f1_score(testy, yhat), auc(lr_recall, lr_precision)
    # Summarize scores
    print(name + ': f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
    # Plot the precision-recall curves
    no_skill = len(testy[testy == 1]) / len(testy)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--')#label='No Skill'
    plt.plot(lr_recall, lr_precision, marker='.', label=name)
    # Axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # Show the legend
    plt.legend()
    # show the plot
    #plt.show()


from sklearn.metrics import confusion_matrix, accuracy_score
def evaluate_model(cls, X, Y):
    trainX, testX, trainy, testy = train_test_split(X, Y, test_size=0.5, random_state=2)
    cls.fit(trainX, trainy)
    print("Train Accuracy :", accuracy_score(trainy, cls.predict(trainX)))
    print("Train Confusion Matrix:")
    print(confusion_matrix(trainy, cls.predict(trainX)))
    print("-"*50)
    print("Test Accuracy :", accuracy_score(testy, cls.predict(testX)))
    print("Test Confusion Matrix:")
    print(confusion_matrix(testy, cls.predict(testX)))

#Logistic Regression (with Hyperparameter tuning)
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV

solvers     = ['newton-cg', 'lbfgs', 'liblinear']
penalty     = ['l2']
c_values    = [100, 10, 1.0, 0.1, 0.01]
grid = dict(solver=solvers, penalty=penalty, C=c_values)

cv          = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=3)
clf         = LogisticRegression()

grid_search = GridSearchCV(estimator=clf, param_grid=grid , cv=cv, scoring='accuracy', error_score=0)
grid_result = grid_search.fit(x_nontree, y_non_tree)

means       = grid_result.cv_results_['mean_test_score']
stds        = grid_result.cv_results_['std_test_score']
params      = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

#Finding Best Hyperparameters
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))
print("Best estimator:\n{}".format(grid_search.best_estimator_))
#End Logistic Regression (with Hyperparameter tuning)

evaluate_model(clf, x_disease, y_disease)
ROC(clf, x_nontree, y_non_tree,name= "Logistic Regression")
#plot_precision_recall_curve(clf, x_nontree, y_non_tree, name= "Logistic Regression")
plot_grid_search(grid_search)
table_grid_search(grid_search)



#Naive bayes - Hyperparameter Tuning
from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import PowerTransformer

cv_method = RepeatedStratifiedKFold(n_splits=5,  n_repeats=3, random_state=999)
params_NB = {'var_smoothing': np.logspace(0,-9, num=100)}

model = GaussianNB()
gs_NB = GridSearchCV(estimator  =   model,
                     param_grid =   params_NB,
                     cv         =   cv_method,
                     verbose    =   1,
                     scoring    =   'accuracy')

Data_transformed = PowerTransformer().fit_transform(x_nontree)
gs_NB.fit(Data_transformed, y_non_tree)

#Finding Best Hyperparameters
print("Best parameters: {}".format(gs_NB.best_params_))
print("Best cross-validation score: {:.2f}".format(gs_NB.best_score_))
print("Best estimator:\n{}".format(gs_NB.best_estimator_))
# End Naive bayes - Hyperparameter Tuning

# ROC and PR Curve
evaluate_model(clf, x_disease, y_disease)
ROC(model, x_nontree, y_non_tree,name= "Naive Bayes")
#plot_precision_recall_curve(model, x_nontree, y_non_tree,name= "Naive Bayes")
plot_grid_search(grid_search)
table_grid_search(grid_search)



#SVM - Hyperparameter Tuning
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import PowerTransformer

model   = SVC(probability=True)
kernel  = ['poly', 'rbf', 'sigmoid']
C       = [50, 10, 1.0, 0.1, 0.01]
gamma   = ['scale']
grid    = dict(kernel   =   kernel,
               C        =   C,
               gamma    =   gamma)

cv = RepeatedStratifiedKFold(n_splits       =   10,
                             n_repeats      =   3,
                             random_state   =   1)

grid_search = GridSearchCV(estimator    =   model,
                           param_grid   =   grid,
                           n_jobs       =   -1,
                           cv           =   cv,
                           scoring      =   'accuracy',
                           error_score  =   0)
grid_result = grid_search.fit(x_nontree, y_non_tree)


#Summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means   = grid_result.cv_results_['mean_test_score']
stds    = grid_result.cv_results_['std_test_score']
params  = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
#SVM - Hyperparameter Tuning End

evaluate_model(clf, x_disease, y_disease)
ROC(model, x_nontree, y_non_tree,name= "SVM")
#plot_precision_recall_curve(model, x_nontree, y_non_tree, name= "SVM")
plot_grid_search(grid_search)
table_grid_search(grid_search)

#KNN using Hyperparameter Tuning
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

model       = KNeighborsClassifier()
n_neighbors = range(1, 21, 2)
weights     = ['uniform', 'distance']
metric      = ['euclidean', 'manhattan', 'minkowski']
grid        = dict(n_neighbors=n_neighbors,weights=weights,metric=metric)

cv          = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(x_nontree, y_non_tree)

#Summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means   = grid_result.cv_results_['mean_test_score']
stds    = grid_result.cv_results_['std_test_score']
params  = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
# End KNN using Hyperparameter Tuning
evaluate_model(clf, x_disease, y_disease)
ROC(model, x_nontree, y_non_tree,name= "KNN")
#plot_precision_recall_curve(model, x_nontree, y_non_tree,name= "KNN")
plot_grid_search(grid_search)
table_grid_search(grid_search)


#Decision Tree Hyperparameter Tuning
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from grid_search_utils import plot_grid_search, table_grid_search

model = DecisionTreeClassifier()

max_depth           = [2, 3, 5, 10, 20]
min_samples_leaf    = [5, 10, 20, 50, 100]
criterion           = ["gini", "entropy"]

# Define grid search
grid        = dict(max_depth=max_depth, min_samples_leaf =min_samples_leaf, criterion=criterion)
cv          = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(x_disease, y_disease)

# Summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
#End Deciosn Tree Hyperparameter Tuning

evaluate_model(model, x_disease, y_disease)
ROC(model, x_disease, y_disease, name= "Decision Tree")
#plot_precision_recall_curve(model, x_disease, y_disease,name= "Decision Tree")
plot_grid_search(grid_search)
table_grid_search(grid_search)


#BaggingClassifier Hyperparameter Tuning
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from grid_search_utils import plot_grid_search, table_grid_search

model = BaggingClassifier()
n_estimators = [10, 100, 1000]

# Define grid search
grid        = dict(n_estimators=n_estimators)
cv          = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(x_disease, y_disease)

# Summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
#End BaggingClassifier Hyperparameter Tuning

evaluate_model(model, x_disease, y_disease)
ROC(model, x_disease, y_disease, name= "Bagging Classifier")
#plot_precision_recall_curve(model, x_disease, y_disease, name= "Bagging Classifier")
plot_grid_search(grid_search)
table_grid_search(grid_search)


#Random Forest Hyperparameter Tuning
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from grid_search_utils import plot_grid_search, table_grid_search

model        = RandomForestClassifier()
n_estimators = [10, 100, 1000]
max_features = ['sqrt', 'log2']

#Define grid search
grid    = dict(n_estimators=n_estimators,max_features=max_features)
cv      = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(x_disease, y_disease)

# Summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means   = grid_result.cv_results_['mean_test_score']
stds    = grid_result.cv_results_['std_test_score']
params  = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
#End Random Forest Hyperparameter Tuning

evaluate_model(model, x_disease, y_disease)
ROC(model, x_disease, y_disease, name= "Random Forest")
#plot_precision_recall_curve(model, x_disease, y_disease, name= "Random Forest")
plot_grid_search(grid_search)
table_grid_search(grid_search)

#Adaboost Hyperparameter Tuning
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier

#from lwoku import RANDOM_STATE, N_JOBS, VERBOSE, get_prediction
from grid_search_utils import plot_grid_search, table_grid_search

model   = AdaBoostClassifier(random_state=1)
n_estimators =  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20, 30]

grid    = dict(n_estimators=n_estimators)
cv      = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

grid_search = GridSearchCV(model, param_grid=grid, scoring = 'accuracy', cv = 3, n_jobs = -1)
grid_result = grid_search.fit(x_disease, y_disease)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means   = grid_result.cv_results_['mean_test_score']
stds    = grid_result.cv_results_['std_test_score']
params  = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
#End Adaboost Hyperparameter Tuning
evaluate_model(model, x_disease, y_disease)
ROC(model, x_disease, y_disease, name= "Adaboost")
#plot_precision_recall_curve(model, x_disease, y_disease, name= "Adaboost")
plot_grid_search(grid_search)
table_grid_search(grid_search)

plt.show()

#End Adaboost Hyperparameter Tuning





























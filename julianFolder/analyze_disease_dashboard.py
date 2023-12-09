import os
import numpy as np
import pandas as pd
#import pandas_bokeh

import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics._plot import precision_recall_curve, roc_curve, confusion_matrix
from sklearn.metrics import precision_score, recall_score

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import PrecisionRecallDisplay

warnings.filterwarnings("ignore")
pd.set_option("display.max_rows",None)
from sklearn import preprocessing
import matplotlib
from sklearn.preprocessing import LabelEncoder

import streamlit as st
#from streamlit_jupyter import StreamlitPatcher, tqdm
#StreamlitPatcher().jupyter()
#import lime.lime_tabular as lt

from pandas import read_csv
#fatality   = read_csv('../datasets/fatalityDataset/heart_failure_clinical_records_dataset.csv')
disease     = read_csv('../datasets/diseaseDataset/heart_statlog_cleveland_hungary_final.csv')

def st_start():
    st.title("Introduction to building Streamlit WebApp")
    st.sidebar.title("This is the sidebar")
    st.sidebar.markdown("Binary classification for  Disease Dataset!!")
st_start()

fig = px.imshow(disease.corr(),title="Correlation Plot")
st.plotly_chart(fig)

fig1 = px.histogram(disease,
                 x          = "target",
                 color      = "sex",
                 hover_data = disease.columns,
                 title      = "Distribution of Heart Diseases",
                 barmode    = "Group")
st.plotly_chart(fig1)

# fig2=px.histogram(disease,
#                  x          ="chest pain type",
#                  color      = "sex",
#                  hover_data = disease.columns,
#                  title      = "Types of Chest Pain"
#                 )
# st.plotly_chart(fig2)

# fig3=px.histogram(disease,
#                  x         ="sex",
#                  hover_data=disease.columns,
#                  title="Sex Ratio in the Data")
# st.plotly_chart(fig3)

# fig4=px.histogram(disease,
#                  x          =   "resting ecg",
#                  hover_data =   disease.columns,
#                  title      =   "Distribution of Resting ECG")
# st.plotly_chart(fig1)


# #function to plot results of training
#
#
#
# plt.figure(figsize=(15,10))
# sns.pairplot(disease, hue="target")
# plt.title("More insight into the data")
# plt.legend("Heart Disease Dataset")
# plt.tight_layout()
# fig5 = plt.plot()
# st.pyplot(fig5)

# plt.figure(figsize=(15,10))
# for i,col in enumerate(disease.columns,1):
#     plt.subplot(4,3,i)
#     plt.title(f"Distribution of {col} Data")
#     sns.histplot(disease[col],kde=True)
#     plt.tight_layout()
#     #plt.plot()
#     fig6 = plt.plot()
#     st.pyplot(fig6)


#Substitute Missing/Zero values with Mean
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=0, strategy='mean')
imputer = imputer.fit(disease[['cholesterol']])
disease['cholesterol'] = imputer.transform(disease[['cholesterol']])

#Substitute Missing values with Mean
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=0, strategy='mean')
imputer = imputer.fit(disease[['resting bp s']])
disease['resting bp s'] = imputer.transform(disease[['resting bp s']])


## Creaeting one hot encoded features for working with non-tree based algorithms
disease_nontree = pd.get_dummies(disease,columns=['sex', 'chest pain type', 'fasting blood sugar', 'resting ecg','ST slope'],drop_first=False)
disease_nontree.head()

# Getting the target column at the end
target  = "target"
y = disease_nontree[target].values
disease_nontree.drop("target",axis=1,inplace=True)
disease_nontree=pd.concat([disease_nontree,disease[target]],axis=1)
disease_nontree.head()

feature_col_nontree=disease_nontree.columns.to_list()
feature_col_nontree.remove(target)

st.set_option('deprecation.showPyplotGlobalUse', False)

#Plot Metrics
def plot_metrics(metrics_list, y_pred, y_test, y_score, class_names, clf):
    if "Confusion Matrix" in metrics_list:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_pred, y_test)
        cm_display = ConfusionMatrixDisplay(cm).plot()
        st.pyplot(plt.show())
    # if "ROC Curve" in metrics_list:
    #     st.subheader("ROC Curve")
    #     roc_curve(model, x_test, y_test)
    #     st.pyplot()
    if "Precision-Recall Curve" in metrics_list:
        st.subheader("Precision-Recall Curve")
        #prec, recall, _ = precision_recall_curve(y_test, y_score, pos_label=clf.classes_[1])
        pr_display = PrecisionRecallDisplay.from_predictions(y_test, y_pred).plot()
        st.pyplot(plt.show())


#### General Information
class_names = [0, 1]
st.sidebar.subheader("Choose classifier")
classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest"))


# Logistic Regression
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler


if classifier == "Logistic Regression":
    st.sidebar.subheader("Hyperparameters")
    C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key="C_LR")
    max_iter = st.sidebar.slider("Maximum iterations", 100, 500, key="max_iter")
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))


    acc_log = []
    kf = model_selection.StratifiedKFold(n_splits=5)
    for fold, (trn_, val_) in enumerate(kf.split(X=disease_nontree, y=y)):
        X_train = disease_nontree.loc[trn_, feature_col_nontree]
        y_train = disease_nontree.loc[trn_, target]

        X_valid = disease_nontree.loc[val_, feature_col_nontree]
        y_valid = disease_nontree.loc[val_, target]

        # print(pd.DataFrame(X_valid).head())
        ro_scaler = MinMaxScaler()
        X_train   = ro_scaler.fit_transform(X_train)
        X_valid   = ro_scaler.transform(X_valid)

        clf = LogisticRegression()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_valid)
        print(f"The fold is : {fold} : ")
        print(classification_report(y_valid, y_pred))
        acc = roc_auc_score(y_valid, y_pred)
        acc_log.append(acc)
        print(f"The accuracy for Fold {fold + 1} : {acc}")
        y_score = clf.decision_function(X_valid)
        pass


    if st.sidebar.button("Classify", key="classify"):
        st.subheader("Logistic Regression Results")

        st.write("Accuracy:", acc.round(2))
        st.write("Precision:", precision_score(y_valid, y_pred, labels=class_names).round(2))
        st.write("Recall:", recall_score(y_valid, y_pred, labels=class_names).round(2))
        plot_metrics(metrics,y_pred,y_valid,y_score,class_names,clf)














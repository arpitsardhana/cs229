import numpy
import sklearn
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer, MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import learning_curve
from sklearn.metrics import precision_recall_curve
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA

# Evaluate the metrics
def  evaluate_print_metrics(clf, name, X_train, X_test, y_train, y_test, x, y, j):
     # 1. evaluate score
     score = clf.score(X_test, y_test)
     print("Score with Cross Validation", cross_val_score(clf, X_train, y_train, cv=3, scoring="accuracy").mean())
     y_train_pred = cross_val_predict(clf, X_train, y_train, cv=3)

     # 2. Confusion matrix, precision, recall and F1 score
     y_train_pred = cross_val_predict(clf, X_train, y_train, cv=3)
     cm = confusion_matrix(y_train, y_train_pred)
     print("Confusion matrix: \n", cm)
     print("Precision score is ", precision_score(y_train, y_train_pred))
     print("Recall score is ", recall_score(y_train, y_train_pred))
     print("F1 score is ", f1_score(y_train, y_train_pred))

     # 3. Learning curve
     plot_learning_curve(clf, name, x, y, j)

# Plot learning curve
def plot_learning_curve(clf,name, x, y, j):
     lbl = [("r-+,r-"), ("b-+,b-"), ("g-+,g-"), ("go-+,go-")]
     g=["red","blue", "green"]
     train_sizes, train_scores, valid_scores = learning_curve(clf, x,y, train_sizes=np.linspace(0.01, 1.0,40), cv=5, scoring="neg_mean_squared_error")
     train_errors = -train_scores.mean(axis=1)
     valid_errors = -valid_scores.mean(axis=1)
     plt.plot(train_sizes, train_errors, lbl[j%4][0], linewidth=2, label="train-%s"%name)
     plt.plot(train_sizes, valid_errors, lbl[j%4][1], linewidth=3, label="valid-%s"%name)
     plt.legend(loc="upper right")
     plt.xlabel("Training set size")
     plt.ylabel("RMSE")
     plt.grid()
     plt.axis([0, 250, 0, 1])
     plt.legend()
     plt.savefig("Disease-%s-LearningCurve-dec8"%name)

# Plot precision recall curve
def plot_pr_curve(dic):
      lbl = ["b-", "--", "r-", "go-","b+"]
      i =0
      plt.figure(figsize=(6, 5))
      for key in dic:
          precisions, recalls, thresholds = dic[key]
          plt.plot(recalls, precisions, lbl[i%5], linewidth=2, label=key)
          i += 1
      plt.xlabel("Recall")
      plt.ylabel("Precision")
      plt.grid()
      plt.legend(loc="lower left")
      plt.savefig("DiseasePrediction-main-PRCurve-Oct27")

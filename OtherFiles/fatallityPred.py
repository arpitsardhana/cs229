import util
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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

names = [
     "Nearest Neighbors",
     "Linear SVM",
     "RBF SVM",
     "Gaussian Process",
     "Decision Tree",
     "Random Forest",
     "Neural Net",
     "AdaBoost",
     "Naive Bayes",
     #"QDA",
]

classifiers = [
     KNeighborsClassifier(3),
     SVC(kernel="linear", C=0.025, random_state=42),
     SVC(gamma=2, C=1, random_state=42),
     GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42),
     DecisionTreeClassifier(max_depth=5, random_state=42),
     RandomForestClassifier(
         max_depth=5, n_estimators=10, max_features=1, random_state=42
     ),
     MLPClassifier(alpha=1, max_iter=1000, random_state=42),
     AdaBoostClassifier(random_state=42),
     GaussianNB(),
     #QuadraticDiscriminantAnalysis(),
]


x,y = util.load_dataset('heart_failure_clinical_records_dataset.csv', label_col='DEATH_EVENT', add_intercept=True)
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.33, random_state=42)

for name, clf in zip(names, classifiers):
         print("=====",'\n', name, " ", clf, " ")

         clf = make_pipeline(StandardScaler(), clf)
         clf.fit(X_train, y_train)
         score = clf.score(X_test, y_test)
         print("Score of classifier : ", score)

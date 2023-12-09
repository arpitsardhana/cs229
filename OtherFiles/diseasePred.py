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


x,y = util.load_dataset('heart_statlog_cleveland_hungary_final.csv', label_col='target', add_intercept=True)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.33, random_state=42)

X = x
linearly_separable = (X, y)
datasets = [
    linearly_separable,
]
X_train, X_test = x_train, x_test
figure = plt.figure(figsize=(27, 9))

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

cm = plt.cm.RdBu
cm_bright = ListedColormap(["#FF0000", "#0000FF"])
i=1
ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
ax.set_title("Input data")
ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")
ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors="k"
)
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_xticks(())
ax.set_yticks(())
i+= 1
for name, clf in zip(names, classifiers):
         ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
         print("====", '\n', name, " ", clf, " ")

         clf = make_pipeline(StandardScaler(), clf)
         clf.fit(X_train, y_train)
         score = clf.score(X_test, y_test)
         print("Score of classifier: ", score)
        # DecisionBoundaryDisplay.from_estimator(
        #     clf, X, cmap=cm, alpha=0.8, ax=ax, eps=0.5
        # )

         # Plot the training points
         ax.scatter(
             X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k"
         )
         # Plot the testing points
         ax.scatter(
             X_test[:, 0],
             X_test[:, 1],
             c=y_test,
             cmap=cm_bright,
             edgecolors="k",
             alpha=0.6,
         )

         ax.set_xlim(x_min, x_max)
         ax.set_ylim(y_min, y_max)
         ax.set_xticks(())
         ax.set_yticks(())
         ax.set_title(name)
         ax.text(
             x_max - 0.3,
             y_min + 0.3,
             ("%.2f" % score).lstrip("0"),
             size=15,
             horizontalalignment="right",
         )
         i += 1

#plt.tight_layout()
#plt.show()

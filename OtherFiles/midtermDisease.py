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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import learning_curve
from sklearn.metrics import precision_recall_curve

# Loads the dataset
def loadDataSet():
    x,y = util.load_dataset('heart_statlog_cleveland_hungary_final.csv', label_col='target', add_intercept=True)
    X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.33, random_state=42)

    return X_train, X_test, y_train, y_test, x, y

# Makes the pipeline
def makePipeline(clf):
    return make_pipeline(StandardScaler(), clf)

# Get the classifiers
def getClassifiers():
     classifier = [
         (LogisticRegression(random_state=42),"Logistic Regression"),
         (SVC(kernel="linear", C=0.025, random_state=42, probability=True), "Linear SVM"),
         (RandomForestClassifier(max_depth=5, n_estimators=10, max_features=4, random_state=42), "Random Forest"),
     ]
     return classifier

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
    lbl = [("r-+,r-"), ("b-+,b-"), ("g-+,g-")]
    g=["red","blue", "green"]
    train_sizes, train_scores, valid_scores = learning_curve(clf, x,y, train_sizes=np.linspace(0.01, 1.0,40), cv=5, scoring="neg_mean_squared_error")
    train_errors = -train_scores.mean(axis=1)
    valid_errors = -valid_scores.mean(axis=1)
    plt.plot(train_sizes, train_errors, lbl[j][0], linewidth=2, label="train-%s"%name, color=g[j])
    plt.plot(train_sizes, valid_errors, lbl[j][1], linewidth=3, label="valid-%s"%name, color=g[j])
    plt.legend(loc="upper right")
    plt.xlabel("Training set size")
    plt.ylabel("RMSE")
    plt.grid()
    plt.axis([0, 250, 0, 1])
    plt.legend()
    plt.savefig("DiseasePrediction-main-%s-LearningCurve-oct27"%name)

# Plot precision recall curve
def plot_pr_curve(dic):
     lbl = ["b-", "--", "r-"]
     i =0
     plt.figure(figsize=(6, 5))
     for key in dic:
         precisions, recalls, thresholds = dic[key]
         plt.plot(recalls, precisions, lbl[i], linewidth=2, label=key)
         i += 1
     plt.xlabel("Recall")
     plt.ylabel("Precision")
     plt.grid()
     plt.legend(loc="lower left")
     plt.savefig("DiseasePrediction-main-PRCurve-Oct27")


def main():
    dic, j = {}, 0
    X_train, X_test, y_train, y_test, x, y = loadDataSet()
    models = getClassifiers()
    for clf, name in models:
         print("=====",'\n', name, " ", clf, " ")

         clf = makePipeline(clf)
         clf.fit(X_train, y_train)

         evaluate_print_metrics(clf, name, X_train, X_test, y_train, y_test, x, y, j)
         y_probas = cross_val_predict(clf, X_train, y_train, cv=3,
                                      method="predict_proba")[:, 1]
         dic[name] = precision_recall_curve(y_train, y_probas)
         j += 1
    plot_pr_curve(dic)

if __name__ == "__main__":
    main()

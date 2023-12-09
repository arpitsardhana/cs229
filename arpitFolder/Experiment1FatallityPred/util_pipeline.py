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

# Loads the dataset
def loadDataSet():
    #x,y = util.load_dataset('heart_statlog_cleveland_hungary_final.csv', label_col='target', add_intercept=True)
    #x,y = util.load_dataset('heart_failure_clinical_records_dataset.csv', label_col='DEATH_EVENT', add_intercept=False)
    hf = pd.read_csv(Path("heart_failure_clinical_records_dataset.csv"))
    hf["creatine_per_sodium"] = hf["serum_creatinine"] / hf["serum_sodium"]

    hf["age_cat"] = pd.cut(hf["age"],
                                bins=[0.,20.,30.,35.,40.,45.,50.,55.,60.,65.,70.,75.,80.,85.,90.,np.inf],
                                labels=[1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15])
    hf = hf.drop(columns=['age'])




    y = hf["DEATH_EVENT"]
    x = hf.drop(columns=['DEATH_EVENT'])

     #x_red, _ = getReducedDimesion(x, 0.99999)
     #x = x_red

    X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.33, random_state=42)

    return X_train, X_test, y_train, y_test, x, y

# Makes the pipeline
def makePipeline(clf):
    default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    min_max_pipeline = make_pipeline(SimpleImputer(strategy="median"), MinMaxScaler())

    log_pipeline = make_pipeline(SimpleImputer(strategy="median"),
                                  FunctionTransformer(np.log, feature_names_out="one-to-one"),
                   #              StandardScaler())
                                  MinMaxScaler(feature_range=(-1,1)))


    preprocessing = ColumnTransformer([#("log", log_pipeline, ["creatinine_phosphokinase", "platelets", "serum_creatinine", "serum_sodium"] #[2,6,7,8]Intercep[3,7,8,9],
                                         #),
                                         ("minmax", min_max_pipeline, [12]),
                                        ], remainder=default_num_pipeline)
    return make_pipeline(preprocessing, clf), preprocessing

# Get the classifiers
def getClassifiers():
  curr =  [
      (getLogisticRegression,"Logistic Regression"),
      #(getSvcKernel, "Linear SVM"), #0.025, kernel=linear
      #(getRandomForest, "Random Forest"),
      (getMLPNeuralNetwork,"Neural Net"),
      (getAdaBoost,"AdaBoost"),



      #(getKNN, "Nearest neighbour"),
      #(getDecisionTree, "Decision Tree"),
      #(getNaiveBayes, "Naie bayes"),
      #(getQDA, "QDA"),
  ]
  new = [ (getGDA, "GDA"),
         (getSvcRBF,"RBF SVM"),
        ]
  ret = curr

  #ret += curr
  #ret =[]
  #for c,n in classifier:
  #    ret.append((makePipeline(c)[0], n))
  return ret

# grid cv search
def GridSearchPipeline():
 clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=4, random_state=42)
 _, pipeline = makePipeline(clf)
 full = Pipeline([
     ("pre", pipeline),
     ("random", clf)
 ])
 param_grid = [{'random__max_features':[4,6,8],
                'random__max_depth':[5,6,7],
                'random__n_estimators':[10,11,12]}]
 grid_search = GridSearchCV(full, param_grid, cv=3, scoring='neg_root_mean_squared_error')
 return [(grid_search, "GridSearchRandomForest")]

def getLogisticRegression(df):
     hf = pd.read_csv(Path("heart_failure_clinical_records_dataset.csv"))
     #hf = hf.sample(frac=1)
     hf["creatine_per_sodium"] = hf["serum_creatinine"] / hf["serum_sodium"]

     #hf["age_cat"] = pd.cut(hf["age"],
     #                            bins=[0.,20.,30.,35.,40.,45.,50.,55.,60.,65.,70.,75.,80.,85.,90.,np.inf],
     #                            labels=[1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15])
     #hf = hf.drop(columns=['age'])




     y = hf["DEATH_EVENT"]
     x = hf.drop(columns=['DEATH_EVENT'])

      #x_red, _ = getReducedDimesion(x, 0.99999)
      #x = x_red

     X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.20, random_state=42)

     clf = LogisticRegression(penalty='l2', solver='liblinear', random_state=42)

     default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
     min_max_pipeline = make_pipeline(SimpleImputer(strategy="median"), MinMaxScaler())

     log_pipeline = make_pipeline(SimpleImputer(strategy="median"),
                                   FunctionTransformer(np.log, feature_names_out="one-to-one"),
                                   #StandardScaler(),
                                   MinMaxScaler(feature_range=(-1,1)))


     preprocessing = ColumnTransformer([("log", log_pipeline, ["creatinine_phosphokinase", "platelets", "serum_creatinine", "serum_sodium"] #[2,6,7, 8]Intercep[3,7,8,9],
                                          ),
                                          ("minmax", min_max_pipeline, [12]),
                                         ], remainder=default_num_pipeline)

     # Any grid search parameter to be entered here.

     clf = make_pipeline(preprocessing, clf)

     #full = Pipeline([ ("pre", preprocessing), ("lr", clf) ])

     #param_grid = [{'lr__penalty':['l1','l2'],
     #               'lr__solver':['liblinear','saga'],
     #               'lr__max_iter':[100,1000]}]
     #grid_search = GridSearchCV(full, param_grid, cv=3, scoring='neg_root_mean_squared_error')

     #grid_search.fit(X_train, y_train)

     #print(grid_search.best_params_)

     clf.fit(X_train, y_train)

     return clf, X_train, X_test, y_train, y_test, x, y

def getAdaBoost(df):
      hf = pd.read_csv(Path("heart_failure_clinical_records_dataset.csv"))
      #hf = hf.sample(frac=1)
      hf["creatine_per_sodium"] = hf["serum_creatinine"] / hf["serum_sodium"]

      hf["age_cat"] = pd.cut(hf["age"],
                                  bins=[0.,20.,30.,35.,40.,45.,50.,55.,60.,65.,70.,75.,80.,85.,90.,np.inf],
                                  labels=[1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15])
      hf = hf.drop(columns=['age'])




      y = hf["DEATH_EVENT"]
      x = hf.drop(columns=['DEATH_EVENT'])

       #x_red, _ = getReducedDimesion(x, 0.99999)
       #x = x_red

      X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.33, random_state=42)

      RandomForest = RandomForestClassifier(max_depth=5, n_estimators=11, max_features=6, random_state=42)
      clf = AdaBoostClassifier(estimator=RandomForest, n_estimators=40, random_state=42)

      default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
      min_max_pipeline = make_pipeline(SimpleImputer(strategy="median"), MinMaxScaler())

      log_pipeline = make_pipeline(SimpleImputer(strategy="median"),
                                    FunctionTransformer(np.log, feature_names_out="one-to-one"),
                     #              StandardScaler())
                                    MinMaxScaler(feature_range=(-1,1)))


      preprocessing = ColumnTransformer([("log", log_pipeline, ["creatinine_phosphokinase", "platelets", "serum_creatinine", "serum_sodium"] #[2,6,  7, 8]Intercep[3,7,8,9],
                                           ),
                                           ("minmax", min_max_pipeline, [12]),
                                          ], remainder=default_num_pipeline)

      # Any grid search parameter to be entered here.
      #full = Pipeline([ ("pre", preprocessing), ("lr", clf) ])
      #RandomForest = RandomForestClassifier(max_depth=5, n_estimators=11, max_features=6, random_state=42)
      #param_grid = [{'lr__n_estimators':[38, 40, 42],
      #                'lr__estimator':[RandomForest],
      #               'lr__learning_rate': [0.8,1,1.1],
      #                'lr__algorithm':['SAMME.R']}]
      #grid_search = GridSearchCV(full, param_grid, cv=3, scoring='neg_root_mean_squared_error')

      #grid_search.fit(X_train, y_train)

      #print(grid_search.best_params_)
      #return grid_search, X_train, X_test, y_train, y_test, x, y

      clf = make_pipeline(preprocessing, clf)
      clf.fit(X_train, y_train)

      return clf, X_train, X_test, y_train, y_test, x, y

def getSvcKernel(df):
       hf = pd.read_csv(Path("heart_failure_clinical_records_dataset.csv"))
       #hf = hf.sample(frac=1)
       hf["creatine_per_sodium"] = hf["serum_creatinine"] / hf["serum_sodium"]
       hf["age_cat"] = pd.cut(hf["age"],
                                   bins=[0.,20.,30.,35.,40.,45.,50.,55.,60.,65.,70.,75.,80.,85.,90.,np.inf],
                                   labels=[1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15])
       hf = hf.drop(columns=['age'])
       y = hf["DEATH_EVENT"]
       x = hf.drop(columns=['DEATH_EVENT'])
       X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.33, random_state=42)
       clf = SVC(kernel="linear", C=0.4, random_state=42, probability=True)

       default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
       min_max_pipeline = make_pipeline(SimpleImputer(strategy="median"), MinMaxScaler())
       log_pipeline = make_pipeline(SimpleImputer(strategy="median"),
                                     FunctionTransformer(np.log, feature_names_out="one-to-one"),
                                     MinMaxScaler(feature_range=(-1,1)))

       preprocessing = ColumnTransformer([#("log", log_pipeline, ["creatinine_phosphokinase", "platelets", "serum_creatinine", "serum_sodium"] #[2,   6,  7, 8]Intercep[3,7,8,9],
                                            #),
                                            ("minmax", min_max_pipeline, [12]),
                                           ], remainder=default_num_pipeline)

       # Any grid search parameter to be entered here.
       #full = Pipeline([ ("pre", preprocessing), ("lr", clf) ])

       #param_grid = [{'lr__C':[0.4,0.33],
       #              'lr__kernel':['linear'],
       #              'lr__gamma':['scale','auto', 2.0],
       #              'lr__class_weight':[None, 'balanced']}]
       #grid_search = GridSearchCV(full, param_grid, cv=3, scoring='neg_root_mean_squared_error')

       #grid_search.fit(X_train, y_train)

       #print(grid_search.best_params_)
       #return grid_search, X_train, X_test, y_train, y_test, x, y

       clf = make_pipeline(preprocessing, clf)
       clf.fit(X_train, y_train)
       return clf, X_train, X_test, y_train, y_test, x, y

def getRandomForest(df):
        hf = pd.read_csv(Path("heart_failure_clinical_records_dataset.csv"))
        #hf = hf.sample(frac=1)
        y = hf["DEATH_EVENT"]
        x = hf.drop(columns=['DEATH_EVENT'])
        X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.33, random_state=42)

        clf = RandomForestClassifier(max_depth=5, n_estimators=11, max_features=6, random_state=42)

        default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
        min_max_pipeline = make_pipeline(SimpleImputer(strategy="median"), MinMaxScaler())

        log_pipeline = make_pipeline(SimpleImputer(strategy="median"),
                                      FunctionTransformer(np.log, feature_names_out="one-to-one"),
                                      MinMaxScaler())


        preprocessing = ColumnTransformer([], remainder=default_num_pipeline)

        # Any grid search parameter to be entered here.
        #full = Pipeline([
        #    ("pre", preprocessing),
        #    ("random", clf)
        #    ])
        #param_grid = [{'random__max_features':[5,6,7],
        #               'random__max_depth':[4,5],
        #               'random__n_estimators':[9,11,2],
        #               'random__criterion': ['gini', 'entropy'],
        #               'random__class_weight':['balanced', None]}]
        #clf = GridSearchCV(full, param_grid, cv=3, scoring='neg_root_mean_squared_error')

        clf = make_pipeline(preprocessing, clf)
        clf.fit(X_train, y_train)
        #print(clf.best_params_)

        return clf, X_train, X_test, y_train, y_test, x, y

def getMLPNeuralNetwork(df):
        hf = pd.read_csv(Path("heart_failure_clinical_records_dataset.csv"))
        #hf = hf.sample(frac=1)
        hf["creatine_per_sodium"] = hf["serum_creatinine"] / hf["serum_sodium"]
        hf["age_cat"] = pd.cut(hf["age"],
                                    bins=[0.,20.,30.,35.,40.,45.,50.,55.,60.,65.,70.,75.,80.,85.,90.,np.inf],
                                    labels=[1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15])
        hf = hf.drop(columns=['age'])
        y = hf["DEATH_EVENT"]
        x = hf.drop(columns=['DEATH_EVENT'])
        X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.33, random_state=42)
        clf = MLPClassifier(hidden_layer_sizes=100, activation='relu', solver='lbfgs', alpha=1, learning_rate='adaptive', max_iter=10000, tol=1e-4, random_state=42)

        default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
        min_max_pipeline = make_pipeline(SimpleImputer(strategy="median"), MinMaxScaler())
        log_pipeline = make_pipeline(SimpleImputer(strategy="median"),
                                      FunctionTransformer(np.log, feature_names_out="one-to-one"),
                                      MinMaxScaler(feature_range=(-1,1)))

        preprocessing = ColumnTransformer([#("log", log_pipeline, ["creatinine_phosphokinase", "platelets",                     "serum_creatinine", "serum_sodium"] #[2,   6,  7, 8]Intercep[3,7,8,9],
                                            # ),
                                             ("minmax", min_max_pipeline, [12]),
                                            ], remainder=default_num_pipeline)

        # Any grid search parameter to be entered here.
        #full = Pipeline([ ("pre", preprocessing), ("lr", clf) ])

        #param_grid = [{'lr__hidden_layer_sizes':[(10,10,),(100,), 500],
        #              'lr__activation':['relu', 'identity'],
        #              'lr__solver':['lbfgs', 'adam'],
                     # 'lr__alpha':[0.00001],
        #               'lr__learning_rate':['adaptive', 'constant'],
        #               'lr__max_iter':[10000],
        #               'lr__tol':[1e-4]}]
        #grid_search = GridSearchCV(full, param_grid, cv=3, scoring='neg_root_mean_squared_error')

        #grid_search.fit(X_train, y_train)

        #print(grid_search.best_params_)
        #return grid_search, X_train, X_test, y_train, y_test, x, y

        clf = make_pipeline(preprocessing, clf)
        clf.fit(X_train, y_train)
        return clf, X_train, X_test, y_train, y_test, x, y

def getSvcRBF(df):
         hf = pd.read_csv(Path("heart_failure_clinical_records_dataset.csv"))
         #hf["creatine_per_sodium"] = hf["serum_creatinine"] / hf["serum_sodium"]
         #hf["age_cat"] = pd.cut(hf["age"],
         #                            bins=[0.,20.,30.,35.,40.,45.,50.,55.,60.,65.,70.,75.,80.,85.,90.,np.inf],
         #                            labels=[1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15])
         #hf = hf.drop(columns=['age'])
         y = hf["DEATH_EVENT"]
         x = hf.drop(columns=['DEATH_EVENT'])
         X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.33, random_state=42)
         clf = SVC(gamma=1, C=0.025, random_state=42, probability=True)

         default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
         min_max_pipeline = make_pipeline(SimpleImputer(strategy="median"), MinMaxScaler())
         log_pipeline = make_pipeline(SimpleImputer(strategy="median"),
                                       FunctionTransformer(np.log, feature_names_out="one-to-one"),
                                       MinMaxScaler(feature_range=(-1,1)))

         preprocessing = ColumnTransformer([#("log", log_pipeline, ["creatinine_phosphokinase", "platelets",                     "serum_creatinine", "serum_sodium"]       #[2,   6,  7, 8]Intercep[3,7,8,9],
                                             # ),
                                             # ("minmax", min_max_pipeline, [12]),
                                             ], remainder=default_num_pipeline)

         # Any grid search parameter to be entered here.

         #clf = make_pipeline(preprocessing, clf)
         clf.fit(X_train, y_train)
         return clf, X_train, X_test, y_train, y_test, x, y

def getNaiveBayes(df):
        hf = pd.read_csv(Path("heart_failure_clinical_records_dataset.csv"))
        hf["creatine_per_sodium"] = hf["serum_creatinine"] / hf["serum_sodium"]
        hf["age_cat"] = pd.cut(hf["age"],
                                    bins=[0.,20.,30.,35.,40.,45.,50.,55.,60.,65.,70.,75.,80.,85.,90.,np.inf],
                                    labels=[1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15])
        hf = hf.drop(columns=['age'])
        y = hf["DEATH_EVENT"]
        x = hf.drop(columns=['DEATH_EVENT'])
        X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.33, random_state=42)
        clf = GaussianNB()

        default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
        min_max_pipeline = make_pipeline(SimpleImputer(strategy="median"), MinMaxScaler())
        log_pipeline = make_pipeline(SimpleImputer(strategy="median"),
                                      FunctionTransformer(np.log, feature_names_out="one-to-one"),
                                      MinMaxScaler(feature_range=(-1,1)))

        preprocessing = ColumnTransformer([#("log", log_pipeline, ["creatinine_phosphokinase", "platelets",                     "serum_creatinine", "serum_sodium"] #[2,   6,  7, 8]Intercep[3,7,8,9],
                                            # ),
                                             ("minmax", min_max_pipeline, [12]),
                                            ], remainder=default_num_pipeline)

        # Any grid search parameter to be entered here.

        clf = make_pipeline(preprocessing, clf)
        clf.fit(X_train, y_train)
        return clf, X_train, X_test, y_train, y_test, x, y

def getDecisionTree(df):
         hf = pd.read_csv(Path("heart_failure_clinical_records_dataset.csv"))
         y = hf["DEATH_EVENT"]
         x = hf.drop(columns=['DEATH_EVENT'])
         X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.33, random_state=42)

         clf = DecisionTreeClassifier(max_depth=5, random_state=42)

         default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
         min_max_pipeline = make_pipeline(SimpleImputer(strategy="median"), MinMaxScaler())

         log_pipeline = make_pipeline(SimpleImputer(strategy="median"),
                                       FunctionTransformer(np.log, feature_names_out="one-to-one"),
                                       MinMaxScaler(feature_range=(-1,1)))

         preprocessing = ColumnTransformer([], remainder=default_num_pipeline)


         clf = make_pipeline(preprocessing, clf)
         clf.fit(X_train, y_train)

         return clf, X_train, X_test, y_train, y_test, x, y

def getQDA(df):
         hf = pd.read_csv(Path("heart_failure_clinical_records_dataset.csv"))
         hf["creatine_per_sodium"] = hf["serum_creatinine"] / hf["serum_sodium"]
         hf["age_cat"] = pd.cut(hf["age"],
                                     bins=[0.,20.,30.,35.,40.,45.,50.,55.,60.,65.,70.,75.,80.,85.,90.,np.inf],
                                     labels=[1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15])
         hf = hf.drop(columns=['age'])
         y = hf["DEATH_EVENT"]
         x = hf.drop(columns=['DEATH_EVENT'])
         X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.33, random_state=42)
         clf = QuadraticDiscriminantAnalysis()

         default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
         min_max_pipeline = make_pipeline(SimpleImputer(strategy="median"), MinMaxScaler())
         log_pipeline = make_pipeline(SimpleImputer(strategy="median"),
                                       FunctionTransformer(np.log, feature_names_out="one-to-one"),
                                       MinMaxScaler(feature_range=(-1,1)))

         preprocessing = ColumnTransformer([#("log", log_pipeline, ["creatinine_phosphokinase",                                 "platelets",                     "serum_creatinine", "serum_sodium"] #[2,   6,  7, 8]Intercep[3,7,8,9],
                                             # ),
                                              ("minmax", min_max_pipeline, [12]),
                                             ], remainder=default_num_pipeline)

         # Any grid search parameter to be entered here.

         clf = make_pipeline(preprocessing, clf)
         clf.fit(X_train, y_train)
         return clf, X_train, X_test, y_train, y_test, x, y

def getKNN(df):
         hf = pd.read_csv(Path("heart_failure_clinical_records_dataset.csv"))
         hf["creatine_per_sodium"] = hf["serum_creatinine"] / hf["serum_sodium"]
         hf["age_cat"] = pd.cut(hf["age"],
                                     bins=[0.,20.,30.,35.,40.,45.,50.,55.,60.,65.,70.,75.,80.,85.,90.,np.inf],
                                     labels=[1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15])
         hf = hf.drop(columns=['age'])
         y = hf["DEATH_EVENT"]
         x = hf.drop(columns=['DEATH_EVENT'])
         X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.33, random_state=42)
         clf = KNeighborsClassifier(3)

         default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
         min_max_pipeline = make_pipeline(SimpleImputer(strategy="median"), MinMaxScaler())
         log_pipeline = make_pipeline(SimpleImputer(strategy="median"),
                                       FunctionTransformer(np.log, feature_names_out="one-to-one"),
                                       MinMaxScaler(feature_range=(-1,1)))

         preprocessing = ColumnTransformer([#("log", log_pipeline, ["creatinine_phosphokinase",                                 "platelets",                     "serum_creatinine", "serum_sodium"] #[2,   6,  7, 8]Intercep[3,7,8,9],
                                             # ),
                                              ("minmax", min_max_pipeline, [12]),
                                             ], remainder=default_num_pipeline)

         # Any grid search parameter to be entered here.

         clf = make_pipeline(preprocessing, clf)
         clf.fit(X_train, y_train)
         return clf, X_train, X_test, y_train, y_test, x, y

def getGDA(df):
          hf = pd.read_csv(Path("heart_failure_clinical_records_dataset.csv"))
       #   hf["creatine_per_sodium"] = hf["serum_creatinine"] / hf["serum_sodium"]
        #  hf["age_cat"] = pd.cut(hf["age"],
        #                              bins=[0.,20.,30.,35.,40.,45.,50.,55.,60.,65.,70.,75.,80.,85.,90.,np.inf],
        #                              labels=[1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15])
        #  hf = hf.drop(columns=['age'])
          y = hf["DEATH_EVENT"]
          x = hf.drop(columns=['DEATH_EVENT'])
          X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.33, random_state=42)
          clf = GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42)

          default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
          min_max_pipeline = make_pipeline(SimpleImputer(strategy="median"), MinMaxScaler())
          log_pipeline = make_pipeline(SimpleImputer(strategy="median"),
                                        FunctionTransformer(np.log, feature_names_out="one-to-one"),
                                        MinMaxScaler(feature_range=(-1,1)))

          preprocessing = ColumnTransformer([#("log", log_pipeline,                                                             ["creatinine_phosphokinase",                                 "platelets",                     "serum_creatinine",              "serum_sodium"] #[2,   6,  7, 8]Intercep[3,7,8,9],
                                               ("minmax", min_max_pipeline, [12]),
                                              ], remainder=default_num_pipeline)

          # Any grid search parameter to be entered here.

        #  clf = make_pipeline(preprocessing, clf)
          clf.fit(X_train, y_train)
          return clf, X_train, X_test, y_train, y_test, x, y

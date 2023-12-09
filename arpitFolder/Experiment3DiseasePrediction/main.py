import util_pipeline
import util_metrics

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_recall_curve
import pandas as pd
from pathlib import Path

def main():
     dic, j = {}, 0
     hf = pd.read_csv(Path("heart_statlog_cleveland_hungary_final.csv"))
     models = util_pipeline.getClassifiers()
     for clfFn, name in models:
          print("=====",'\n', name, " ")

          clf, X_train, X_test, y_train, y_test, x, y = clfFn(hf.copy())
          util_metrics.evaluate_print_metrics(clf, name, X_train, X_test, y_train, y_test, x, y, j)
          y_probas = cross_val_predict(clf, X_train, y_train, cv=3,
                                       method="predict_proba")[:, 1]
          dic[name] = precision_recall_curve(y_train, y_probas)
          j += 1
     util_metrics.plot_pr_curve(dic)

if __name__ == "__main__":
     main()

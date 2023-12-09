import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import numpy as np
from sklearn.cluster import KMeans


import matplotlib.pyplot as plt

plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

# Loads the dataset
def loadDataSet():
      hf = pd.read_csv(Path("heart_failure_clinical_records_dataset.csv"))
      hf.head()
      #hf = hf[["serum_creatinine", "ejection_fraction","DEATH_EVENT"]]
      #hf["creatine_per_sodium"] = hf["serum_creatinine"] / hf["serum_sodium"]

      #hf["age_cat"] = pd.cut(hf["age"],
      #                           bins=[0.,20.,30.,35.,40.,45.,50.,55.,60.,65.,70.,  75.,80.,85.,90.,np.inf],
      #                           labels=[1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15])
      #hf = hf.drop(columns=['age'])




      y = hf["DEATH_EVENT"]
      x = hf.drop(columns=['DEATH_EVENT'])
      X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.33,       random_state=42)

      return X_train, X_test, y_train, y_test, x, y

 X_train, X_test, y_train, y_test, x, y = loadDataSet()

def getReducedDimesion(x, percentage):
   pca = PCA(n_components=percentage)
   x_red = pca.fit_transform(x)
   return x_red, pca

x_red, out = getReducedDimesion(x,0.99998)
 #print(out.n_components_)

def plot_variance_curve(x):
   pca = PCA()
   pca.fit(x)
   cumsum = np.cumsum(pca.explained_variance_ratio_)
   d = np.argmax(cumsum <= 0.95) + 1
   plt.figure(figsize=(6,4))
   plt.plot(cumsum, linewidth=3)
   plt.axis([0,12,0,2])
   plt.xlabel("Dimesions")
   plt.ylabel("Explained Variance")
   plt.plot([d,d], [0, 0.95], "k:")
   plt.plot([0,d],[0.95,0.95], "k:")
   plt.plot(d, 0.95, "ko")
   plt.annotate("Elbow", xy=(65, 0.85), xytext=(70, 0.7),                            arrowprops=dict(arrowstyle="->"))
   plt.grid(True)
   plt.show()

 #plot_variance_curve(x)



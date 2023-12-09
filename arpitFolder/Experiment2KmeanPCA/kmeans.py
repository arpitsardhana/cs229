import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer, MinMaxScaler
from util_pipeline import *


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
     #                           bins=[0.,20.,30.,35.,40.,45.,50.,55.,60.,65.,70.,75.,80.,85.,90.,np.inf],
     #                           labels=[1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15])
     #hf = hf.drop(columns=['age'])




     y = hf["DEATH_EVENT"]
     x = hf.drop(columns=['DEATH_EVENT'])
     X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.33, random_state=42)

     return X_train, X_test, y_train, y_test, x, y

X_train, X_test, y_train, y_test, x, y = loadDataSet()
selected_dimensions = ['serum_creatinine', 'ejection_fraction']
selected_data = x[selected_dimensions]
kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
kmeans.fit(selected_data)
df = x
df['cluster'] = kmeans.labels_

plt.scatter(df['serum_creatinine'], df['ejection_fraction'], c=df['cluster'], cmap='viridis', edgecolor='k')
#plt.title('KMeans Clustering')
#plt.xlabel('Serum Creatinine')
#plt.ylabel('Ejection Fraction')
#plt.show()

# Fitting with model
clf = LogisticRegression(penalty='l2', solver='liblinear', random_state=42)
default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
min_max_pipeline = make_pipeline(SimpleImputer(strategy="median"), MinMaxScaler())
log_pipeline = make_pipeline(SimpleImputer(strategy="median"),FunctionTransformer(np.log, feature_names_out="one-to-one"),MinMaxScaler(feature_range=(-1,1)))
preprocessing = ColumnTransformer([("log", log_pipeline, ["serum_creatinine"])], remainder=default_num_pipeline)
clf = make_pipeline(preprocessing, clf)
clf.fit(selected_data,y)
logreg = clf
#plt.show()

### Generated graph
xx, yy = np.meshgrid(np.linspace(selected_data['serum_creatinine'].min(), selected_data['serum_creatinine'].max(), 100),
                     np.linspace(selected_data['ejection_fraction'].min(), selected_data['ejection_fraction'].max(), 100))
aa = xx
bb = yy
val = xx.shape
print(xx)
print(yy)
#plt.show()
inputArr = np.c_[xx.ravel(), yy.ravel()]
dfArr = pd.DataFrame(inputArr, columns=['serum_creatinine', 'ejection_fraction'])
xx = dfArr['serum_creatinine']
yy = dfArr['ejection_fraction']

Z = clf.predict(dfArr)
Z = Z.reshape(val)
#print(Z)
print("input array shape: ", inputArr.shape, " xx array shape: ", val, " df array shape ", dfArr.shape, " shape of Z", Z.shape, "  type of Z", type(Z))
#Z_df = pd.DataFrame(Z, index=xx[:, 0], columns=yy[0, :])
#Z = Z.reshape(xx.shape)

plt.contour(aa, bb, Z, colors='r', linewidths=2, levels=[0.5], alpha=0.7)
plt.title('K means based profile prediction')
plt.xlabel('Serum Creatinine')
plt.ylabel('Ejection Fraction')
#plt.show()
plt.legend()
plt.savefig("Kmeans based profile prediction")


def kmeans(x):
  kmean = KMeans(n_clusters=2, n_init=10, random_state=42)
  y_pred = kmean.fit_predict(x)
  return kmean, y_pred


def plot_data(X,y):
    #plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)
    plt.plot(X[y==0, 0], X[y==0, 1], "yo", label="Non Fatal")
    plt.plot(X[y==1, 0], X[y==1, 1], "g^", label="Fatal")

def plot_centroids(centroids, weights=None, circle_color='w', cross_color='k'):
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='o', s=35, linewidths=8,
                color=circle_color, zorder=10, alpha=0.9)
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=2, linewidths=12,
                color=cross_color, zorder=11, alpha=1)

def plot_decision_boundaries(clusterer, X, y, resolution=1000, show_centroids=True,
                             show_xlabels=True, show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                cmap="Pastel2")
    plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                linewidths=1, colors='k')
    plot_data(X,y)
    if show_centroids:
        plot_centroids(clusterer.cluster_centers_)

    if show_xlabels:
        plt.xlabel("$x_1$")
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabels:
        plt.ylabel("$x_2$", rotation=0)
    else:
        plt.tick_params(labelleft=False)


#newx = x_red[y==1]
def plot_k_means(k,newx,y):
  kmeans = KMeans(n_clusters=k, n_init=20, random_state=42)
  y_pred = kmeans.fit_predict(newx)
  plt.figure(figsize=(8, 4))
  plot_decision_boundaries(kmeans, newx,y)
  plt.show()
#x_new = x[['serum_creatinine', 'ejection_fraction']]
#x_new.shape
#plot_k_means(2, x_new, y)
#plot_k_means(2, x_new, x_new[y==0])


#plot_k_means(4, x_red,y)
#plt.figure(figsize=(8, 4))
#xf = x[y==1]
#xnf = x[y==0]
#plt.plot(xf[0], xf[1], "yo", label="Non Fatal")
#plt.plot(xnf[0], xnf[1], "g^", label="Fatal")
plt.show()

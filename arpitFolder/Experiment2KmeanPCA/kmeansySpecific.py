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
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


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
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
kmeans.fit(selected_data)
df = selected_data.copy()
df['cluster'] = kmeans.labels_
#plt.scatter(df['serum_creatinine'], df['ejection_fraction'], c=df['cluster'], cmap='viridis', edgecolor='k')
markers = ['x', 's', '^']
colors = ['red', 'green', 'blue']
sizes = [100,50,100]
circle_color='w'
cross_color='k'
#print("aailable in kmenas", dir(kmeans))

for cluster, marker, size, centroids in zip(df['cluster'].unique(), markers, sizes, kmeans.cluster_centers_):
    cluster_data = df[df['cluster'] == cluster]
    cluster_data_y0 = cluster_data[y == 0]
    plt.scatter(cluster_data_y0['serum_creatinine'], cluster_data_y0['ejection_fraction'], s=size, marker=marker, color='green', label=f'Cluster {cluster} y=0')
    cluster_data_y1 = cluster_data[y == 1]
    plt.scatter(cluster_data_y1['serum_creatinine'], cluster_data_y1['ejection_fraction'], s=size, marker=marker, color='red', label=f'Cluster {cluster} y=1')
    print("for cluster: ", cluster, " num non-fatal:", len(cluster_data_y0), "  num fatal", len(cluster_data_y1))
    plt.scatter(centroids[0], centroids[1], marker='H', s=35, linewidths=12, color='black', zorder=11, alpha=1)
    #plt.scatter(centroids[0], centroids[1], marker=marker, s=2, linewidths=12, color=cross_color, zorder=11, alpha=1)

print("centroids: ", kmeans.cluster_centers_)
####
#selected_data_y0 = selected_data[y == 0]
#selected_data_y1 = selected_data[y == 1]

#kmeans0 = KMeans(n_clusters=2, n_init=10, random_state=42)
#kmeans1 = KMeans(n_clusters=2, n_init=10, random_state=42)
#kmeans0.fit(selected_data_y0)
#kmeans1.fit(selected_data_y1)
#df0 = selected_data_y0
#df1 = selected_data_y1
#df0['cluster'] = kmeans0.labels_
#df1['cluster'] = kmeans1.labels_

#plt.scatter(df0['serum_creatinine'], df0['ejection_fraction'], c=df0['cluster'], cmap='viridis', edgecolor='k')
#plt.scatter(df1['serum_creatinine'], df1['ejection_fraction'], c=df1['cluster'], cmap='plasma', edgecolor='r')
#plt.title('KMeans Clustering')
#plt.xlabel('Serum Creatinine')
#plt.ylabel('Ejection Fraction')
#plt.show()

#####
# Fitting with model
clf = LogisticRegression(penalty='l2', class_weight='balanced', solver='liblinear', random_state=42)

#RandomForest = RandomForestClassifier(max_depth=5, n_estimators=11, max_features=2, random_state=42)
#clf = AdaBoostClassifier(estimator=RandomForest, n_estimators=40, random_state=42)

default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
min_max_pipeline = make_pipeline(SimpleImputer(strategy="median"), MinMaxScaler())
log_pipeline = make_pipeline(SimpleImputer(strategy="median"),FunctionTransformer(np.log, feature_names_out="one-to-one"),MinMaxScaler(feature_range=(-1,1)))
preprocessing = ColumnTransformer([("log", log_pipeline, ["serum_creatinine"])], remainder=default_num_pipeline)
clf = make_pipeline(preprocessing, clf)
df = pd.read_csv(Path("heart_failure_clinical_records_dataset.csv"))
df = df.sample(frac = 1)
y1 = df["DEATH_EVENT"]
x1 = df.drop(columns=['DEATH_EVENT'])
selected_data1 = x1[selected_dimensions]
clf.fit(selected_data1,y1)
y_pred = clf.predict(x1)
y_test = y1
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
print('Confusion Matrix:')
print(conf_matrix)


logreg = clf
#plt.show()

### Generated graph
xx, yy = np.meshgrid(np.linspace(selected_data['serum_creatinine'].min(), selected_data['serum_creatinine'].max(), 100),
                     np.linspace(selected_data['ejection_fraction'].min(), selected_data['ejection_fraction'].max(), 100))
aa = xx
bb = yy
val = xx.shape
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
plt.title('Kmeans based profile prediction')
plt.xlabel('Serum Creatinine')
plt.ylabel('Ejection Fraction')
plt.legend(loc="upper right")
plt.savefig("kmeans_based_profile.png")


def kmeans(x):
  kmean = KMeans(n_clusters=2, n_init=10, random_state=42)
  y_pred = kmean.fit_predict(x)
  return kmean, y_pred


def plot_data(X,y):
    #plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)
    print("Num: non-fatal is: "(X[y==0, 0].shape))
    print("Num: fatal is: "(X[y==1, 0].shape))
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

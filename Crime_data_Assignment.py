# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 04:21:15 2021

@author: Amartya
"""
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

crime=pd.read_csv("F:\Softwares\Data Science Assignments\Python-Assignment\H-Clustering\\crime_data.csv")

def norm(i):
    x=((i-i.min())/(i.max()-i.min()))
    return(x)

data=norm(crime.iloc[:,1:])
data.describe()

dendogram=sch.dendrogram(sch.linkage(data,method='complete'))
clust= AgglomerativeClustering(n_clusters=6,linkage='complete',affinity='euclidean').fit(data)
clust.labels_

cl=pd.Series(clust.labels_)
crime['H_Clusters']=cl
crime.H_Clusters.value_counts()

##Kmeans

twss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,random_state=0)
    kmeans.fit(data)
    twss.append(kmeans.inertia_)
    
plt.plot(range(1,11),twss)
plt.title("Elbow Curve or Scree Plot")
plt.xlabel("No. of Clusters")
plt.ylabel("Total Within Sum of Squares")

kmeans=KMeans(n_clusters=5,random_state=0)
pred_y=kmeans.fit_predict(data)

clst=pd.DataFrame(pred_y)
crime['K_Clusters']=clst
crime.K_Clusters.value_counts()

crime.groupby("K_Clusters").mean()


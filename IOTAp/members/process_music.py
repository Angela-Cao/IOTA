import numpy as np
import pandas as pd
from io import StringIO
from transformers import pipeline
import matplotlib.pyplot as plt
import numpy as np
import pickle
import skfda
from skfda import datasets
from skfda.exploratory.visualization.clustering import (
    ClusterMembershipLinesPlot,
    ClusterMembershipPlot,
    ClusterPlot,
)
from skfda.ml.clustering import FuzzyCMeans, KMeans


#scales valance values [-1,1]->[0,1]
def scale(val):
  return((val+1)/2)

#set up
path = "./data/valence.csv"
df_valence = pd.read_csv("./data/valence.csv",encoding='Latin-1')
df_arousal = pd.read_csv("./data/arousal.csv",encoding='Latin-1')

#count non-missing values
valence_value = df_valence.iloc[0][1:]
count = np.count_nonzero(~np.isnan(valence_value))
count

#apply scale function
valence_value =  scale(valence_value)
df_valence.iloc[:, 1:(count+1)] = scale(df_valence.iloc[:, 1:(count+1)].values)
df_arousal.iloc[:, 1:(count+1)] = scale(df_arousal.iloc[:, 1:(count+1)].values)

#creates list of time values
time = []
for i in range(count):
  value = 0.5*(i+1)
  time.append(value)
print(time)



k = 0
# Create a subplot with two plots side by side
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].plot(time,scale(df_valence.iloc[k][1:count+1]))
axs[0].set_title('Valence')
axs[1].plot(time,scale(df_arousal.iloc[k][1:count+1]))
axs[1].set_title('Arousal')


data_matrix = df_arousal.iloc[:, 1:(count+1)].values



def kMeans_music(df, n_clusters, seed):
  data_matrix = df
  fd_music = skfda.FDataGrid(
    data_matrix=data_matrix,
    grid_points=time,
  )
  kmeans = KMeans(n_clusters=n_clusters, random_state=seed)    
  kmeans.fit(fd_music)
#  fd_music[kmeans.labels_==2]
  return(kmeans)

def kMeans_valence(n_clusters, seed):
  valence_data = df_valence.iloc[:, 0:(count+1)]
  pickle.dump(valence_data, open("./data/valence_data.pkl", "wb"))
  return(kMeans_music(df_valence.iloc[:, 1:(count+1)].values, n_clusters, seed))

def kMeans_arousal(n_clusters, seed):
  arousal_data = df_arousal.iloc[:, 0:(count+1)]
  pickle.dump(arousal_data, open("./data/arousal_data.pkl", "wb"))
  return(kMeans_music(df_arousal.iloc[:, 1:(count+1)].values, n_clusters, seed))

n_clusters = 4
seed = 2
kmeans_valence = kMeans_valence(n_clusters, seed)
kmeans_arousal = kMeans_arousal(n_clusters, seed)

#kmeans.fit(fd_music[1:100])
#kmeans.fit(fd_music)

#print(kmeans.predict(fd_music[11]))
pickle.dump(kmeans_valence, open("./models/kmeans_valence.pkl", "wb"))
pickle.dump(kmeans_arousal, open("./models/kmeans_arousal.pkl", "wb"))






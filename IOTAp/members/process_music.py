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



# def scale(val):
#   return((val+1)/2)


path = "./data/valence.csv"
df_valence = pd.read_csv("./data/valence.csv",encoding='Latin-1')
df_arousal = pd.read_csv("./data/arousal.csv",encoding='Latin-1')
#df_valence.iloc[0][1:].plot()
#plt.show()

valence_value = df_valence.iloc[0][1:]
count = np.count_nonzero(~np.isnan(valence_value))
count

# valence_value =  scale(valence_value)
# df_valence.iloc[:, 1:(count+1)] = scale(df_valence.iloc[:, 1:(count+1)].values)
# df_arousal.iloc[:, 1:(count+1)] = scale(df_arousal.iloc[:, 1:(count+1)].values)

time = []
for i in range(count):
  value = 0.5*(i+1)
  time.append(value)
print(time)



# k = 0
# # Create a subplot with two plots side by side
# fig, axs = plt.subplots(1, 2, figsize=(10, 5))
# axs[0].plot(time,scale(df_valence.iloc[k][1:count+1]))
# axs[0].set_title('Valence')
# axs[1].plot(time,scale(df_arousal.iloc[k][1:count+1]))
# axs[1].set_title('Arousal')


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
  pickle.dump(valence_data, open("./data/valence_data_unscaled.pkl", "wb"))
  return(kMeans_music(df_valence.iloc[:, 1:(count+1)].values, n_clusters, seed))

def kMeans_arousal(n_clusters, seed):
  arousal_data = df_arousal.iloc[:, 0:(count+1)]
  pickle.dump(arousal_data, open("./data/arousal_data_unscaled.pkl", "wb"))
  return(kMeans_music(df_arousal.iloc[:, 1:(count+1)].values, n_clusters, seed))

n_clusters = 3
seed = 2
kmeans_valence = kMeans_valence(n_clusters, seed)
kmeans_arousal = kMeans_arousal(n_clusters, seed)



#kmeans.fit(fd_music[1:100])
# kmeans_valence.fit(valence_data)

#print(kmeans.predict(fd_music[11]))
# pickle.dump(kmeans_valence, open("./models/kmeans_valence.pkl", "wb"))
# pickle.dump(kmeans_arousal, open("./models/kmeans_arousal.pkl", "wb"))



# import matplotlib.pyplot as plt
# import skfda
# from skfda.datasets import fetch_growth
# from skfda.exploratory.visualization.fpca import FPCAPlot
# from skfda.preprocessing.dim_reduction import FPCA
# from skfda.representation.basis import (
#     BSplineBasis,
#     FourierBasis,
#     MonomialBasis,
# )


# fd_music = skfda.FDataGrid(
#     data_matrix=data_matrix,
#     grid_points=time,
# )

# basis_fd = fd_music.to_basis(BSplineBasis(n_basis=5))
# fpca = FPCA(n_components=2, components_basis=FourierBasis(n_basis=5))
# fpca.fit(basis_fd)
# fpca.components_.plot()


# scores = fpca.transform(basis_fd)
# plt.figure()
# plt.scatter(scores[:, 0], scores[:, 1])

# fpca.components_

# basis_fd

# fd_music[kmeans_pcscore.labels_==3]


# kmeans_pcscore = KMeans(init="random", n_clusters=3, n_init=10, max_iter=300, random_state=42)
# kmeans_pcscore.fit(scores)

# kmeans_pcscore.labels_










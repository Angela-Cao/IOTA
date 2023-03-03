# import numpy as np
# import pandas as pd
# from io import StringIO
# from transformers import pipeline
# import matplotlib.pyplot as plt
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


# path = "./data/valence.csv"
# df_valence = pd.read_csv("./data/valence.csv",encoding='Latin-1')
# df_arousal = pd.read_csv("./data/arousal.csv",encoding='Latin-1')
# #df_valence.iloc[0][1:].plot()
# #plt.show()

# valence_value = df_valence.iloc[0][1:]
# count = np.count_nonzero(~np.isnan(valence_value))
# count

# valence_value =  scale(valence_value)
# df_valence.iloc[:, 1:(count+1)] = scale(df_valence.iloc[:, 1:(count+1)].values)
# df_arousal.iloc[:, 1:(count+1)] = scale(df_arousal.iloc[:, 1:(count+1)].values)

count = 60
time = []
for i in range(count):
  value = 0.5*(i+1)
  time.append(value)
print(time)



# # k = 0
# # # Create a subplot with two plots side by side
# # fig, axs = plt.subplots(1, 2, figsize=(10, 5))
# # axs[0].plot(time,scale(df_valence.iloc[k][1:count+1]))
# # axs[0].set_title('Valence')
# # axs[1].plot(time,scale(df_arousal.iloc[k][1:count+1]))
# # axs[1].set_title('Arousal')


# data_matrix = df_arousal.iloc[:, 1:(count+1)].values

# def kMeans_music(df, n_clusters, seed):
#   data_matrix = df
#   fd_music = skfda.FDataGrid(
#     data_matrix=data_matrix,
#     grid_points=time,
#   )
#   kmeans = KMeans(n_clusters=n_clusters, random_state=seed)    
#   kmeans.fit(fd_music)
# #  fd_music[kmeans.labels_==2]
#   return(kmeans)

# def kMeans_valence(n_clusters, seed):
#   valence_data = df_valence.iloc[:, 0:(count+1)]
#   # pickle.dump(valence_data, open("./data/valence_data.pkl", "wb"))
#   return(kMeans_music(df_valence.iloc[:, 1:(count+1)].values, n_clusters, seed))

# def kMeans_arousal(n_clusters, seed):
#   arousal_data = df_arousal.iloc[:, 0:(count+1)]
#   # pickle.dump(arousal_data, open("./data/arousal_data.pkl", "wb"))
#   return(kMeans_music(df_arousal.iloc[:, 1:(count+1)].values, n_clusters, seed))

# n_clusters = 3
# seed = 2
# kmeans_valence = kMeans_valence(n_clusters, seed)
# kmeans_arousal = kMeans_arousal(n_clusters, seed)



#kmeans.fit(fd_music[1:100])
# kmeans_valence.fit(valence_data)

#print(kmeans.predict(fd_music[11]))
# pickle.dump(kmeans_valence, open("./models/kmeans_valence.pkl", "wb"))
# pickle.dump(kmeans_arousal, open("./models/kmeans_arousal.pkl", "wb"))



import matplotlib.pyplot as plt
import skfda
from skfda.datasets import fetch_growth
from skfda.exploratory.visualization.fpca import FPCAPlot
from skfda.preprocessing.dim_reduction import FPCA
from skfda.representation.basis import (
    BSplineBasis,
    FourierBasis,
    MonomialBasis,
)

def time_grids_long():
  count = 120
  time = []
  for i in range(count):
    value = 0.5*(i+1)
    time.append(value)
  return(time)


valence_data = pickle.load(open("./data/valence_data.pkl", "rb"))
arousal_data = pickle.load(open("./data/arousal_data.pkl", "rb"))

fd_valence = skfda.FDataGrid(
    data_matrix=valence_data.iloc[:, 1:(count+1)],
    grid_points=time,
)

fd_arousal = skfda.FDataGrid(
    data_matrix=arousal_data.iloc[:, 1:(count+1)],
    grid_points=time,
)


fd_valence_d = fd_valence.derivative()
fd_arousal_d = fd_arousal.derivative()


# fd =fd_valence.concatenate(fd_arousal)

fd = skfda.FDataGrid(
  data_matrix =  np.concatenate((valence_data.iloc[:, 1:(count+1)], arousal_data.iloc[:, 1:(count+1)]),axis=1),    
  grid_points=time_grids_long(),
)



n_components = 3

def get_fpca_score(fd_music):
  basis_fd = fd_music.to_basis(BSplineBasis(n_basis=5))
  fpca = FPCA(n_components=n_components, components_basis=FourierBasis(n_basis=8))
  fpca.fit(basis_fd)
  # fpca.components_.plot()
  scores = fpca.transform(basis_fd)
  return(scores)

# fdd = fd.derivative()

score_valence = get_fpca_score(fd_valence)
score_arousal = get_fpca_score(fd_arousal)

score_valence_d = get_fpca_score(fd_valence_d)
score_arousal_d = get_fpca_score(fd_arousal_d)

print(type(score_valence))

raw_scores = np.concatenate((score_valence, score_arousal),axis=1)

# raw_scores = np.concatenate((score_valence, score_arousal, score_valence_d, score_arousal_d),axis=1)

# raw_scores =  score_arousal

# print(raw_scores)

# print(raw_scores.shape)

def mm_normalize(data):
  (nrows, ncols) = data.shape  # (20,4)
  mins = np.zeros(shape=(ncols), dtype=np.float32)
  maxs = np.zeros(shape=(ncols), dtype=np.float32)
    
  for i in range(0,nrows):
    for j in range(0,ncols):
      mins[j] = np.min([data[i][j], mins[j]])
      maxs[j] = np.max([data[i][j], maxs[j]])

  result = np.copy(data)
  for i in range(0,nrows):
    for j in range(0,ncols):
      result[i][j] = (data[i][j] - mins[j]) / (maxs[j] - mins[j])
  return (result, mins, maxs)

(scores, mins, maxs) = mm_normalize(raw_scores)
print(scores)

# plt.figure()
# plt.scatter(scores[:, 0], scores[:, 1])

# print(fpca.components_)

# print(basis_fd)

# from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# fd_music[kmeans_pcscore.labels_==3]
n_clusters = 6
kmeans_pcscore = KMeans(init="random", n_clusters=n_clusters, n_init=10, max_iter=1000, random_state=102)
kmeans_pcscore.fit(scores)
print(kmeans_pcscore.labels_)


# ClusterPlot(kmeans, fd_music[kmeans_pcscore.labels_==2]).plot()

# fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# fd_valence[kmeans_pcscore.labels_==cluster_index].plot()
# fd_arousal[kmeans_pcscore.labels_==cluster_index].plot()

# cluster_index = 0


import pandas as pd
#set up
nrc_set = pd.read_csv (r'./data/NRC-VAD-Lexicon.csv')
nrc_set['Valence'] = (nrc_set['Valence']-0.5)*2
nrc_set['Arousal'] = (nrc_set['Arousal']-0.5)*2


# Convert two columns to a dictionary
d = nrc_set.set_index('Word').T.to_dict('list')

words = ['tense', 'distressed', 'frustrated', 'depressed', 'sad', 'miserable', 'sad', 'gloomy', 'afraid', 
         'alarmed', 'angry', 'annoyed', 
         'bored', 'tired', 'drowsy', 'sleepy',
         'aroused', 'excited', 'astonished', 'delighted', 'glad', 'pleased', 'happy', 'satisfied', 
         'content', 'relaxed', 'tranquil', 'ease', 'calm']

valence = []
arousal = []
for word in words:  
  temp = d[word]
  valence.append(temp[0])
  arousal.append(temp[1])


def show_cluster(cluster_index):
  # fig, axs = plt.subplots(1, 2, figsize=(10, 5))

  mean_valence = skfda.exploratory.stats.mean(fd_valence[kmeans_pcscore.labels_==cluster_index])  
  fig1 = mean_valence.plot()

  mean_arousal = skfda.exploratory.stats.mean(fd_arousal[kmeans_pcscore.labels_==cluster_index])
  mean_arousal.plot(fig1)  
  plt.xlabel('Time')  
  # plt.ylabel('Valence/Arousal')  
  plt.legend(['Valence', 'Arousal'])
  plt.savefig('Cluster'+str(cluster_index)+'.pdf')  
  # plt.show()  
  # return(fig)



for i in range(0,n_clusters):
  show_cluster(i)

# Create a scatter plot
  
  # plt.plot(valence, arousal,'bo')
  # v_vals = df_valence.data_matrix[0,:,:]
  # a_vals = df_arousal.data_matrix[0,:,:]
  # plt.plot(v_vals, a_vals, 'red', linewidth='4') 

  # # plt.arrow(v_vals[len(v_vals)-2], a_vals[len(a_vals)-2],v_vals[len(v_vals)-1]-v_vals[len(v_vals)-2], a_vals[len(a_vals)-1]-a_vals[len(a_vals)-2],  width=0.2)

  # # Set x and y limits centered around 0
  # plt.xlim(-1.2, 1.2)
  # plt.ylim(-1.2, 1.2)

  # # Add a horizontal and vertical line at 0
  # plt.axhline(y=0, color='k')
  # plt.axvline(x=0, color='k')

  # # Add labels for x and y axes
  # plt.xlabel('Valence')
  # plt.ylabel('Arousal')
  # # # Add text labels for each point
  # for i  in range(0,len(words)):
  #   plt.text(valence[i]-0.1, arousal[i]+0.03, words[i],fontsize=12, color='black', fontweight='bold')  




def show_cluster_2dim(cluster_index):
  mean_valence = skfda.exploratory.stats.mean(fd_valence[kmeans_pcscore.labels_==cluster_index])  
  # mean_valence = skfda.exploratory.stats.depth_based_median(fd_valence[kmeans_pcscore.labels_==cluster_index])  
  mean_arousal = skfda.exploratory.stats.mean(fd_arousal[kmeans_pcscore.labels_==cluster_index])
  # mean_arousal = skfda.exploratory.stats.depth_based_median(fd_valence[kmeans_pcscore.labels_==cluster_index])  

  # print(mean_valence.data_matrix[0])
  # print(mean_arousal.data_matrix[0])
  plt.plot(mean_valence.data_matrix[0], mean_arousal.data_matrix[0])
  plt.xlim([0.2, 0.8])
  plt.ylim([0.2, 0.8])  
  plt.xlabel('Valence')  
  plt.ylabel('Arousal')  
  # Set x and y limits centered around 0
  plt.xlim(-1.2, 1.2)
  plt.ylim(-1.2, 1.2)

  # Add a horizontal and vertical line at 0
  plt.axhline(y=0, color='k')
  plt.axvline(x=0, color='k')

  # Add labels for x and y axes
  plt.xlabel('Valence')
  plt.ylabel('Arousal')
  # # Add text labels for each point
  for i  in range(0,len(words)):
    plt.text(valence[i]-0.1, arousal[i]+0.03, words[i],fontsize=12, color='black', fontweight='bold')  

  # plt.legend(['Valence', 'Arousal'])
  plt.savefig('Cluster'+str(cluster_index)+'_2dim.pdf')  
  # plt.savefig('Cluster'+str(cluster_index)+'_2dim_median.pdf')  

  # plt.show()

# for i in range(0,n_clusters):
#   show_cluster_2dim(i)



def show_clusters(cluster_index):
  mean_valence = skfda.exploratory.stats.mean(fd_valence[kmeans_pcscore.labels_==cluster_index])  
  # mean_valence = skfda.exploratory.stats.depth_based_median(fd_valence[kmeans_pcscore.labels_==cluster_index])  
  mean_arousal = skfda.exploratory.stats.mean(fd_arousal[kmeans_pcscore.labels_==cluster_index])
  # mean_arousal = skfda.exploratory.stats.depth_based_median(fd_valence[kmeans_pcscore.labels_==cluster_index])  

  # fig, axs = plt.subplots(1, 2, figsize=(10, 5))

  # plt.plot(valence, arousal,'bo', linewidth=0.5)
  plt.plot(mean_valence.data_matrix[0], mean_arousal.data_matrix[0], linewidth='2')
  plt.xlabel('Valence')  
  plt.ylabel('Arousal')  
  # Set x and y limits centered around 0
  plt.xlim(-1, 1)
  plt.ylim(-1, 1)

  # Add a horizontal and vertical line at 0
  plt.axhline(y=0, color='k')
  plt.axvline(x=0, color='k')

  # Add labels for x and y axes
  plt.xlabel('Valence')
  plt.ylabel('Arousal')
  # # Add text labels for each point
  for i  in range(0,len(words)):
    plt.text(valence[i], arousal[i], words[i],fontsize=6, color='black')  

  # plt.legend(['Valence', 'Arousal'])
  # plt.savefig('Cluster'+str(cluster_index)+'_2dim.pdf')  
  # plt.savefig('Cluster'+str(cluster_index)+'_2dim_median.pdf')  


fig, axs = plt.subplots(1, 1, figsize=(6, 6))
for i in range(0,n_clusters):
  show_clusters(i)
plt.savefig('Clusters_2dim.pdf')  


# from sklearn.model_selection import train_test_split
# import skfda
# from skfda.misc.metrics import PairwiseMetric, linf_distance
# from skfda.ml.classification import RadiusNeighborsClassifier

# X_train, X_test, y_train, y_test = train_test_split(
#     # fd_valence,
#     # fd_arousal,    
#     fd,
#     kmeans_pcscore.labels_,
#     test_size=0.25,
#     shuffle=True,
#     stratify=kmeans_pcscore.labels_,
#     random_state=0,
# )


# radius = 0.5
# sample = X_test[0]  # Center of the ball

# # fig = X_train.plot(group=y_train, group_colors=['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'])

# # Plot ball
# # sample.plot(fig=fig, color='red', linewidth=3)
# # lower = sample - radius
# # upper = sample + radius
# # fig.axes[0].fill_between(
# #     sample.grid_points[0],
# #     lower.data_matrix.flatten(),
# #     upper.data_matrix[0].flatten(),
# #     alpha=0.25,
# #     color='C1',
# # )
# # plt.show()


# # Creation of pairwise distance
# l_inf = PairwiseMetric(linf_distance)
# distances = l_inf(sample, X_train)[0]  # L_inf distances to 'sample'

# # # Plot samples in the ball
# # fig = X_train[distances <= radius].plot(color='C0')
# # sample.plot(fig=fig, color='red', linewidth=3)
# # fig.axes[0].fill_between(
# #     sample.grid_points[0],
# #     lower.data_matrix.flatten(),
# #     upper.data_matrix[0].flatten(),
# #     alpha=0.25,
# #     color='C1',
# # )

# radius_nn = RadiusNeighborsClassifier(radius=radius, weights='distance')
# radius_nn.fit(X_train, y_train)


# pred = radius_nn.predict(X_test)
# # print(pred)


# test_score = radius_nn.score(X_test, y_test)
# print(test_score)




# print(mean_valence[[0]][0])

# print(mean_valence.data_matrix[0])



# axs[0].plot(time, mean_valence)

# mean_arousal = skfda.exploratory.stats.mean(fd_arousal[kmeans_pcscore.labels_==cluster_index])
# axs[1].plot(time, mean_arousal)


# fd_valence.plot(group=kmeans_pcscore.labels_, group_colors=['red', 'blue', 'green', 'yellow', 'black', 'orange', 'lightgreen', 'lightblue'])
# fd_arousal.plot(group=kmeans_pcscore.labels_, group_colors=['red', 'blue', 'green', 'yellow', 'black', 'orange', 'lightgreen', 'lightblue'])
# plt.show()


# fd_valence.plot(group=kmeans_pcscore.labels_, group_colors=['red', 'blue', 'green', 'yellow', 'black', 'orange'])
# fd_arousal.plot(group=kmeans_pcscore.labels_, group_colors=['red', 'blue', 'green', 'yellow', 'black', 'orange'])
# plt.show()


# fd_valence.plot(group=kmeans_pcscore.labels_, group_colors=['red', 'blue', 'green', 'yellow', 'black'])
# fd_arousal.plot(group=kmeans_pcscore.labels_, group_colors=['red', 'blue', 'green', 'yellow', 'black'])


# x = np.linspace(0, 20, 100)
# plt.plot(x, np.sin(x))






# kmeans_pcscore = KMeans(init="random", n_clusters=3, n_init=10, max_iter=300, random_state=42)
# kmeans_pcscore.fit(scores)

# kmeans_pcscore.labels_



# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.model_selection import GridSearchCV, train_test_split

# import skfda
# from skfda.ml.classification import KNeighborsClassifier


# knn = KNeighborsClassifier(n_neighbors=6)
# knn.fit(X_train, y_train)

# pred = knn.predict(X_test)
# # print(pred)

# score = knn.score(X_test, y_test)
# print(score)



import numpy as np
import pandas as pd
from io import StringIO
import math
import skfda
import pickle

from skfda.misc.hat_matrix import (
    KNeighborsHatMatrix,
    LocalLinearRegressionHatMatrix,
    NadarayaWatsonHatMatrix,
)
from skfda.misc.kernels import uniform
from skfda.preprocessing.smoothing import KernelSmoother
from skfda.preprocessing.smoothing.validation import SmoothingParameterSearch

#NRC VAD set up
nrc_set = pd.read_csv (r'./data/NRC-VAD-Lexicon.csv')

# Convert two columns to a dictionary
d = nrc_set.set_index('Word').T.to_dict('list')

#returns 1st item of "word" in dictionar "d"
def v(word):
  return(d[word][0])


#creates list "windows" to store resulting windows
def window_moving(lst, window_size, step_size):
    windows = []
    for i in range(0, len(lst), step_size):
        end = i + window_size
        if end > len(lst):
            break
        windows.append(lst[i:end])
    return windows

#computes sum of valance values igw
def window_valence(window):
  step_size = 2
  w_size = len(window)
  sum = 0  
  k = 0
  for i in range(0,w_size):
    if window[i] in d.keys():
        sum = sum+v(window[i])
        k = k+1
  #returns avg valance igw
  return(sum/max(1,k))



#computes sum of arousal values igw
def window_arousal(window):
  step_size = 2
  w_size = len(window)
  sum = 0  
  k = 0 
  for i in range(0,w_size):
    if window[i] in d.keys():
        sum = sum+d[window[i]][1]
        k = k+1 
  #returns avg arousal igw
  return(sum/max(1,k))

#finds max valance igw
def window_valence_2(window):
  max = 0
  for i in range(0,len(window)):
    if window[i] in d.keys():
      v_val = v(window[i])
      if(v_val >max):
        max = v_val        
  return(max)

#returns "time" list, "count" values as 0.5 intervals
def time_grids():
  count = 60
  time = []
  for i in range(count):
    value = 0.5*(i+1)
    time.append(value)
  return(time)

#computes avg valance&arousal of given text through moving windows, returns df
def text2va(text):   
  word = text.split(" ")
  windows = window_moving(word, 7, 2)
  print(len(windows))
  valences = []
  arousal = []
  for i in range(0,len(windows)):
    valences.append(window_valence(windows[i]))
    arousal.append(window_arousal(windows[i]))
  df = pd.DataFrame({'valence': valences, 'arousal': arousal})
  print(df)
  return(df)

#smooth v values in df w/ kernel smoother
def smooth_valence(df):
  time = time_grids()
  count = len(time)
  my_list = df['valence'][0:count]
  # my_list = [float('nan') if x == 0 else x for x in text_valence]
  nan_indices = [i for i, x in enumerate(my_list) if isinstance(x, float) and math.isnan(x)]
  # print(nan_indices)
  # my_time = time
  # for i in nan_indices:
  #   my_time[i] = None
  my_list = [my_list[i] for i in range(len(my_list)) if i not in nan_indices]
  my_time = [time[i] for i in range(len(time)) if i not in nan_indices]
  grid_points = my_time
  data_matrix = [
    my_list
  ]
  fd_text = skfda.FDataGrid(
    data_matrix=data_matrix,
    grid_points=grid_points,
  )

  nw = KernelSmoother(
    kernel_estimator=NadarayaWatsonHatMatrix(bandwidth=2),
    output_points = time[0:len(time)],
  ) 
  nw.fit(fd_text)
  return(nw.transform(fd_text))

#smooth a values in df w/ kernel smoother
def smooth_arousal(df):
  time = time_grids()
  count = len(time)
  my_list = df['arousal'][0:count]
  nan_indices = [i for i, x in enumerate(my_list) if isinstance(x, float) and math.isnan(x)]
  my_list = [my_list[i] for i in range(len(my_list)) if i not in nan_indices]
  my_time = [time[i] for i in range(len(time)) if i not in nan_indices]
  grid_points = my_time
  data_matrix = [
    my_list
  ]
  fd_text = skfda.FDataGrid(
    data_matrix=data_matrix,
    grid_points=grid_points,
  )

  nw = KernelSmoother(
    kernel_estimator=NadarayaWatsonHatMatrix(bandwidth=2),
    output_points = time[0:len(time)],
  ) 
  nw.fit(fd_text)
  return(nw.transform(fd_text))

#calculates Euclidean distance of a&b
def dist(a,b):
  distance = np.linalg.norm(a - b)
  return(distance)

#convert text to v&a, smooth values, calculates distance
def match_music(text):
  df_va = text2va(text)
  df_valence = smooth_valence(df_va)
  df_arousal = smooth_arousal(df_va)
  df_valence = df_valence.data_matrix[0][0]
  df_arousal = df_arousal.data_matrix[0][0]
  kmeans_valence = pickle.load(open("./models/kmeans_valence.pkl", 'rb'))
  kmeans_arousal = pickle.load(open("./models/kmeans_arousal.pkl", 'rb'))            
  valence_data = pickle.load(open("./data/valence_data.pkl", "rb"))
  arousal_data = pickle.load(open("./data/arousal_data.pkl", "rb"))
  num_rows = valence_data.shape[0]
  min = 100000 
  k = -1
  for i in range(0, num_rows):
    valence_value = valence_data.iloc[i][1:61]
    dist_valence = dist(df_valence,valence_value)
    arousal_value = arousal_data.iloc[i][1:61]
    dist_arousal = dist(df_arousal,arousal_value)
    dist_current = dist_valence+dist_arousal 
    if(dist_current<min):
      min = dist_current
      k = i
  return(arousal_data.iloc[k][0])


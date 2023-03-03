import numpy as np
import pandas as pd
from io import StringIO
import math
import skfda
import pickle
import matplotlib
import matplotlib.pyplot as plt
from skfda.misc.metrics import PairwiseMetric, linf_distance


from skfda.misc.hat_matrix import (
    KNeighborsHatMatrix,
    LocalLinearRegressionHatMatrix,
    NadarayaWatsonHatMatrix,
)
from skfda.misc.kernels import uniform
from skfda.preprocessing.smoothing import KernelSmoother
from skfda.preprocessing.smoothing.validation import SmoothingParameterSearch
from transformers import BertTokenizer


#set up
nrc_set = pd.read_csv (r'./data/NRC-VAD-Lexicon.csv')
nrc_set['Valence'] = (nrc_set['Valence']-0.5)*2
nrc_set['Arousal'] = (nrc_set['Arousal']-0.5)*2





# Convert two columns to a dictionary
d = nrc_set.set_index('Word').T.to_dict('list')



def v(word):
  return(d[word][0])



def time_grids():
  count = 60
  time = []
  for i in range(count):
    value = 0.5*(i+1)
    time.append(value)
  return(time)


def text2va(text):   
  tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
  words = tokenizer.tokenize(text)
  valences = []
  arousal = []
  count = 0
  for i in range(0,len(words)):
      word = words[i]
      if word in d.keys():
        valences.append(d[word][0])
        arousal.append(d[word][1])  
        count = count+1
      else:
        valences.append(None)
        arousal.append(None)  
  df = pd.DataFrame({'words':words, 'valence': valences, 'arousal': arousal})
  print(df)
  return(df,count)



def smooth_text_va(text_va):

  bandwidth = 30
  
  # time = time_grids()
  # count = len(time)
  my_list = text_va
  text_len = len(my_list)
  time = range(0,text_len)
  nan_indices = [i for i, x in enumerate(my_list) if isinstance(x, float) and math.isnan(x)]
  my_list = [my_list[i] for i in range(text_len) if i not in nan_indices]
  my_time = [time[i] for i in range(text_len) if i not in nan_indices]
  grid_points = my_time
  data_matrix = [
    my_list
  ]
  fd_text = skfda.FDataGrid(
    data_matrix=data_matrix,
    grid_points=grid_points,
  )

  output_points = range(np.min(my_time), np.max(my_time))
  nw = KernelSmoother(
    kernel_estimator=NadarayaWatsonHatMatrix(bandwidth=bandwidth),
    output_points =  output_points   
  ) 
  nw.fit(fd_text)
  return(nw.transform(fd_text))




def match_music(text):
  df_va = text2va(text)
  print(len(df_va))
  text_l = len(df_va)
  if text_l<60:
    text_len = text_l
  else:
    text_len = 60
  df_valence = smooth_valence(df_va[0:text_len])
  df_arousal = smooth_arousal(df_va[0:text_len])
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
    valence_value = valence_data.iloc[i][1:(text_len+1)]
    dist_valence = dist(df_valence,valence_value)
    arousal_value = arousal_data.iloc[i][1:(text_len+1)]
    dist_arousal = dist(df_arousal,arousal_value)
    dist_current = dist_valence+dist_arousal 
    if(dist_current<min):
      min = dist_current
      k = i
  return(i, arousal_data.iloc[k][0])


def smooth_valence(df):  
  return(smooth_text_va(df['valence']))

def smooth_arousal(df):
  return(smooth_text_va(df['arousal']))  
 
def dist(a,b):
  distance = np.linalg.norm(a - b)
  return(distance)


# l1_distance: Final = LpDistance(p=1)
# l2_distance: Final = LpDistance(p=2)
# linf_distance: Final = LpDistance(p=math.inf)


# def dist(a,b):
#   l_inf = PairwiseMetric(linf_distance)
#   distance = l_inf(a, b)[0]  
#   return(distance)


def plot_va(df_va):
  time = time_grids()
  count = len(time)  
  # text_l = len(df_va)
  # if text_l<60:
  #   text_len = text_l
  # else:
  #   text_len = 60
  # df_valence = smooth_valence(df_va[0:text_len])
  # df_arousal = smooth_arousal(df_va[0:text_len])
  # Create a subplot with two plots side by side
  fig, axs = plt.subplots(1, 2, figsize=(10, 5))
  axs[0].plot(time,df_va["valence"][0:count])
  axs[0].set_title('Valence')
  axs[1].plot(time,df_va["arousal"][0:count])
  axs[1].set_title('Arousal')
  return(fig)


def plot_music_va(i):
  time = time_grids()
  count = len(time)  
  text_len = 60
  valence_data = pickle.load(open("./data/valence_data.pkl", "rb"))
  arousal_data = pickle.load(open("./data/arousal_data.pkl", "rb"))
  num_rows = valence_data.shape[0]
  valence_value = valence_data.iloc[i][1:(text_len+1)]
  arousal_value = arousal_data.iloc[i][1:(text_len+1)]
  fig, axs = plt.subplots(1, 2, figsize=(10, 5))
  axs[0].plot(time, valence_value)
  axs[0].set_title('Valence')
  axs[1].plot(time, arousal_value)
  axs[1].set_title('Arousal')
  return(fig)



def plot_valence_arousal(df_va):
  nrow = len(df_va['words'])  
  print(nrow)
  time = range(0,nrow)
  df_valence = smooth_valence(df_va)  
  df_arousal = smooth_arousal(df_va)  
  fig, axs = plt.subplots(1, 2, figsize=(10, 5))
  axs[0].plot(time,df_va["valence"][0:nrow],'bo')
  axs[0].plot(df_valence.grid_points[0], df_valence.data_matrix[0,:,:], linewidth='4', color='red')
  axs[0].set_title('Valence')
  axs[0].set_xlabel('Word Index')
  axs[0].set_ylabel('Valence')
  axs[1].plot(time,df_va["arousal"][0:nrow],'bo')
  axs[1].plot(df_arousal.grid_points[0],df_arousal.data_matrix[0,:,:], linewidth='4', color='red')
  axs[1].set_title('Arousal')
  axs[1].set_xlabel('Word Index')
  axs[1].set_ylabel('Arousal')
  return(fig)


def plot_smooth_va(df_va):
  output_points_v, df_valence = smooth_valence(df_va)
  output_points_a, df_arousal = smooth_arousal(df_va)  
  fig, axs = plt.subplots(1, 2, figsize=(10, 5))
  axs[0].plot(output_points_v,df_valence,'bo')
  axs[0].set_title('Valence')
  axs[1].plot(output_points_a,df_arousal,'bo')
  axs[1].set_title('Arousal')
  # plt.show()
  return(fig)


# print(df_valence.data_matrix[0][0])



# file_id = match_music("today is a snowing day. We stay at home working and studying")
# print(file_id)

# # file_name = "./data/the_happy_family.txt"
# file_name = "./data/story_3pigs.txt"
# with open(file_name, 'r') as file:
#   text = file.read()
#   df_va,count = text2va(text)
#   # plot_va(df_va)  
#   print(count)
#   df_valence = smooth_valence(df_va)  
#   print(df_valence)
#   # print(df_valence.data_matrix[0][0])
#   # df_valence.plot()
#   # df_arousal = smooth_arousal(df_va)  
#   # df_arousal.plot()
#   plot_valence_arousal(df_va)
#   plt.show()



  


#   plt.show()
  # print(output_points_v)
  # print(df_valence)
  # plot_valence_arousal(df_va)
  # plot_smooth_va(df_va)

#   print(df_va)

#   file_id = match_music(text)
#   print(file_id)
# print(file_id)




# from playsound import playsound
# # Play an MP3 file
# #playsound('./data/MEMD_audio/'+str(round(arousal_data.iloc[k][0]))+'.mp3')
# playsound('./data/MEMD_audio/'+str(round(file_id))+'.mp3')



# from aip import AipSpeech


# import ffmpeg
# import sys
# import os

# """ 你的 APPID AK SK """
# APP_ID = '24528497'
# API_KEY = '377orEuzgLGooAo3VaCYdeYk'
# SECRET_KEY = 'Y35c7d7UwCPGCCcNCBSHfRPpHcw2cdYw'

# client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)

# def sentence2audio(sentence,filename):
#     result  = client.synthesis(sentence, 'En', 1, {
#         'vol': 8,
#         'per':0,
#         'pit':4,
#         'speed':7,
#     })
    
#     # 识别正确返回语音二进制 错误则返回dict 参照下面错误码
#     if not isinstance(result, dict):
#         with open(filename, 'wb') as f:
#             f.write(result)


# sentence = 'Along time ago, there lived an old poet, a thoroughly kind old poet. As he was sitting one evening in his room, a dreadful storm arose without, and the rain streamed down from heaven; but the old poet sat warm and comfortable in his chimney-corner, where the fire blazed and the roasting apple hissed.'
# filename = './data/temp/1.mp3'
# sentence2audio(sentence,filename)

# # file_name = "./data/the_happy_family.txt"
# # with open(file_name, 'r') as file:
# #    text=file.read()
# #    sentence2audio(text,filename)
   






   

























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


def window_moving(lst, window_size, step_size):
    windows = []
    for i in range(0, len(lst), step_size):
        end = i + window_size
        if end > len(lst):
            break
        windows.append(lst[i:end])
    return windows



def window_valence(window):
  w_size = len(window)
  sum = 0  
  k = 0
  for i in range(0,w_size):
    if window[i] in d.keys():
        sum = sum+d[window[i]][0]
        k = k+1
#  return(step_size*sum/w_size)
  return(sum/max(1,k))




def window_arousal(window):
  w_size = len(window)
  sum = 0  
  k = 0 
  for i in range(0,w_size):
    if window[i] in d.keys():
        sum = sum+d[window[i]][1]
        k = k+1 
#  return(step_size*sum/w_size)
  return(sum/max(1,k))


def window_valence_2(window):
  max = 0
  for i in range(0,len(window)):
    if window[i] in d.keys():
      v_val = d[window[i]][0]
      if(np.abs(v_val) >max):
        max = v_val        
  if max ==0:
    max = None      
  return(max)

def window_arousal_2(window):
  max = 0
  for i in range(0,len(window)):
    if window[i] in d.keys():
      v_val = d[window[i]][1]
      if(np.abs(v_val) >max):
        max = v_val
  if max ==0:
    max = None      
  return(max)


def time_grids():
  count = 60
  time = []
  for i in range(count):
    value = 0.5*(i+1)
    time.append(value)
  return(time)

def our_tokenizer(text):
  # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
  # words = tokenizer.tokenize(text)
  text0 = text.replace('?'," ").replace('.', " ").replace(',', " ").replace("!"," ")
  # words = text.split()   
  return(text0.split(' '))



def text2va(words):   
  # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
  # words = tokenizer.tokenize(text)
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
  # print(df)
  return(df,count)


def text2va_window_smoothing(words):     
  windows = window_moving(words, 8, 2)
  count = len(windows)
  valences = []
  arousal = []
  for i in range(0,len(windows)):
    # print(windows[i])
    valences.append(window_valence_2(windows[i]))
    arousal.append(window_arousal_2(windows[i]))
  df = pd.DataFrame({'valence': valences, 'arousal': arousal})
  # print(df)
  return(df,count)



def smooth_text_va(text_va):

  bandwidth = 5
  
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

valence_data = pickle.load(open("./data/valence_data.pkl", "rb"))
arousal_data = pickle.load(open("./data/arousal_data.pkl", "rb"))
smoothed_valence_data = pickle.load(open("./data/smoothed_valence_data.pkl", "rb"))
smoothed_arousal_data = pickle.load(open("./data/smoothed_arousal_data.pkl", "rb"))


def match_music(df_va):
  # df_va,count = text2va(text)
  # df_va,count = text2va_window_smoothing(text)
  # print(len(df_va))
  df_valence = smooth_valence(df_va)
  df_arousal = smooth_arousal(df_va)
  df_valence = df_valence.data_matrix[:,:,0]
  df_arousal = df_arousal.data_matrix[:,:,0]

  text_l = len(df_arousal[0])
  # print(text_l)
  if text_l<60:
    text_len = text_l
  else:
    text_len = 60
  
  # print("df_arousal is" )
  # print(df_arousal[0][0:text_len])

  # kmeans_valence = pickle.load(open("./models/kmeans_valence.pkl", 'rb'))
  # kmeans_arousal = pickle.load(open("./models/kmeans_arousal.pkl", 'rb'))            
  num_rows = valence_data.shape[0]
  min = 100000 
  k = -1
  for i in range(0, num_rows):
    valence_value = smoothed_valence_data.data_matrix[i,:,:]
    # valence_data.iloc[i][1:(text_len+1)]
    dist_valence = dist(df_valence[0][0:text_len],valence_value)
    arousal_value = smoothed_arousal_data.data_matrix[i,:,:]
    # arousal_data.iloc[i][1:(text_len+1)]
    dist_arousal = dist(df_arousal[0][0:text_len],arousal_value)
    dist_current = dist_valence+dist_arousal     
    # print(dist_current)
    # print(df_valence[0][0:text_len])
    # print(valence_value)
    if(dist_current<min):
      min = dist_current
      k = i
  return(k, arousal_data.iloc[k][0])


def smooth_valence(df):  
  return(smooth_text_va(df['valence']))

def smooth_arousal(df):
  return(smooth_text_va(df['arousal']))  
 
def dist(a,b):
  # print(pd.DataFrame({'a':a, 'b':b, 'd':a-b}))
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
  fig, axs = plt.subplots(2, 1, figsize=(10, 5))
  axs[0].plot(time,df_va["valence"][0:count])
  axs[0].set_title('Valence')
  axs[1].plot(time,df_va["arousal"][0:count])
  axs[1].set_title('Arousal')
  return(fig)


def plot_music_va(i):
  time = time_grids()
  # count = len(time)  
  text_len = 60

  num_rows = valence_data.shape[0]
  valence_value = valence_data.iloc[i][1:(text_len+1)]
  arousal_value = arousal_data.iloc[i][1:(text_len+1)]
  smoothed_valence_value = smoothed_valence_data.data_matrix[i,:,:]
  smoothed_arousal_value = smoothed_arousal_data.data_matrix[i,:,:]

  fig, axs = plt.subplots(2, 1, figsize=(10, 5))
  axs[0].plot(time, valence_value, linewidth='4', color='blue')
  axs[0].plot(time, smoothed_valence_value, linewidth='4', color='red')
  axs[0].set_ylabel('Valence')
  axs[1].plot(time, arousal_value, linewidth='4', color='blue')
  axs[1].plot(time, smoothed_arousal_value, linewidth='4', color='red')
  axs[1].set_ylabel('Arousal')
  axs[1].set_xlabel('Time')
  plt.savefig('va_music_emotion.png', dpi=300) 
  return(fig)



def plot_valence_arousal(df_va):
  nrow = len(df_va['valence'])  
  time = range(0,nrow)
  df_valence = smooth_valence(df_va)  
  df_arousal = smooth_arousal(df_va)  
  fig, axs = plt.subplots(2, 1, figsize=(10, 5))
  axs[0].plot(time,df_va["valence"][0:nrow],'bo')
  axs[0].plot(df_valence.grid_points[0], df_valence.data_matrix[0,:,:], linewidth='4', color='red')
  # axs[0].set_title('Valence')
  # axs[0].set_xlabel('Word Index')
  axs[0].set_ylabel('Valence')
  axs[1].plot(time,df_va["arousal"][0:nrow],'bo')
  axs[1].plot(df_arousal.grid_points[0],df_arousal.data_matrix[0,:,:], linewidth='4', color='red')
  # axs[1].set_title('Arousal')
  axs[1].set_xlabel('Word Index')
  axs[1].set_ylabel('Arousal')
  plt.savefig('va_text_timevarying.png', dpi=300) 
  return(fig)


def plot_smooth_va(df_va):
  output_points_v, df_valence = smooth_valence(df_va)
  output_points_a, df_arousal = smooth_arousal(df_va)  
  fig, axs = plt.subplots(2, 1, figsize=(10, 5))
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

#   plt.show()
  # print(output_points_v)
  # print(df_valence)
  # plot_valence_arousal(df_va)
  # plot_smooth_va(df_va)

#   print(df_va)

#   file_id = match_music(text)
#   print(file_id)
# print(file_id)

# file_name = "./data/the_happy_family.txt"
# with open(file_name, 'r') as file:
#   text = file.read()
#   df_va,count = text2va(text)
#   plot_va(df_va)
#   print(df_va)

#   file_id = match_music(text)
#   print(file_id)
# print(file_id)



# from playsound import playsound
# # Play an MP3 file
# #playsound('./data/MEMD_audio/'+str(round(arousal_data.iloc[k][0]))+'.mp3')
# playsound('./data/MEMD_audio/'+str(round(file_id))+'.mp3')


# sentence = 'Along time ago, there lived an old poet, a thoroughly kind old poet. As he was sitting one evening in his room, a dreadful storm arose without, and the rain streamed down from heaven; but the old poet sat warm and comfortable in his chimney-corner, where the fire blazed and the roasting apple hissed.'
# filename = './data/temp/1.mp3'
# sentence2audio(sentence,filename)

# # file_name = "./data/the_happy_family.txt"
# # with open(file_name, 'r') as file:
# #    text=file.read()
# #    sentence2audio(text,filename)
   





words = [
  # 'tense', 
        'distressed',
        'frustrated', 
        'depressed', 
        'sad', 
        'miserable',
        'gloomy', 
        'afraid', 
        'alarmed', 
        'angry', 
        'annoyed', 
        'bored', 
        'tired', 
        'drowsy', 
        'sleepy',
        'aroused', 
        'excited', 
        # 'astonished',
        'astonish',  
        'delighted', 
        'glad', 
        'pleased', 
        'happy', 
        'satisfied', 
        'content', 
        'relaxed', 
        'tranquil', 
#        'ease', 
         'easy', 
        'calm',
        'concern',
        'desire',
        'empathy',
        'remorse',
        # 'sorrow',
        'thrill',
        'rage',
        'warmth',
        # 'love',
        'joy',
        'passion'
        ]

valence = []
arousal = []
for word in words:  
  temp = d[word]
  valence.append(temp[0])
  arousal.append(temp[1])


# print(valence)
# print(arousal)
  
# df = pd.DataFrame({'words':words, 'valence':valence, 'arousal':arousal})
# print(df)

# # print(d.keys())

# # Create a scatter plot
# plt.scatter(valence, arousal)

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
#     plt.text(valence[i]-0.1, arousal[i]+0.03, words[i],fontsize=12, color='black', fontweight='bold')  
# # Show the plot
# # plt.show()



def plot_va_with_words(df_va):
  nrow = len(df_va['valence'])  
  time = range(0,nrow)
  df_valence = smooth_valence(df_va)  
  df_arousal = smooth_arousal(df_va)  

  # Create a scatter plot
  fig, axs = plt.subplots(1, 1, figsize=(10, 8))
  plt.plot(valence, arousal,'bo')
  v_vals = df_valence.data_matrix[0,:,:]
  a_vals = df_arousal.data_matrix[0,:,:]
  plt.plot(v_vals, a_vals, 'red', linewidth='4') 

  # plt.arrow(v_vals[len(v_vals)-2], a_vals[len(a_vals)-2],v_vals[len(v_vals)-1]-v_vals[len(v_vals)-2], a_vals[len(a_vals)-1]-a_vals[len(a_vals)-2],  width=0.2)

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
    if words[i]=='tranquil':
      plt.text(valence[i]+0.01, arousal[i]+0.02, words[i],fontsize=12, color='black')  #, fontweight='bold'
    else:
      plt.text(valence[i]-0.1, arousal[i]+0.03, words[i],fontsize=12, color='black')  #, fontweight='bold'
  plt.savefig('va_text_emotion.png', dpi=300)    
  return(fig)
  


def generate_va():
   # Create a scatter plot
  fig, axs = plt.subplots(1, 1, figsize=(10, 8))
  plt.plot(valence, arousal,'bo')
  plt.xlim(-1.2, 1.2)
  plt.ylim(-1.2, 1.2)
  # Add a horizontal and vertical line at 0
  plt.axhline(y=0, color='k')
  plt.axvline(x=0, color='k')
  # Add labels for x and y axes
  plt.xlabel('Valence', fontsize=18)
  plt.ylabel('Arousal', fontsize=18)
  # # Add text labels for each point
  for i  in range(0,len(words)):   
    if words[i]=='tranquil':
      plt.text(valence[i]+0.01, arousal[i]+0.02, words[i],fontsize=12, color='black', fontweight='bold')  #, fontweight='bold'
    else:
      plt.text(valence[i]-0.1, arousal[i]+0.03, words[i],fontsize=12, color='black', fontweight='bold')  #, fontweight='bold'
  # plt.savefig('va.pdf')
  plt.savefig('va.png', dpi=300)

generate_va()


# file_name = "./data/the_happy_family.txt"
# # file_name = "./data/story_3pigs.txt"
# file_name = "./data/shortStory.txt"
# file_name = "./data/100west.txt"
# file_name = "./data/thrillStory.txt"
# file_name = "./data/20131113.txt"



# file_name = "./data/wolflamb.txt"
# with open(file_name, 'r') as file:
#   text = file.read()
#   # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
#   # words = tokenizer.tokenize(text)
#   # # print(words)
#   # windows = window_moving(words, 8, 2)
#   # print(windows)
#   # # print(len(windows))
#   # valences = []
#   # arousal = []
#   # for i in range(0,len(windows)):
#   #   print(windows[i])
#   #   valences.append(window_valence(windows[i]))
#   #   arousal.append(window_arousal(windows[i]))
#   # df = pd.DataFrame({'valence': valences, 'arousal': arousal})
#   # print(df)
#   # df_va,count = text2va(text)
#   df_va,count = text2va_window_smoothing(text)


#   # print(count)
#   # df_valence = smooth_valence(df_va)  
#   # print(df_valence)
#   # # print(df_valence.data_matrix[0][0])
#   # df_valence.plot()
#   # df_arousal = smooth_arousal(df_va)  
#   # df_arousal.plot()
#   plot_va_with_words(df_va)
#   plt.show()














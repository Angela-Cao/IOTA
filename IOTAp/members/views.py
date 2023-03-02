from django.http import HttpResponse
from django.template import loader

def members(request):
  template = loader.get_template('myfirst.html')
  return HttpResponse(template.render())

from django.shortcuts import render

def second_page(request):
    return render(request, 'second_page.html')

def myfirst(request):
    return render(request, 'myfirst.html')


from django.shortcuts import render

def second_page(request):
    if request.method == 'POST':
        # If the user submitted the form, get the text from the request and render the template with the text
        text = request.POST['text']
        return render(request, 'second_page.html', {'text': text})
    else:
        # If the user hasn't submitted the form yet, render the template without any text
        return render(request, 'second_page.html')






from django.http import HttpResponse
from django.template import loader
from .forms import CustomerForm 
from .forms import TextForm 
from rest_framework import viewsets 
from rest_framework.decorators import api_view 
from django.core import serializers 
from rest_framework.response import Response 
from rest_framework import status 
from django.http import JsonResponse 
from rest_framework.parsers import JSONParser 
from .models import Customer 
from .serializer import CustomerSerializers 

import pickle
import json 
import numpy as np 
from sklearn import preprocessing 
import pandas as pd 
from django.shortcuts import render, redirect 
from django.contrib import messages 

from playsound import playsound

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io, base64
from django.db.models.functions import TruncDay
from matplotlib.ticker import LinearLocator


from .process_text import (
     text2va,
     smooth_valence,
     match_music,
     plot_valence_arousal,
)

from .process_music import (
     kMeans_music,
     kMeans_valence,
     kMeans_arousal,
     scale,
)

def members(request):
  template = loader.get_template('myfirst.html')
  return HttpResponse(template.render())

from django.shortcuts import render

def second_page(request):
    return render(request, 'second_page.html')

def myfirst(request):
    return render(request, 'myfirst.html')


class CustomerView(viewsets.ModelViewSet): 
    queryset = Customer.objects.all() 
    serializer_class = CustomerSerializers 

def status(df):
    try:
        scaler=pickle.load(open("/Users/HP-k/DeployML/DjangoAPI/Scaler.sav", 'rb'))
        model=pickle.load(open("/Users/HP-k/DeployML/DjangoAPI/Prediction.sav", 'rb'))
        X = scaler.transform(df) 
        y_pred = model.predict(X) 
        y_pred=(y_pred>0.80) 
        result = "Yes" if y_pred else "No"
        return result 
    except ValueError as e: 
        return Response(e.args[0], status.HTTP_400_BAD_REQUEST) 

def FormView(request):
    if request.method=='POST':
        form=CustomerForm(request.POST or None)
        if form.is_valid():
            Gender = form.cleaned_data['gender']
            Age = form.cleaned_data['age']
            EstimatedSalary = form.cleaned_data['salary']
            df=pd.DataFrame({'gender':[Gender], 'age':[Age], 'salary':[EstimatedSalary]})
            df["gender"] = 1 if "male" else 2
            result = status(df)
            return render(request, 'status.html', {"data": result}) 
            
    form=CustomerForm()
    return render(request, 'form.html', {'form':form})

def TextView(request):
    if request.method=='POST':
        form=TextForm(request.POST or None)
        if form.is_valid():
            Text = form.cleaned_data['text']
            df=pd.DataFrame({'text':[Text]})
            # file_index = match_music(Text)
            # audio_file = 'MEMD_audio/'+str(round(file_index))+'.mp3'
            audio_file = ""           

            df_va,count = text2va(Text)
            # fig = plot_va(df_av)
            fig = plot_valence_arousal(df_va)
            # fig_music = plot_music_va(i)

            # fig, ax = plt.subplots(figsize=(10,4))
            # ax.plot([0, 1, 3, 4, 5], [0, 1, 3, 4, 5], '--bo')
            flike = io.BytesIO()
            fig.savefig(flike)
            b64 = base64.b64encode(flike.getvalue()).decode()
            # playsound(audio_file)
            #     audio_file = 'MEMD_audio/2.mp3'                                           
            return render(request, 'status.html', {"data": audio_file, "Text":Text, "chart":b64}) 
    form=TextForm()
    return render(request, 'second_page.html', {'form':form})

def show_story(request):
    try:
        # Open the file in read-only mode
        file_name = './data/story_3pigs.txt'

        with open(file_name, 'r') as file:
        # Read the contents of the file            
            text = file.read()
            file_index = match_music('./data/story_3pigs.txt')
            # df_valence = smooth_valence(df_va)
            kmeans_valence = pickle.load(open("./models/kmeans_valence.pkl", 'rb'))
            kmeans_arousal = pickle.load(open("./models/kmeans_arousal.pkl", 'rb'))            
            # cluster_index = kmeans_valence.predict(df_valence)
            #kmeans.fit(fd_music)
            # print(df_va)
            audio_file_name = './data/MEMD_audio/'+str(round(file_index))+'.mp3'
            playsound(audio_file_name)            
            return render(request, 'show_story.html', {"data": text,  "audio_file_name":audio_file_name}) 
            # return render(request, 'show_story.html', {"data": df_va}) 
    except ValueError as e: 
        return Response(e.args[0], status.HTTP_400_BAD_REQUEST) 

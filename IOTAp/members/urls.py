from django.urls import path
from . import views

urlpatterns = [
    path('members/', views.members, name='members'),
    path('second_page/', views.second_page, name='second_page'),
    path('myfirst/', views.myfirst, name='myfirst'),
    path('FormView/', views.FormView, name='FormView'),
    path('TextView/', views.TextView, name='TextView'),    
    path('show_story/', views.show_story, name='show_story'), 
]


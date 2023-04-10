from django.urls import path
from . import views

urlpatterns = [
    path('members/', views.members, name='members'),
    path('second_page/', views.second_page, name='second_page'),
    path('index/', views.index, name='index'),
    path('FormView/', views.FormView, name='FormView'),
    path('TextView/', views.TextView, name='TextView'),    
    path('show_story/', views.show_story, name='show_story'), 
]


from django.urls import path
from . import views

urlpatterns = [
    path('home/', views.home, name='home'), 
    path('analysis/', views.analysis, name='analysis'), 
    path('dashboard1/', views.dashboard1, name='dashboard1'), 
    path('dashboard2/', views.dashboard2, name='dashboard2'), 
    path('ml/', views.ml, name='ml'), 
    path('preprocess/', views.preprocess, name='preprocess'), 
    path('sentiment/', views.sentiment, name='sentiment'), 
    path('suggestor/', views.suggestor, name='suggestor'), 
]
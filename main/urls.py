from . import views
from django.urls import path

app_name = "main"

urlpatterns = [
    path('', views.Home.as_view(), name="home"),
    path('upload/', views.upload, name="upload"),
]

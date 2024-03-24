from django.urls import path
from . import views

urlpatterns = [
    path("", views.login),
    path("register/",views.register,name="register"),
    path("login/",views.login,name="login"),
    path("user/",views.user,name="user"),
]
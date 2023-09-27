"""project URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.conf.urls import url
from app import views


urlpatterns = [
    path('admin/', admin.site.urls),
    url(r'^home$', views.home, name='startpage'),
    url(r'^breast_photo$', views.breast_photo_upload, name='breast_cancer'),
    url(r'^breast_cancer$', views.breast_cancer, name='breast_cancer2'),
    url(r'^lung_photo', views.lung_photo_upload, name='lung_cancer'),
    url(r'^lung_cancer$', views.lung_cancer, name='lung_cancer2'),
    url(r'^skin_photo$', views.skin_photo_upload, name='skin_cancer'),
    url(r'^skin_cancer$', views.skin_cancer, name='skin_cancer2'),
    url(r'^$', views.risk_analysis_data_submit, name='risk_analysis'),
    url(r'^risk_analysis$', views.risk_analysis, name='risk_analysis2'),

    url(r'user_login/$', views.user_login, name='user_login'),
    url(r'^register/$', views.register, name='register'),
    url(r'^logout/$', views.user_logout, name='logout'),

]

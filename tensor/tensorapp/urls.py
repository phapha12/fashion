from django.urls import include, path
from . import views

app_name = 'tensorapp'

urlpatterns = [
	path('', views.home, name='home'),
	path('predict', views.predict_form, name='predict_form')
]
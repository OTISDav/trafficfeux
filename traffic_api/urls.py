# traffic_light_ai_project/traffic_control_project/traffic_api/urls.py

from django.urls import path
from .views import PredictTrafficLightPhase # Importe la vue que nous avons définie

urlpatterns = [
    # Définition de l'URL pour l'endpoint de prédiction
    # Quand on accède à /api/predict/, c'est la vue PredictTrafficLightPhase qui est appelée.
    path('predict/', PredictTrafficLightPhase.as_view(), name='predict_phase'),
]
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .services import traffic_controller


class PredictTrafficLightPhase(APIView):
    #GET
    def get(self, request, *args, **kwargs):
        decision = traffic_controller.get_next_phase_decision()

        if "error" in decision:
            return Response(decision, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response(decision, status=status.HTTP_200_OK)

    #POST
    def post(self, request, *args, **kwargs):
        # 1. Validation des données d'entrée reçues via POST
        required_keys = [
            'x_A_direct', 'x_A_left',
            'x_B_direct', 'x_B_left',
            'x_C_direct', 'x_C_left',
            'x_D_direct', 'x_D_left'
        ]

        traffic_data_dict = request.data  # Les données POST sont dans request.data (parsed par DRF)

        if not all(key in traffic_data_dict for key in required_keys):
            return Response(
                {
                    "error": "Données de trafic incomplètes. Assurez-vous d'envoyer toutes les 8 valeurs (x_A_direct, x_A_left, etc.)."},
                status=status.HTTP_400_BAD_REQUEST
            )

        traffic_data_list = []
        try:
            for key in required_keys:
                value = traffic_data_dict[key]
                # Vérifie que la valeur est un nombre et est positive
                if not isinstance(value, (int, float)) or value < 0:
                    return Response(
                        {"error": f"La valeur pour '{key}' doit être un nombre positif (reçu: {value})."},
                        status=status.HTTP_400_BAD_REQUEST
                    )
                traffic_data_list.append(float(value))  # Convertit en float pour le modèle ML

        except Exception as e:
            return Response({"error": f"Erreur lors de la lecture des données: {str(e)}"},
                            status=status.HTTP_400_BAD_REQUEST)

        # 2. Appeler le contrôleur pour obtenir la prédiction manuelle
        decision = traffic_controller.get_manual_prediction(traffic_data_list)

        if "error" in decision:
            return Response(decision, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response(decision, status=status.HTTP_200_OK)
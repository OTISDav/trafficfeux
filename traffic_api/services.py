import torch
import os
import json
from ml_model.traffic_model import TrafficLightModel

class TrafficLightController:
    _instance = None  # Pour implémenter un Singleton (une seule instance de ce contrôleur)

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(TrafficLightController, cls).__new__(cls)
            cls._instance._initialize()  # Initialise le contrôleur une seule fois
        return cls._instance

    def _initialize(self):
        self.phase_definitions = {
            0: {"description": "Phase 0: Trafic Nord-Sud DIRECT", "default_duration": 45},
            1: {"description": "Phase 1: Trafic Est-Ouest DIRECT", "default_duration": 45},
            2: {"description": "Phase 2: Trafic Nord VIRAGE À GAUCHE (Protégé)", "default_duration": 20},
            3: {"description": "Phase 3: Trafic Sud VIRAGE À GAUCHE (Protégé)", "default_duration": 20},
            4: {"description": "Phase 4: Trafic Est VIRAGE À GAUCHE (Protégé)", "default_duration": 20},
            5: {"description": "Phase 5: Trafic Ouest VIRAGE À GAUCHE (Protégé)", "default_duration": 20},
        }

        # --- Chargement du Modèle ML Entraîné ---

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(project_root, 'ml_model', 'traffic_light_model.pth')

        input_size = 8
        num_phases = len(self.phase_definitions)

        self.model = TrafficLightModel(input_size, num_phases)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            self.model.eval()
            print(f"Modèle ML chargé avec succès depuis {model_path}")
        except FileNotFoundError:
            print(
                f"Erreur: Le fichier modèle {model_path} est introuvable. Assurez-vous d'avoir entraîné le modèle (exécutez 'python -m ml_model.train_model').")
            self.model = None
        except Exception as e:
            print(f"Erreur lors du chargement du modèle: {e}")
            self.model = None

        # --- Gestion des scénarios de démonstration ---
        self.simulation_scenarios = []
        self.current_scenario_index = 0

        # Chemin vers le fichier JSON des scénarios (à la racine du projet, dans le dossier data)
        scenarios_file_path = os.path.join(project_root, 'data', 'simulation_scenarios.json')
        self._load_simulation_scenarios(scenarios_file_path)

    def _load_simulation_scenarios(self, file_path):
        """Charge les scénarios de trafic depuis un fichier JSON."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.simulation_scenarios = json.load(f)
            print(f"Chargé {len(self.simulation_scenarios)} scénarios de simulation depuis {file_path}")
        except FileNotFoundError:
            print(
                f"Erreur: Fichier de scénarios de simulation '{file_path}' introuvable. Assurez-vous qu'il existe dans le dossier 'data'.")
            self.simulation_scenarios = []
        except json.JSONDecodeError:
            print(f"Erreur: Impossible de lire le fichier JSON '{file_path}'. Vérifiez son format.")
            self.simulation_scenarios = []

    # --- MÉTHODE GET : Pour la démonstration séquentielle des scénarios ---
    def get_next_phase_decision(self) -> dict:
        if self.model is None:
            return {"error": "Modèle ML non chargé. Impossible de prédire."}
        if not self.simulation_scenarios:
            return {
                "error": "Aucun scénario de simulation chargé. Veuillez configurer 'data/simulation_scenarios.json'."}

        current_scenario = self.simulation_scenarios[self.current_scenario_index]
        scenario_name = current_scenario["scenario_name"]
        traffic_data_dict = current_scenario["traffic_data"]

        # Convertit le dictionnaire de trafic en une liste ordonnée pour l'entrée du modèle
        traffic_data_list = [
            traffic_data_dict['x_A_direct'], traffic_data_dict['x_A_left'],
            traffic_data_dict['x_B_direct'], traffic_data_dict['x_B_left'],
            traffic_data_dict['x_C_direct'], traffic_data_dict['x_C_left'],
            traffic_data_dict['x_D_direct'], traffic_data_dict['x_D_left'],
        ]

        decision_result = self._predict_and_format_output(traffic_data_list)

        decision_result["scenario_name"] = scenario_name
        decision_result["input_traffic_data"] = traffic_data_dict

        # Avancer l'index pour le prochain appel (boucle si on arrive à la fin)
        self.current_scenario_index = (self.current_scenario_index + 1) % len(self.simulation_scenarios)
        decision_result["next_scenario_index"] = self.current_scenario_index

        return decision_result

    # --- MÉTHODE POST : Pour la saisie manuelle des données ---
    def get_manual_prediction(self, traffic_data_list: list) -> dict:
        if self.model is None:
            return {"error": "Modèle ML non chargé. Impossible de prédire."}

        return self._predict_and_format_output(traffic_data_list)

    # --- Méthode interne pour éviter la répétition de code de prédiction ---
    def _predict_and_format_output(self, traffic_data_list: list) -> dict:
        """
        Méthode interne pour exécuter la prédiction du modèle et formater la sortie.
        """
        # Convertit la liste Python en tenseur PyTorch de type float32
        input_tensor = torch.tensor([traffic_data_list], dtype=torch.float32)

        with torch.no_grad():  # Désactive le calcul des gradients pour la prédiction (plus rapide et moins de mémoire)
            outputs = self.model(input_tensor)
            # torch.argmax trouve l'indice de la valeur maximale (notre phase prédite)
            predicted_phase_id = torch.argmax(outputs, dim=1).item()

        # Récupère les informations détaillées de la phase prédite
        phase_info = self.phase_definitions.get(predicted_phase_id, {
            "description": "Phase inconnue",
            "default_duration": 30  # Durée par défaut si la phase n'est pas trouvée (ne devrait pas arriver)
        })

        return {
            # "predicted_phase_id": predicted_phase_id,
            "resultat_de_la_preiction": phase_info["description"],
            "temps_du_feu": phase_info["default_duration"]
        }



traffic_controller = TrafficLightController()

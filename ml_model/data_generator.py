import random
import torch
import os

def generate_traffic_data(num_samples=100000, max_people_direct=150, max_people_left=75, output_dir='data'):
    """
    Génère des données simulées pour la régulation de feux tricolores.
    Chaque échantillon contient 8 valeurs d'entrée (flux de trafic) et 1 label (ID de la phase optimale).
    """
    data = []

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Début de la génération de {num_samples} échantillons de données de trafic (8 entrées, 6 phases)...")

    for _ in range(num_samples):
        # Simuler les 8 flux d'entrée du trafic (Nord, Est, Sud, Ouest -> Direct et Virage Gauche)
        x_A_direct = random.randint(0, max_people_direct)
        x_A_left = random.randint(0, max_people_left)
        x_B_direct = random.randint(0, max_people_direct)
        x_B_left = random.randint(0, max_people_left)
        x_C_direct = random.randint(0, max_people_direct)
        x_C_left = random.randint(0, max_people_left)
        x_D_direct = random.randint(0, max_people_direct)
        x_D_left = random.randint(0, max_people_left)

        inputs = [
            x_A_direct, x_A_left,
            x_B_direct, x_B_left,
            x_C_direct, x_C_left,
            x_D_direct, x_D_left
        ]

        # Déterminer la phase qui a le plus de trafic
        scores = [
            (x_A_direct + x_C_direct),
            (x_B_direct + x_D_direct),
            x_A_left,
            x_C_left,
            x_B_left,
            x_D_left
        ]

        y = scores.index(max(scores)) # La phase avec le score le plus élevé est le label attendu

        data.append((inputs, y))

    features = torch.tensor([d[0] for d in data], dtype=torch.float32)
    labels = torch.tensor([d[1] for d in data], dtype=torch.long)

    torch.save(features, os.path.join(output_dir, 'features.pt'))
    torch.save(labels, os.path.join(output_dir, 'labels.pt'))

    print(f"Données générées et sauvegardées dans '{output_dir}/features.pt' et '{output_dir}/labels.pt'.")
    print(f"Aperçu des 5 premiers échantillons :")
    for i in range(min(5, num_samples)):
        print(f"  Entrées: {data[i][0]}, Phase attendue: {data[i][1]}")

if __name__ == "__main__":
    generate_traffic_data(num_samples=100000, max_people_direct=150, max_people_left=75)
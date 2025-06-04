import torch
import torch.nn as nn
import torch.optim as optim
import os
from ml_model.traffic_model import TrafficLightModel

def train_model(data_dir='data', model_save_path='ml_model/traffic_light_model.pth', num_epochs=12000, learning_rate=0.00005):
    """
    Entraîne le modèle de régulation de feux tricolores et le sauvegarde.
    """
    print(f"1. Chargement des données d'entraînement depuis '{data_dir}'...")
    try:
        features = torch.load(os.path.join(data_dir, 'features.pt'))
        labels = torch.load(os.path.join(data_dir, 'labels.pt'))
    except FileNotFoundError:
        print(f"Erreur: Fichiers de données non trouvés dans '{data_dir}'. Exécutez 'data_generator.py'.")
        return

    input_size = features.shape[1]
    num_phases = len(torch.unique(labels))

    print(f"   Taille d'entrée : {input_size}, Nombre de phases : {num_phases}")

    print("\n2. Instanciation du Modèle...")
    model = TrafficLightModel(input_size, num_phases)
    print(model)

    print("\n3. Définition de la Fonction de Perte et de l'Optimiseur...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"\n4. Début de l'entraînement pour {num_epochs} époques...")
    for epoch in range(num_epochs):
        outputs = model(features)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 500 == 0:
            print(f'   Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    print("\n5. Entraînement terminé.")

    print("\n6. Évaluation finale du modèle...")
    model.eval()
    with torch.no_grad():
        outputs = model(features)
        _, predicted = torch.max(outputs.data, 1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f'   Précision sur les données d\'entraînement: {accuracy:.2f}%')

    print("\n7. Sauvegarde du modèle entraîné...")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"   Modèle sauvegardé à '{model_save_path}'")

if __name__ == "__main__":
    train_model()
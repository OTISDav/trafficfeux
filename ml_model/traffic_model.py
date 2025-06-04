import torch
import torch.nn as nn

class TrafficLightModel(nn.Module):
    """
    Modèle de réseau de neurones simple pour prédire la phase de feux tricolores.
    Prend 8 entrées (flux de trafic) et prédit l'une des 6 phases.
    """
    def __init__(self, input_size, num_phases):
        super(TrafficLightModel, self).__init__()
        self.fc = nn.Linear(input_size, num_phases)

    def forward(self, x):
        output = self.fc(x)
        return output

if __name__ == "__main__":
    input_size = 8
    num_phases = 6
    model = TrafficLightModel(input_size, num_phases)
    print("Structure du modèle TrafficLightModel :")
    print(model)

    dummy_input = torch.tensor([[
        75.0, 20.0,
        10.0,  5.0,
        80.0, 25.0,
        12.0,  7.0
    ]], dtype=torch.float32)
    with torch.no_grad():
        output = model(dummy_input)
    print("\nSortie brute du modèle (logits) pour un échantillon :")
    print(output)
    print(f"ID de la phase prédite : {torch.argmax(output, dim=1).item()}")
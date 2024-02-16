import torch

# Beispiel-Listen
liste1 = [1, 2, 3, 4, 5]
liste2 = [6, 7, 8, 9, 10]

# Erstellen Sie einen Vektor aus den Listen
vektor = torch.tensor([liste1, liste2])

# Zugriff auf das erste Element in der ersten Liste
erstes_element = vektor[0, 0].item()

print("Das erste Element in der ersten Liste des Tensors ist:", erstes_element)

import argparse
import os
import json
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np

# Répertoire racine du projet
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Charger les classes depuis le JSON
classes_path = os.path.join(BASE_DIR, 'modeles', 'classes_cnn.json')
with open(classes_path, 'r', encoding='utf-8') as f:
    CLASSES = json.load(f)

# Définir les transformations (doivent être identiques à l'entraînement)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def load_model(weights_path, num_classes):
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()
    return model

def predict_image(model, img_path):
    img = Image.open(img_path).convert('RGB')
    img_t = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_t)
        _, pred = torch.max(outputs, 1)
        proba = torch.softmax(outputs, dim=1).max().item()
    label = CLASSES[pred.item()]
    return label, proba

def main():
    parser = argparse.ArgumentParser(description='Prédire la classe d\'une image avec le CNN')
    parser.add_argument('image', type=str, help='Chemin de l\'image à prédire')
    default_weights = os.path.join(BASE_DIR, 'modeles', 'best_cnn_model.pth')
    parser.add_argument('--weights', type=str, default=default_weights, help='Poids du modèle CNN')
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Image introuvable : {args.image}")
        return
    if not os.path.exists(args.weights):
        print(f"Fichier de poids introuvable : {args.weights}")
        return

    model = load_model(args.weights, num_classes=len(CLASSES))
    label, proba = predict_image(model, args.image)
    print(f"Prédiction : {label} (confiance : {proba:.2f})")

if __name__ == '__main__':
    main()

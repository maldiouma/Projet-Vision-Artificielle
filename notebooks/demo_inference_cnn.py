# Démo d'inférence CNN sur une image

import torch
from torchvision import models, transforms
from PIL import Image
import joblib
import matplotlib.pyplot as plt

# Charger le label encoder
le = joblib.load('../modeles/label_encoder_cnn.pkl')  # Adapter le chemin si besoin

# Charger le modèle
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(le.classes_))
model.load_state_dict(torch.load('../modeles/best_cnn_model.pth', map_location='cpu'))
model.eval()

# Définir les transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def predict_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img_t = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_t)
        _, pred = torch.max(outputs, 1)
        proba = torch.softmax(outputs, dim=1).max().item()
    label = le.inverse_transform([pred.item()])[0]
    return label, proba, img

# Exemple d'utilisation
img_path = '../donnees/images_pretraitees/EXEMPLE.jpg'  # À remplacer par une vraie image
label, proba, img = predict_image(img_path)
print(f"Prédiction : {label} (confiance : {proba:.2f})")
plt.imshow(img)
plt.title(f"Prédiction : {label} ({proba:.2f})")
plt.axis('off')
plt.show()

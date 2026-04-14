# Classification automatique de dechets par vision artificielle

## Probleme resolu

Dans un centre de tri, les objets defilent sur un tapis convoyeur. Un operateur doit identifier
manuellement la matiere de chaque objet (carton, verre, metal, papier, plastique, ordures) avant
de l'orienter vers la bonne filiere de recyclage. Ce processus est lent, fatiguant et source
d'erreurs.

Ce projet construit un systeme complet qui remplace cette etape manuelle : une camera capture
l'image de l'objet, un modele de deep learning predit sa classe en quelques millisecondes, et
le resultat est expose via une API REST que n'importe quelle interface industrielle peut interroger.

## Ce que fait ce projet, etape par etape

### 1. Ingestion et preparation des donnees

- Telechargement du dataset Garbage Classification (2 467 images, 6 classes) depuis Kaggle
- Generation d'un fichier de metadonnees (`metadata.parquet`) : chemin, classe, dimensions, hash MD5
- Nettoyage : suppression des doublons exacts par comparaison de hash
- Split reproductible train/val/test (70/15/15) avec stratification par classe
- Reequilibrage du split train par sur-echantillonnage des classes minoritaires

### 2. Analyse exploratoire (EDA)

- Distribution des classes avant et apres reequilibrage
- Distribution des tailles d'images (largeur, hauteur)
- Detection d'images corrompues
- Rapport de qualite des donnees (`donnees/rapport_qualite.txt`)

### 3. Preprocessement des images

- Redimensionnement a 224x224 pixels
- Normalisation avec les moyennes ImageNet
- Augmentation sur le train : flip horizontal, rotation, variation de luminosite/contraste

### 4. Modele baseline : HOG + SVM

- Extraction de features HOG (Histogram of Oriented Gradients)
- Classification par SVM avec noyau RBF
- Sert de reference pour mesurer le gain apporte par le CNN

### 5. Modele principal : ResNet18 (Transfer Learning)

- ResNet18 pre-entraine sur ImageNet, derniere couche adaptee aux 6 classes
- Entrainement sur 10 epochs avec AdamW, CrossEntropyLoss, scheduler StepLR
- Arret precoce si aucune amelioration sur la validation
- **Resultat obtenu : 83.95% de precision sur le jeu de test (F1-macro : 0.83)**

### 6. Evaluation

- Rapport de classification par classe (precision, rappel, F1)
- Matrice de confusion
- Comparaison CNN vs baseline HOG+SVM

### 7. API d'inference

- API REST construite avec FastAPI
- Endpoint `/predict` : recoit une image, retourne la classe predite et le score de confiance
- Deployable localement ou via Docker

## Structure du projet

```
donnees/
  images_pretraitees/     # Images redimensionnees et normalisees
    train/{classe}/
    val/{classe}/
    test/{classe}/
  metadata_equilibre.parquet   # Metadonnees avec splits
  rapport_qualite.txt          # Rapport EDA

notebooks/
  eda_qualite.ipynb             # Analyse exploratoire des donnees
  entrainement_cnn_colab.ipynb  # Entrainement ResNet18 + evaluation
  demo_inference_cnn.ipynb      # Demonstration de prediction sur images

src/
  data/       # Scripts de preparation des donnees
  modeles/    # Scripts d'entrainement (CNN, SVM)
  api/        # Service d'inference FastAPI

modeles/
  best_cnn_model.pth   # Poids du meilleur modele entraine
  classes_cnn.json     # Liste des classes pour l'inference

docker/       # Dockerfile et docker-compose
reports/      # Rapport technique, dataset card, figures
```

## Resultats

| Modele       | Accuracy test | F1-macro |
|--------------|---------------|----------|
| HOG + SVM    | baseline      | -        |
| ResNet18 CNN | **83.95%**    | **0.83** |

Detail par classe (ResNet18, 10 epochs, CPU) :

| Classe    | Precision | Rappel | F1   |
|-----------|-----------|--------|------|
| cardboard | 0.98      | 0.88   | 0.93 |
| paper     | 0.93      | 0.94   | 0.94 |
| metal     | 0.70      | 0.95   | 0.81 |
| glass     | 0.74      | 0.84   | 0.79 |
| plastic   | 0.92      | 0.60   | 0.73 |
| trash     | 0.83      | 0.75   | 0.79 |

## Lancer le projet

```bash
# Installer les dependances
pip install torch torchvision pandas pyarrow scikit-learn matplotlib seaborn Pillow tqdm

# Executer les notebooks dans l'ordre
# 1. eda_qualite.ipynb            -> analyse des donnees
# 2. entrainement_cnn_colab.ipynb -> entrainement du modele
# 3. demo_inference_cnn.ipynb     -> test de prediction

# Lancer l'API d'inference
python -m src.api.main
# ou avec Docker
docker-compose up --build
```

## Technologies utilisees

- **Python 3.10**
- **PyTorch / torchvision** : deep learning, ResNet18
- **scikit-learn** : SVM baseline, encodage, metriques
- **Pandas / PyArrow** : gestion des metadonnees
- **Pillow** : traitement d'images
- **FastAPI** : API REST d'inference
- **Docker** : containerisation du service
- **Jupyter** : exploration et documentation

## References

- Szeliski, *Computer Vision: Algorithms and Applications*
- Shanmugamani, *Deep Learning for Computer Vision*
- He et al., *Deep Residual Learning for Image Recognition* (ResNet, 2015)
- Bishop, *Pattern Recognition and Machine Learning*

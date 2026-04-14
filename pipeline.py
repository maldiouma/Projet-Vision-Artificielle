import os
import subprocess

def run_pipeline():
    """
    Exécute les étapes principales du pipeline de vision artificielle.
    Chaque étape doit être implémentée dans un module séparé.
    """
    # 1. Ingestion Kaggle
    os.system("python -m src.data.ingest_kaggle")
    # 2. Génération du manifeste
    os.system("python -m src.data.make_manifest")
    # 3. Génération des métadonnées et splits
    os.system("python -m src.data.make_metadata")
    # 4. Contrôle qualité et EDA
    os.system("python -m src.data.eda_quality")
    # 5. Prétraitement
    os.system("python -m src.transforms.preprocess")
    # 6. Baseline HOG + SVM
    os.system("python -m src.models.baseline_hog_svm")
    # 7. Entraînement CNN
    os.system("python -m src.models.train_cnn")
    # 8. Évaluation
    os.system("python -m src.models.evaluate")
    # 9. Export modèle
    os.system("python -m src.models.export")
    print("Pipeline terminé. Consultez les logs et rapports pour les résultats.")

if __name__ == "__main__":
    run_pipeline()

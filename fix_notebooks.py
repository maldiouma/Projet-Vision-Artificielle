import json
import os

# ─── EDA notebook ─────────────────────────────────────────────────────────────

EDA_CELLS = [
    {
        "id": "7382100b", "cell_type": "markdown",
        "source": [
            "# Analyse exploratoire des donnees (EDA)\n",
            "Dans ce notebook j'explore la qualite et la distribution des images du dataset Garbage Classification."
        ]
    },
    {
        "id": "d4cab036", "cell_type": "code",
        "source": [
            "# J'importe les bibliotheques necessaires\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "from PIL import Image\n",
            "import os\n",
            "\n",
            "# Je definis le dossier racine du projet\n",
            "base_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))\n",
            "print(f\"Base dir: {base_dir}\")"
        ]
    },
    {
        "id": "b96c0d11", "cell_type": "code",
        "source": [
            "# Je charge les metadonnees du dataset equilibre\n",
            "meta_path = os.path.join(base_dir, 'donnees', 'metadata_equilibre.parquet')\n",
            "meta = pd.read_parquet(meta_path)\n",
            "\n",
            "# Je corrige les chemins : les images sont stockees dans donnees/images_pretraitees/{split}/{etiquette}/{fichier}\n",
            "def fix_image_path(row):\n",
            "    filename = os.path.basename(str(row['chemin_fichier']))\n",
            "    etiquette = row['etiquette']\n",
            "    split = row['split']\n",
            "    candidate = os.path.join(base_dir, 'donnees', 'images_pretraitees', split, etiquette, filename)\n",
            "    if os.path.exists(candidate):\n",
            "        return candidate\n",
            "    # Si le split ne correspond pas, je cherche dans tous les splits\n",
            "    for s in ['train', 'val', 'test']:\n",
            "        c = os.path.join(base_dir, 'donnees', 'images_pretraitees', s, etiquette, filename)\n",
            "        if os.path.exists(c):\n",
            "            return c\n",
            "    return None\n",
            "\n",
            "meta['chemin_fichier'] = meta.apply(fix_image_path, axis=1)\n",
            "\n",
            "# Je filtre les entrees dont le chemin n'a pas pu etre resolu\n",
            "meta_valid = meta[meta['chemin_fichier'].notna()].copy()\n",
            "\n",
            "# Je deduplique pour l'EDA car le split train est sur-echantillonne\n",
            "meta_unique = meta_valid.drop_duplicates(subset='chemin_fichier').copy()\n",
            "\n",
            "print(f\"Images uniques trouvees : {len(meta_unique)}\")\n",
            "print(f\"Chemins non resolus : {meta['chemin_fichier'].isna().sum()}\")\n",
            "\n",
            "meta.head()"
        ]
    },
    {
        "id": "ea83f03f", "cell_type": "code",
        "source": [
            "# Je visualise la distribution des classes sur les images uniques\n",
            "plt.figure(figsize=(8, 4))\n",
            "sns.countplot(x='etiquette', data=meta_unique,\n",
            "              order=meta_unique['etiquette'].value_counts().index)\n",
            "plt.title(\"Distribution des classes\")\n",
            "plt.xticks(rotation=45)\n",
            "plt.tight_layout()\n",
            "plt.show()\n",
            "\n",
            "print(\"Distribution par classe :\")\n",
            "print(meta_unique['etiquette'].value_counts())"
        ]
    },
    {
        "id": "e7323f4c", "cell_type": "code",
        "source": [
            "# Je visualise la distribution des tailles d'images\n",
            "plt.figure(figsize=(10, 4))\n",
            "\n",
            "plt.subplot(1, 2, 1)\n",
            "sns.histplot(meta_unique['largeur'], kde=True, color='blue')\n",
            "plt.title(\"Distribution des largeurs\")\n",
            "\n",
            "plt.subplot(1, 2, 2)\n",
            "sns.histplot(meta_unique['hauteur'], kde=True, color='orange')\n",
            "plt.title(\"Distribution des hauteurs\")\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()\n",
            "\n",
            "print(f\"Largeur - min: {meta_unique['largeur'].min()}, max: {meta_unique['largeur'].max()}, moyenne: {meta_unique['largeur'].mean():.0f}\")\n",
            "print(f\"Hauteur - min: {meta_unique['hauteur'].min()}, max: {meta_unique['hauteur'].max()}, moyenne: {meta_unique['hauteur'].mean():.0f}\")"
        ]
    },
    {
        "id": "3ad798b8", "cell_type": "code",
        "source": [
            "# Je verifie si des images sont corrompues sur un echantillon de 100 images\n",
            "corrompues = []\n",
            "sample = meta_unique.sample(min(100, len(meta_unique)), random_state=42)\n",
            "\n",
            "for idx, row in sample.iterrows():\n",
            "    try:\n",
            "        with Image.open(row['chemin_fichier']) as img:\n",
            "            img.verify()\n",
            "    except Exception as e:\n",
            "        corrompues.append((row['chemin_fichier'], str(e)))\n",
            "\n",
            "print(f\"Images corrompues sur {len(sample)} testees : {len(corrompues)}\")\n",
            "if corrompues:\n",
            "    for path, error in corrompues[:5]:\n",
            "        print(f\"  - {os.path.basename(path)} : {error}\")\n",
            "else:\n",
            "    print(\"Toutes les images testees sont valides.\")"
        ]
    },
    {
        "id": "a235a3a4", "cell_type": "code",
        "source": [
            "# Je detecte les doublons exacts par comparaison de hash md5\n",
            "if 'md5' not in meta_unique.columns:\n",
            "    print(\"La colonne 'md5' est absente des metadonnees.\")\n",
            "else:\n",
            "    doublons = meta_unique[meta_unique.duplicated('md5', keep=False)]\n",
            "    print(f\"Doublons exacts detectes : {doublons.shape[0]}\")\n",
            "    if not doublons.empty:\n",
            "        for md5, count in doublons['md5'].value_counts()[doublons['md5'].value_counts() > 1].items():\n",
            "            print(f\"  - {md5} : {count} fichiers\")\n",
            "    else:\n",
            "        print(\"Aucun doublon detecte.\")"
        ]
    },
    {
        "id": "c1e2e3f4", "cell_type": "code",
        "source": [
            "# Je verifie la repartition des images par split et par classe\n",
            "print(\"Repartition par split (images uniques) :\")\n",
            "print(meta_unique['split'].value_counts())\n",
            "\n",
            "print(\"\\nRepartition par classe et par split :\")\n",
            "print(pd.crosstab(meta_unique['etiquette'], meta_unique['split']))"
        ]
    },
    {
        "id": "c5e6f7g8", "cell_type": "markdown",
        "source": [
            "---\n\n",
            "**Conclusion de l'EDA**\n\n",
            "- Les donnees sont correctement chargees et les chemins resolus.\n",
            "- La distribution des classes est equilibree.\n",
            "- Aucune image corrompue n'a ete detectee sur l'echantillon teste."
        ]
    }
]

# ─── CNN Training notebook ─────────────────────────────────────────────────────

CNN_CELLS = [
    {
        "id": "a691ba98", "cell_type": "markdown",
        "source": [
            "# Entrainement d'un modele CNN (ResNet18) - Transfer Learning\n",
            "Dans ce notebook j'entraine un ResNet18 pre-entraine sur le dataset de classification de dechets."
        ]
    },
    {
        "id": "f3ca3094", "cell_type": "code",
        "source": [
            "# J'importe toutes les bibliotheques necessaires\n",
            "import os\n",
            "import json\n",
            "import pandas as pd\n",
            "import numpy as np\n",
            "import torch\n",
            "import torch.nn as nn\n",
            "import torch.optim as optim\n",
            "from torch.utils.data import Dataset, DataLoader\n",
            "from torchvision import models, transforms\n",
            "from PIL import Image\n",
            "from sklearn.preprocessing import LabelEncoder\n",
            "from sklearn.metrics import classification_report, confusion_matrix\n",
            "from tqdm import tqdm\n",
            "import matplotlib.pyplot as plt\n",
            "import warnings\n",
            "warnings.filterwarnings('ignore')\n",
            "\n",
            "# Je definis le dossier racine du projet\n",
            "base_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))\n",
            "print(f\"Base dir: {base_dir}\")\n",
            "print(f\"CUDA disponible : {torch.cuda.is_available()}\")\n",
            "\n",
            "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
            "print(f\"Device utilise : {device}\")"
        ]
    },
    {
        "id": "3e220262", "cell_type": "markdown",
        "source": ["## 2. Chargement et preparation des donnees"]
    },
    {
        "id": "c48fbd56", "cell_type": "code",
        "source": [
            "# Je charge les metadonnees du dataset equilibre\n",
            "meta_path = os.path.join(base_dir, 'donnees', 'metadata_equilibre.parquet')\n",
            "print(f\"Chargement de : {meta_path}\")\n",
            "\n",
            "if not os.path.exists(meta_path):\n",
            "    raise FileNotFoundError(f\"Fichier metadonnees manquant : {meta_path}\")\n",
            "\n",
            "meta = pd.read_parquet(meta_path)\n",
            "print(f\"Total lignes (avec oversampling) : {len(meta)}\")\n",
            "\n",
            "# Je corrige les chemins car les images sont stockees dans images_pretraitees/{split}/{etiquette}/{fichier}\n",
            "def fix_image_path(row):\n",
            "    filename = os.path.basename(str(row['chemin_fichier']))\n",
            "    etiquette = row['etiquette']\n",
            "    split = row['split']\n",
            "    candidate = os.path.join(base_dir, 'donnees', 'images_pretraitees', split, etiquette, filename)\n",
            "    if os.path.exists(candidate):\n",
            "        return candidate\n",
            "    for s in ['train', 'val', 'test']:\n",
            "        c = os.path.join(base_dir, 'donnees', 'images_pretraitees', s, etiquette, filename)\n",
            "        if os.path.exists(c):\n",
            "            return c\n",
            "    return None\n",
            "\n",
            "meta['chemin_fichier'] = meta.apply(fix_image_path, axis=1)\n",
            "\n",
            "# Je supprime les entrees dont le chemin n'a pas pu etre resolu\n",
            "meta_valid = meta[meta['chemin_fichier'].notna()].copy()\n",
            "missing = len(meta) - len(meta_valid)\n",
            "if missing > 0:\n",
            "    print(f\"{missing} chemins non resolus (images oversamplees sans copie sur disque)\")\n",
            "    meta = meta_valid\n",
            "else:\n",
            "    print(f\"Tous les chemins sont resolus ({len(meta)} lignes)\")\n",
            "\n",
            "print(f\"\\nRepartition par split :\")\n",
            "print(meta['split'].value_counts())\n",
            "print(f\"\\nRepartition par classe :\")\n",
            "print(meta['etiquette'].value_counts())"
        ]
    },
    {
        "id": "c48fbd57", "cell_type": "code",
        "source": [
            "# J'encode les etiquettes en valeurs numeriques\n",
            "le = LabelEncoder()\n",
            "meta['label_enc'] = le.fit_transform(meta['etiquette'])\n",
            "\n",
            "print(f\"Classes : {le.classes_.tolist()}\")\n",
            "print(f\"Nombre de classes : {len(le.classes_)}\")"
        ]
    },
    {
        "id": "c48fbd58", "cell_type": "code",
        "source": [
            "# Je definis les transformations pour le train (avec augmentation) et pour val/test\n",
            "transform_train = transforms.Compose([\n",
            "    transforms.Resize((224, 224)),\n",
            "    transforms.RandomHorizontalFlip(p=0.5),\n",
            "    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),\n",
            "    transforms.RandomRotation(15),\n",
            "    transforms.ToTensor(),\n",
            "    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])\n",
            "])\n",
            "\n",
            "transform_val = transforms.Compose([\n",
            "    transforms.Resize((224, 224)),\n",
            "    transforms.ToTensor(),\n",
            "    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])\n",
            "])\n",
            "\n",
            "print(\"Transformations definies.\")"
        ]
    },
    {
        "id": "c48fbd59", "cell_type": "code",
        "source": [
            "# Je cree un Dataset PyTorch personnalise pour charger les images depuis le disque\n",
            "class ImageDataset(Dataset):\n",
            "    def __init__(self, df, transform):\n",
            "        self.df = df.reset_index(drop=True)\n",
            "        self.transform = transform\n",
            "\n",
            "    def __len__(self):\n",
            "        return len(self.df)\n",
            "\n",
            "    def __getitem__(self, idx):\n",
            "        img_path = self.df.loc[idx, 'chemin_fichier']\n",
            "        if not os.path.exists(img_path):\n",
            "            raise FileNotFoundError(f\"Image non trouvee : {img_path}\")\n",
            "        img = Image.open(img_path).convert('RGB')\n",
            "        img = self.transform(img)\n",
            "        label = self.df.loc[idx, 'label_enc']\n",
            "        return img, label\n",
            "\n",
            "print(\"Dataset defini.\")"
        ]
    },
    {
        "id": "c48fbd60", "cell_type": "code",
        "source": [
            "# Je separe les donnees par split et je cree les DataLoaders\n",
            "train_df = meta[meta['split'] == 'train']\n",
            "val_df   = meta[meta['split'] == 'val']\n",
            "test_df  = meta[meta['split'] == 'test']\n",
            "\n",
            "print(f\"Train : {len(train_df)} images | Val : {len(val_df)} images | Test : {len(test_df)} images\")\n",
            "\n",
            "batch_size = 32 if torch.cuda.is_available() else 16\n",
            "\n",
            "train_ds = ImageDataset(train_df, transform_train)\n",
            "val_ds   = ImageDataset(val_df,   transform_val)\n",
            "test_ds  = ImageDataset(test_df,  transform_val)\n",
            "\n",
            "# num_workers=0 pour la compatibilite Windows\n",
            "train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)\n",
            "val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)\n",
            "test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)\n",
            "\n",
            "print(f\"Batch size : {batch_size}\")\n",
            "print(f\"Batches - Train : {len(train_loader)} | Val : {len(val_loader)} | Test : {len(test_loader)}\")"
        ]
    },
    {
        "id": "713473f8", "cell_type": "markdown",
        "source": ["## 3. Architecture du modele (ResNet18 + Transfer Learning)"]
    },
    {
        "id": "fce82525", "cell_type": "code",
        "source": [
            "# Je charge ResNet18 pre-entraine sur ImageNet et j'adapte la derniere couche au nombre de classes\n",
            "print(\"Chargement de ResNet18 pre-entraine...\")\n",
            "model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)\n",
            "num_ftrs = model.fc.in_features\n",
            "model.fc = nn.Linear(num_ftrs, len(le.classes_))\n",
            "model = model.to(device)\n",
            "\n",
            "print(f\"Modele charge sur {device} | Nombre de classes : {len(le.classes_)}\")"
        ]
    },
    {
        "id": "8cfb08dd", "cell_type": "markdown",
        "source": ["## 4. Optimiseur et fonction de cout"]
    },
    {
        "id": "e89f52ff", "cell_type": "code",
        "source": [
            "# Je definis la fonction de cout, l'optimiseur et le scheduler de taux d'apprentissage\n",
            "criterion = nn.CrossEntropyLoss()\n",
            "optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)\n",
            "scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)\n",
            "\n",
            "print(\"Optimiseur : AdamW | Fonction de cout : CrossEntropyLoss | Scheduler : StepLR\")"
        ]
    },
    {
        "id": "328cd2e2", "cell_type": "markdown",
        "source": ["## 5. Boucle d'entrainement"]
    },
    {
        "id": "4f365b5f", "cell_type": "code",
        "source": [
            "# J'entraine le modele pendant 10 epochs avec arret precoce si aucune amelioration\n",
            "num_epochs = 10\n",
            "best_val_acc = 0.0\n",
            "best_model_path = os.path.join(base_dir, 'modeles', 'best_cnn_model.pth')\n",
            "patience = 3\n",
            "patience_counter = 0\n",
            "\n",
            "train_losses, val_losses = [], []\n",
            "train_accuracies, val_accuracies = [], []\n",
            "\n",
            "for epoch in range(num_epochs):\n",
            "    # Phase d'entrainement\n",
            "    model.train()\n",
            "    running_loss, running_corrects, total = 0.0, 0, 0\n",
            "\n",
            "    pbar = tqdm(train_loader, desc=f\"Epoch {epoch+1}/{num_epochs} [Train]\", leave=True)\n",
            "    for inputs, labels in pbar:\n",
            "        inputs, labels = inputs.to(device), labels.to(device)\n",
            "        optimizer.zero_grad()\n",
            "        outputs = model(inputs)\n",
            "        loss = criterion(outputs, labels)\n",
            "        loss.backward()\n",
            "        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)\n",
            "        optimizer.step()\n",
            "        _, preds = torch.max(outputs, 1)\n",
            "        running_loss += loss.item() * inputs.size(0)\n",
            "        running_corrects += torch.sum(preds == labels.data).item()\n",
            "        total += labels.size(0)\n",
            "        pbar.set_postfix({'loss': f'{running_loss/total:.4f}', 'acc': f'{running_corrects/total:.4f}'})\n",
            "\n",
            "    epoch_loss = running_loss / total\n",
            "    epoch_acc  = running_corrects / total\n",
            "    train_losses.append(epoch_loss)\n",
            "    train_accuracies.append(epoch_acc)\n",
            "\n",
            "    # Phase de validation\n",
            "    model.eval()\n",
            "    val_loss, val_corrects, val_total = 0.0, 0, 0\n",
            "    with torch.no_grad():\n",
            "        pbar_val = tqdm(val_loader, desc=f\"Epoch {epoch+1}/{num_epochs} [Val]\", leave=True)\n",
            "        for inputs, labels in pbar_val:\n",
            "            inputs, labels = inputs.to(device), labels.to(device)\n",
            "            outputs = model(inputs)\n",
            "            loss = criterion(outputs, labels)\n",
            "            _, preds = torch.max(outputs, 1)\n",
            "            val_loss += loss.item() * inputs.size(0)\n",
            "            val_corrects += torch.sum(preds == labels.data).item()\n",
            "            val_total += labels.size(0)\n",
            "\n",
            "    val_epoch_loss = val_loss / val_total\n",
            "    val_epoch_acc  = val_corrects / val_total\n",
            "    val_losses.append(val_epoch_loss)\n",
            "    val_accuracies.append(val_epoch_acc)\n",
            "\n",
            "    print(f\"Epoch {epoch+1}/{num_epochs} | Train Loss : {epoch_loss:.4f} Acc : {epoch_acc:.4f} | Val Loss : {val_epoch_loss:.4f} Acc : {val_epoch_acc:.4f}\")\n",
            "\n",
            "    # Je sauvegarde le modele si la precision de validation s'ameliore\n",
            "    if val_epoch_acc > best_val_acc:\n",
            "        best_val_acc = val_epoch_acc\n",
            "        patience_counter = 0\n",
            "        os.makedirs(os.path.dirname(best_model_path), exist_ok=True)\n",
            "        torch.save(model.state_dict(), best_model_path)\n",
            "        print(f\"  Meilleur modele sauvegarde (Val Acc : {val_epoch_acc:.4f})\")\n",
            "    else:\n",
            "        patience_counter += 1\n",
            "        if patience_counter >= patience:\n",
            "            print(f\"  Arret precoce : aucune amelioration depuis {patience} epochs.\")\n",
            "            break\n",
            "\n",
            "    scheduler.step()\n",
            "\n",
            "print(f\"\\nEntrainement termine. Meilleur modele sauvegarde dans : {best_model_path}\")"
        ]
    },
    {
        "id": "4f365b60", "cell_type": "code",
        "source": [
            "# J'affiche les courbes de loss et d'accuracy pour analyser l'entrainement\n",
            "plt.figure(figsize=(12, 4))\n",
            "\n",
            "plt.subplot(1, 2, 1)\n",
            "plt.plot(train_losses, label='Train Loss')\n",
            "plt.plot(val_losses,   label='Val Loss')\n",
            "plt.xlabel('Epoch')\n",
            "plt.ylabel('Loss')\n",
            "plt.title('Evolution de la loss')\n",
            "plt.legend()\n",
            "plt.grid(True, alpha=0.3)\n",
            "\n",
            "plt.subplot(1, 2, 2)\n",
            "plt.plot(train_accuracies, label='Train Accuracy')\n",
            "plt.plot(val_accuracies,   label='Val Accuracy')\n",
            "plt.xlabel('Epoch')\n",
            "plt.ylabel('Accuracy')\n",
            "plt.title('Evolution de la precision')\n",
            "plt.legend()\n",
            "plt.grid(True, alpha=0.3)\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "id": "9f757438", "cell_type": "markdown",
        "source": ["## 6. Evaluation sur le jeu de test"]
    },
    {
        "id": "080d841b", "cell_type": "code",
        "source": [
            "# Je charge le meilleur modele sauvegarde et je l'evalue sur le jeu de test\n",
            "print(f\"Chargement du meilleur modele : {best_model_path}\")\n",
            "model.load_state_dict(torch.load(best_model_path, map_location=device))\n",
            "model.eval()\n",
            "\n",
            "test_loss, test_corrects, test_total = 0.0, 0, 0\n",
            "all_preds, all_labels = [], []\n",
            "\n",
            "with torch.no_grad():\n",
            "    for inputs, labels in tqdm(test_loader, desc=\"Evaluation test\"):\n",
            "        inputs, labels = inputs.to(device), labels.to(device)\n",
            "        outputs = model(inputs)\n",
            "        loss = criterion(outputs, labels)\n",
            "        _, preds = torch.max(outputs, 1)\n",
            "        test_loss += loss.item() * inputs.size(0)\n",
            "        test_corrects += torch.sum(preds == labels.data).item()\n",
            "        test_total += labels.size(0)\n",
            "        all_preds.extend(preds.cpu().numpy())\n",
            "        all_labels.extend(labels.cpu().numpy())\n",
            "\n",
            "test_loss /= test_total\n",
            "test_acc   = test_corrects / test_total\n",
            "\n",
            "print(f\"\\nTest Loss : {test_loss:.4f} | Test Accuracy : {test_acc:.4f} ({test_acc*100:.2f}%)\")\n",
            "print(\"\\nRapport de classification :\")\n",
            "print(classification_report(all_labels, all_preds, target_names=le.classes_, digits=4))"
        ]
    },
    {
        "id": "080d841c", "cell_type": "code",
        "source": [
            "# J'affiche la matrice de confusion pour analyser les erreurs par classe\n",
            "from sklearn.metrics import ConfusionMatrixDisplay\n",
            "\n",
            "cm = confusion_matrix(all_labels, all_preds)\n",
            "disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)\n",
            "\n",
            "plt.figure(figsize=(10, 10))\n",
            "disp.plot(cmap='Blues', values_format='d')\n",
            "plt.title('Matrice de confusion - jeu de test')\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "id": "c4ef3279", "cell_type": "markdown",
        "source": ["## 7. Sauvegarde du modele et des classes"]
    },
    {
        "id": "5980b300", "cell_type": "code",
        "source": [
            "# Je sauvegarde la liste des classes dans un fichier JSON pour l'inference\n",
            "classes_list = le.classes_.tolist()\n",
            "classes_path = os.path.join(base_dir, 'modeles', 'classes_cnn.json')\n",
            "os.makedirs(os.path.dirname(classes_path), exist_ok=True)\n",
            "\n",
            "with open(classes_path, 'w', encoding='utf-8') as f:\n",
            "    json.dump(classes_list, f, ensure_ascii=False, indent=2)\n",
            "\n",
            "print(f\"Modele sauvegarde : {os.path.abspath(best_model_path)}\")\n",
            "print(f\"Classes sauvegardees : {os.path.abspath(classes_path)}\")\n",
            "print(f\"Classes : {classes_list}\")\n",
            "print(\"Sauvegarde terminee.\")"
        ]
    },
    {
        "id": "a8a4bc6f", "cell_type": "markdown",
        "source": [
            "---\n\n",
            "**Conclusion**\n\n",
            "L'entrainement est termine. Les fichiers suivants ont ete generes :\n",
            "- `modeles/best_cnn_model.pth` : poids du meilleur modele\n",
            "- `modeles/classes_cnn.json` : liste des classes pour l'inference"
        ]
    }
]

# ─── Demo Inference notebook ───────────────────────────────────────────────────

DEMO_CELLS = [
    {
        "id": "78186f4d", "cell_type": "markdown",
        "source": [
            "# Demonstration d'inference - Modele CNN\n",
            "Dans ce notebook je charge le modele entraine et je teste ses predictions sur des images du dataset."
        ]
    },
    {
        "id": "d7f88f84", "cell_type": "code",
        "source": [
            "# J'importe les bibliotheques necessaires pour l'inference\n",
            "import torch\n",
            "from torchvision import models, transforms\n",
            "from PIL import Image\n",
            "import json\n",
            "import matplotlib.pyplot as plt\n",
            "import os\n",
            "\n",
            "# Je definis le dossier racine du projet\n",
            "base_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))\n",
            "print(f\"Base dir: {base_dir}\")"
        ]
    },
    {
        "id": "dd99ded3", "cell_type": "code",
        "source": [
            "# Je charge la liste des classes depuis le fichier JSON genere lors de l'entrainement\n",
            "classes_path = os.path.join(base_dir, 'modeles', 'classes_cnn.json')\n",
            "\n",
            "if not os.path.exists(classes_path):\n",
            "    raise FileNotFoundError(\n",
            "        f\"Fichier des classes introuvable : {classes_path}\\n\"\n",
            "        \"Executer d'abord le notebook 'entrainement_cnn_colab.ipynb'.\"\n",
            "    )\n",
            "\n",
            "with open(classes_path, 'r', encoding='utf-8') as f:\n",
            "    CLASSES = json.load(f)\n",
            "\n",
            "print(f\"Classes chargees : {CLASSES}\")"
        ]
    },
    {
        "id": "dd99ded4", "cell_type": "code",
        "source": [
            "# Je charge le modele ResNet18 avec les poids entraines\n",
            "model_path = os.path.join(base_dir, 'modeles', 'best_cnn_model.pth')\n",
            "\n",
            "if not os.path.exists(model_path):\n",
            "    raise FileNotFoundError(\n",
            "        f\"Modele introuvable : {model_path}\\n\"\n",
            "        \"Executer d'abord le notebook 'entrainement_cnn_colab.ipynb'.\"\n",
            "    )\n",
            "\n",
            "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
            "model = models.resnet18(weights=None)\n",
            "model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES))\n",
            "model.load_state_dict(torch.load(model_path, map_location=device))\n",
            "model.eval()\n",
            "model.to(device)\n",
            "\n",
            "print(f\"Modele charge sur {device}\")"
        ]
    },
    {
        "id": "561fa736", "cell_type": "code",
        "source": [
            "# Je definis les transformations identiques a celles utilisees lors de la validation\n",
            "transform = transforms.Compose([\n",
            "    transforms.Resize((224, 224)),\n",
            "    transforms.ToTensor(),\n",
            "    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])\n",
            "])\n",
            "\n",
            "print(\"Transformations definies.\")"
        ]
    },
    {
        "id": "7aa8f98c", "cell_type": "code",
        "source": [
            "# Je definis une fonction qui predit la classe d'une image et retourne le top 3\n",
            "def predict_image(img_path):\n",
            "    img = Image.open(img_path).convert('RGB')\n",
            "    img_t = transform(img).unsqueeze(0).to(device)\n",
            "    with torch.no_grad():\n",
            "        outputs = model(img_t)\n",
            "        proba = torch.softmax(outputs, dim=1)\n",
            "        confidence, pred = torch.max(proba, 1)\n",
            "    label = CLASSES[pred.item()]\n",
            "    confidence_score = confidence.item()\n",
            "    top3 = torch.topk(proba[0], min(3, len(CLASSES)))\n",
            "    return label, confidence_score, img, top3\n",
            "\n",
            "print(\"Fonction de prediction definie.\")"
        ]
    },
    {
        "id": "ecc6cf16", "cell_type": "code",
        "source": [
            "# Je collecte toutes les images disponibles dans les sous-dossiers images_pretraitees/{split}/{classe}/\n",
            "images_root = os.path.join(base_dir, 'donnees', 'images_pretraitees')\n",
            "\n",
            "if not os.path.exists(images_root):\n",
            "    raise FileNotFoundError(f\"Dossier d'images introuvable : {images_root}\")\n",
            "\n",
            "image_files = []\n",
            "for split in os.listdir(images_root):\n",
            "    split_dir = os.path.join(images_root, split)\n",
            "    if not os.path.isdir(split_dir):\n",
            "        continue\n",
            "    for classe in os.listdir(split_dir):\n",
            "        classe_dir = os.path.join(split_dir, classe)\n",
            "        if not os.path.isdir(classe_dir):\n",
            "            continue\n",
            "        for fname in os.listdir(classe_dir):\n",
            "            if fname.lower().endswith(('.jpg', '.png', '.jpeg')):\n",
            "                image_files.append(os.path.join(classe_dir, fname))\n",
            "\n",
            "if not image_files:\n",
            "    raise FileNotFoundError(f\"Aucune image trouvee dans {images_root}\")\n",
            "\n",
            "print(f\"Total images disponibles : {len(image_files)}\")\n",
            "\n",
            "# Je teste le modele sur la premiere image disponible\n",
            "img_path = image_files[0]\n",
            "print(f\"Image de test : {os.path.relpath(img_path, base_dir)}\")\n",
            "\n",
            "label, confidence, img, top3 = predict_image(img_path)\n",
            "\n",
            "print(f\"\\nPrediction : {label}\")\n",
            "print(f\"Confiance  : {confidence:.2%}\")\n",
            "print(f\"\\nTop {len(top3.values)} predictions :\")\n",
            "for i, (conf, idx) in enumerate(zip(top3.values, top3.indices)):\n",
            "    print(f\"  {i+1}. {CLASSES[idx.item()]} : {conf.item():.2%}\")\n",
            "\n",
            "plt.figure(figsize=(6, 6))\n",
            "plt.imshow(img)\n",
            "plt.title(f\"Prediction : {label} ({confidence:.0%})\")\n",
            "plt.axis('off')\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "id": "ecc6cf17", "cell_type": "code",
        "source": [
            "# Je teste le modele sur plusieurs images aleatoires pour evaluer visuellement ses predictions\n",
            "import random\n",
            "\n",
            "def test_multiple_images(num_images=5):\n",
            "    selected = random.sample(image_files, min(num_images, len(image_files)))\n",
            "    fig, axes = plt.subplots(1, len(selected), figsize=(4 * len(selected), 4))\n",
            "    if len(selected) == 1:\n",
            "        axes = [axes]\n",
            "    for ax, path in zip(axes, selected):\n",
            "        try:\n",
            "            lbl, conf, img, _ = predict_image(path)\n",
            "            ax.imshow(img)\n",
            "            ax.set_title(f\"{lbl}\\n({conf:.0%})\")\n",
            "            ax.axis('off')\n",
            "        except Exception as e:\n",
            "            ax.text(0.5, 0.5, f\"Erreur : {str(e)[:30]}\", ha='center', va='center')\n",
            "            ax.axis('off')\n",
            "    plt.tight_layout()\n",
            "    plt.show()\n",
            "\n",
            "random.seed(42)\n",
            "test_multiple_images(num_images=3)"
        ]
    },
    {
        "id": "c5e6f7g8", "cell_type": "markdown",
        "source": [
            "---\n\n",
            "**Conclusion**\n\n",
            "Le modele charge correctement les images, effectue les predictions et retourne les scores de confiance par classe."
        ]
    }
]


def make_cell(c):
    base = {
        "cell_type": c["cell_type"],
        "id": c["id"],
        "metadata": {},
        "source": c["source"]
    }
    if c["cell_type"] == "code":
        base["execution_count"] = None
        base["outputs"] = []
    elif c["cell_type"] == "markdown":
        base["attachments"] = {}
    return base


KERNEL = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.10.0"
    }
}

notebooks = {
    "eda_qualite.ipynb": EDA_CELLS,
    "entrainement_cnn_colab.ipynb": CNN_CELLS,
    "demo_inference_cnn.ipynb": DEMO_CELLS,
}

base = os.path.join(os.path.dirname(__file__), "notebooks")

for fname, cells in notebooks.items():
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": KERNEL,
        "cells": [make_cell(c) for c in cells]
    }
    path = os.path.join(base, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"Ecrit : {path}")

print("Tous les notebooks ont ete mis a jour.")

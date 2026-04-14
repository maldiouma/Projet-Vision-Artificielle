# Dataset Card – Garbage Classification

**Source** : Kaggle (slug : asdasdasasdas/garbage-classification)
**Date de téléchargement** : JJ/MM/AAAA
**Hash ZIP** : <à compléter>
**Version** : <à compléter>

**Licence / Conditions d’usage** :
Voir la page officielle Kaggle pour la licence et les conditions d’utilisation.

**Description** :
- 6 classes : cardboard (393), glass (491), metal (400), paper (584), plastic (472), trash (127)
- 2467 images, format JPG/PNG
- Tailles images : min/median/max à compléter

**Qualité** :
- Images corrompues : à détecter
- Doublons : à détecter (hash, pHash)
- Déséquilibre de classes : oui (voir chiffres)
- Anomalies : bruit, fonds, occlusions possibles

**Split** :
- Stratifié (ex : 70/15/15), seed fixe
- Fichier splits.csv généré

**Risques** :
- Biais de fond/éclairage
- Généralisation limitée
- Confusion entre certaines classes (ex : plastic vs glass)

**But** : Permettre un audit reproductible du dataset (téléchargement, hash, split, seed).

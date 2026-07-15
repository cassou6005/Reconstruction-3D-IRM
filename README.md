# Reconstruction 3D d'IRM cérébrales

## Description

Ce projet propose deux approches pour la reconstruction et l'analyse d'IRM cérébrales 3D à partir du jeu de données **OASIS** :

- **SIFT 3D + KNN** : une méthode de vision par ordinateur reposant sur l'extraction de points clés SIFT 3D et leur mise en correspondance grâce à l'algorithme des *k plus proches voisins (KNN)*.
- **U-Net** : une approche basée sur l'apprentissage profond utilisant un réseau de neurones convolutif de type U-Net afin d'effectuer la reconstruction des volumes IRM.

Les deux méthodes sont comparées à l'aide de métriques de qualité d'image afin d'évaluer leurs performances et leurs limites.

---

## Méthodes

### SIFT 3D + KNN

Cette approche consiste à :

- extraire les points clés SIFT 3D des volumes IRM ;
- rechercher les correspondances entre images avec l'algorithme KNN ;
- reconstruire les structures cérébrales à partir de ces correspondances.

Cette méthode est simple à mettre en œuvre mais reste sensible à la qualité des points clés et aux variations anatomiques.

### U-Net

Le modèle U-Net est entraîné sur les volumes IRM afin d'apprendre directement la reconstruction des images.

Cette approche permet généralement d'obtenir des résultats plus précis et plus robustes grâce à l'apprentissage automatique.

---

## Évaluation

Les performances sont évaluées avec les métriques suivantes :

- **SSIM (Structural Similarity Index)** : mesure la similarité structurelle entre l'image reconstruite et l'image de référence.
- **MSE (Mean Squared Error)** : mesure l'erreur quadratique moyenne entre la reconstruction et l'image réelle.

---

## Installation

Installer les dépendances nécessaires :

```bash
pip install SimpleITK numpy matplotlib pillow SSIM_PIL scipy
```

Selon la version utilisée, installer également le framework de deep learning correspondant (TensorFlow ou PyTorch).

---

## Jeu de données

Le projet utilise le jeu de données **OASIS Brain MRI**.

Téléchargement :

- http://www.matthewtoews.com/projects/oasis/
- http://www.matthewtoews.com/projects/oasis/oasis_brains.zip

Le dossier contient :

- les volumes IRM 3D ;
- les fichiers `.hdr`, `.img` et `.mhd` ;
- les points clés SIFT (`.key`).

Découpage utilisé :

- **0001 → 0399** : entraînement
- **0400 → 0416** : évaluation

---

## Configuration

Avant d'exécuter les scripts, vérifier les chemins d'accès vers :

- le dossier contenant les images IRM ;
- les fichiers de points clés SIFT ;
- les dossiers de sortie.

Modifier ces chemins en fonction de votre environnement.

---

## Structure du projet

```
.
├── KNN/
│   ├── ...
│
├── UNet/
│   ├── ...
│
├── Dataset/
│   ├── OASIS/
│
├── Results/
│
└── README.md
```

---

## Résultats

Le projet compare une méthode classique de vision par ordinateur (**SIFT 3D + KNN**) à une approche d'apprentissage profond (**U-Net**).

Cette comparaison met en évidence les avantages et les limites de chacune des méthodes pour la reconstruction d'IRM cérébrales.

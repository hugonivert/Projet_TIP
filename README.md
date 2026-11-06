# Projet TIP – Challenge Food-11

Classification d'images de nourriture en **11 catégories** avec un réseau de neurones convolutif (CNN) sous MATLAB.
Le sujet complet est dans `[SUJET]_food-11.html` (non versionné, à récupérer sur Moodle).

> ⚠️ Ce README est à mettre à jour à chaque étape importante du projet (voir [État d'avancement](#état-davancement)).

---

## 1. Mise en place (sur n'importe quelle machine)

### 1.1 Prérequis

| Élément | Version / détail |
|---|---|
| MATLAB | **R2024a ou plus récent** (le code utilise `trainnet`, `minibatchpredict`, `scores2label`) |
| Deep Learning Toolbox | obligatoire (réseau, `trainnet`, `augmentedImageDatastore`) |
| Image Processing Toolbox | obligatoire (`imresize`, `montage`, `imtile`) |
| Parallel Computing Toolbox | optionnel, uniquement pour entraîner sur GPU (NVIDIA) |

Pour vérifier les toolboxes installées, tapez `ver` dans la console MATLAB.

### 1.2 Cloner le dépôt

```bash
git clone https://github.com/hugonivert/Projet_TIP.git
cd Projet_TIP
```

### 1.3 Ajouter le jeu de données

Le jeu de données **n'est pas sur GitHub** (trop lourd, et le sujet interdit les binaires dans le rendu).

1. Téléchargez `food-11.zip` (fourni par le tuteur sur Moodle).
2. Décompressez-le **à la racine du dépôt**, pour obtenir exactement cette arborescence :

```
Projet_TIP/
├── CNN_Projet.m
├── prepareData.m
├── README.md
├── .gitignore
└── food-11/              ← ignoré par git
    ├── train/
    │   ├── bread/            (1356 images)
    │   ├── dairy/            (573)
    │   ├── dessert/          (2000)
    │   ├── egg/              (1313)
    │   ├── fried/            (1174)
    │   ├── meat/             (1774)
    │   ├── noodles-pasta/    (587)
    │   ├── rice/             (376)
    │   ├── seafood/          (1202)
    │   ├── soup/             (2000)
    │   └── vegetable-fruit/  (941)
    └── test/                 (3097 images 0.jpg, 1.jpg, … SANS étiquette)
```

Attention aux doubles dossiers à la décompression (`food-11/food-11/train` ❌). Le chemin doit être `food-11/train` ✅.

Contrôle rapide : 13 296 images dans `train`, 3 097 dans `test`.

### 1.4 Lancer le code

1. Ouvrez MATLAB.
2. **Placez le dossier courant de MATLAB à la racine du dépôt** (`Projet_TIP/`). Les chemins dans le code sont relatifs (`"food-11/train"`), donc lancer le script depuis un autre dossier fait échouer `imageDatastore`.
3. Lancez `prepareData.m` pour vérifier que les données sont bien trouvées (affiche 16 images au hasard).
4. Lancez `CNN_Projet.m` pour le pipeline complet.

### 1.5 (Optionnel) Utiliser le GPU

Tapez `canUseGPU` dans la console MATLAB : `1` = GPU utilisable, `0` = CPU uniquement (sur Mac, ce sera toujours `0`).

Si vous avez `1`, dans `CNN_Projet.m`, à la fin du bloc `trainingOptions` :

```matlab
    Metrics="accuracy", ...          % remplacer la ")" par ", ..."
    ExecutionEnvironment="gpu");     % décommenter cette ligne
```

Décommenter seulement la ligne `ExecutionEnvironment` provoque une erreur de syntaxe : il faut aussi modifier la ligne `Metrics` au-dessus.

---

## 2. Architecture du code

### Fichiers

| Fichier | Rôle |
|---|---|
| `prepareData.m` | Script de vérification : charge `food-11/train` et affiche une mosaïque de 16 images. |
| `CNN_Projet.m` | Pipeline principal : données → réseau → entraînement → évaluation. |
| `.gitignore` | Exclut `food-11/` et `.DS_Store` du dépôt. |

### Déroulement de `CNN_Projet.m`

1. **`readFoodImage`** : fonction appelée sur chaque image lue. Elle convertit l'image en RGB (niveaux de gris → 3 canaux, RGBA → RGB), la redimensionne en 128×128 et normalise les pixels entre 0 et 1.
2. **Chargement des données** :
   - `imageDatastore` sur `food-11/train`, avec l'étiquette = nom du sous-dossier ;
   - `imageDatastore` sur `food-11/test` (sans étiquette) ;
   - `splitEachLabel(..., 0.8, "randomized")` : 80 % entraînement / 20 % validation, en gardant les proportions de chaque classe (≈ 10 637 / 2 659 images) ;
   - `augmentedImageDatastore` : fournit les images au réseau par lots, au format `[128 128 3]` ;
   - affichage de 25 images d'entraînement pour contrôle.
3. **Réseau** : un CNN minimal (1 convolution 3×3 de 8 filtres → batch norm → ReLU → max pooling → couche dense de 11 sorties → softmax).
4. **Entraînement** : `trainnet` avec SGD à momentum, 16 époques, lots de 500 images, la validation étant évaluée toutes les 50 itérations.
5. **Évaluation** : précision sur le train, puis sur le test (voir problèmes connus).

### Notions utiles

- **Train / validation / test** : on entraîne sur *train* et on choisit les hyperparamètres et l'architecture d'après la *validation*. Le *test* n'a pas d'étiquettes : seul le serveur d'évaluation les connaît.
- **Ordre des fichiers de test** : MATLAB les trie par ordre alphabétique (`0, 1, 10, 100, …`) et non par ordre numérique. Pour le JSON, associez toujours la prédiction au **nom du fichier**, pas à sa position dans la liste.

---

## 3. Rendu attendu (rappel du sujet)

- **Au moins 2 réseaux différents** entraînés, validés et comparés (le transfer learning est autorisé). On soumet le meilleur.
- **Fichier JSON de prédictions** sur le test, au format strict, avec les noms de classes en minuscules identiques aux noms des dossiers :
  ```json
  { "0": "bread", "1": "bread", "2": "fried" }
  ```
- **Classement** : score de frugalité, qui combine le F1 (moyenne sur les classes) et le **nombre de paramètres** du réseau. Un petit réseau bien entraîné peut battre un gros réseau.
- **Leaderboard** : une **seule** entrée par groupe. Testez sans cocher « update leaderboard » et ne cochez que pour la soumission finale.
- **Date limite** : 15 novembre, 23:59 sur Moodle. Rendre le code commenté et exécutable **sans fichiers binaires**, plus un rapport de 10 pages maximum.

---

## 4. État d'avancement

- [x] Chargement et préparation des données (vérifié sous MATLAB R2026b)
- [ ] Augmentation de données (symétries, rotations, zoom…) sur le train
- [ ] Gestion du déséquilibre des classes (rice : 376 images contre 2000 pour dessert et soup)
- [ ] Graine aléatoire fixe (`rng`) pour un découpage train/validation reproductible
- [ ] Réseau n°1 (CNN « maison »)
- [ ] Réseau n°2 (par ex. transfer learning sur un réseau pré-entraîné léger)
- [ ] Évaluation sur la validation (précision, F1, matrice de confusion)
- [ ] Génération du JSON de prédictions sur le test
- [ ] Rapport

### Problèmes connus

- **`CNN_Projet.m`, fin du script** : `labelsTrain` et `labelsTest` ne sont pas définis, donc le script plante après l'entraînement. De toute façon, le test n'a pas d'étiquettes : il faut évaluer sur la validation et générer le JSON pour le test.
- **Ligne `subset = readall(subset(...))`** : la variable porte le nom de la fonction qu'elle appelle. Ça fonctionne grâce au `clear` en début de script, mais il vaudrait mieux la renommer.
- **`ReadFcn` personnalisée** : elle ralentit la lecture des images et refait le redimensionnement déjà fait par `augmentedImageDatastore`. À optimiser.

---

## 5. Conventions de travail

- Ne **jamais** versionner `food-11/` ni d'autres images ou modèles entraînés (`.mat`).
- Commits explicites : on décrit ce qui a changé dans le message, et si une manipulation est nécessaire pour lancer le code, on l'indique ici dans le README.
- Avant de pousser, vérifier que `CNN_Projet.m` s'exécute depuis la racine du dépôt.

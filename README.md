# Projet TIP : Challenge Food-11

Classification d'images de nourriture en **11 catégories** avec un réseau de neurones convolutif (CNN) sous MATLAB.
Le sujet complet est dans `[SUJET]_food-11.html` (non versionné, à récupérer sur Moodle).

> Ce README est à mettre à jour à chaque étape importante du projet (voir [État d'avancement](#état-davancement)).

---

## 1. Mise en place (sur n'importe quelle machine)

### 1.1 Prérequis

| Élément | Version / détail |
|---|---|
| MATLAB | **R2024a ou plus récent** (le code utilise `trainnet`, `minibatchpredict`, `scores2label`) |
| Deep Learning Toolbox | obligatoire (réseau, `trainnet`, `augmentedImageDatastore`, `confusionmat`) |
| Image Processing Toolbox | obligatoire (`montage`) |
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
├── evaluerReseau.m
├── makeCourseProgressOutputFcn.m
├── prepareData.m
├── README.md
├── suiviCNN.txt
├── .gitignore
├── built/                <- créé par CNN_Projet.m, ignoré par git
│   ├── essais.csv
│   ├── historique_<essai>_graine<n>.csv
│   ├── predictions_<essai>_graine<n>.json
│   └── reseau_<essai>_graine<n>.mat
└── food-11/              <- ignoré par git
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
    └── test/                 (3097 images 0.jpg, 1.jpg, ... SANS étiquette)
```

Attention aux doubles dossiers à la décompression (`food-11/food-11/train` est incorrect). Le chemin doit être `food-11/train`.

Contrôle rapide : 13 296 images dans `train`, 3 097 dans `test`.

### 1.4 Lancer le code

1. Ouvrez MATLAB.
2. **Placez le dossier courant de MATLAB à la racine du dépôt** (`Projet_TIP/`). Les chemins dans le code sont relatifs (`"food-11/train"`), donc lancer le script depuis un autre dossier fait échouer `imageDatastore`.
3. Lancez `prepareData.m` pour vérifier que les données sont bien trouvées (affiche 16 images au hasard).
4. Lancez `CNN_Projet.m` pour le pipeline complet. Les prédictions du test sont écrites dans `built/predictions_<essai>_graine<n>.json` (le dossier `built/` est créé automatiquement), un fichier par essai. Par défaut, le script reproduit le meilleur réglage à ce stade (essai 8 de `suiviCNN.txt`, environ 45 min sur le CPU d'un Mac).

### 1.5 Faire un essai

Tous les réglages sont regroupés dans le bloc `Réglages de l'essai` en haut de `CNN_Projet.m` (structure `cfg`) : nom de l'essai, graines, taille des images, filtres des blocs de convolution, optimiseur, taux d'apprentissage et son calendrier, augmentation de données, poids de classes, dropout, taille des lots, nombre d'epochs, affichage des graphiques. Pour un nouvel essai, on ne modifie que ce bloc.

- `cfg.nomEssai` : nom court qui décrit l'essai (il apparaît dans le journal).
- `cfg.graineDecoupage = 42` : **ne jamais la changer**. Elle garantit le même découpage train/validation pour tous les essais et pour tout le groupe.
- `cfg.graineEntrainement` : 1 par défaut. Pour mesurer la part de hasard d'un résultat, relancer le même réglage avec 2 puis 3 et comparer les moyennes.
- `cfg.filtres = [16 32 64 96 128]` : un bloc de convolution par valeur, avec ce nombre de filtres. On change l'architecture en modifiant seulement cette ligne.
- `cfg.calendrierTaux` : `"none"` (taux constant) ou `"cosine"` (le taux décroît en douceur jusqu'au taux de départ x 1e-5, atteint à la dernière epoch). Avec `"cosine"`, tout le calendrier dépend du nombre d'epochs : pour choisir ce nombre, on compare des essais complets.
- `cfg.augmentation` : `true` par défaut (adoptée à l'essai 5). Avec `true`, chaque image du train est transformée au hasard à chaque epoch (retournement horizontal une fois sur deux, translation de +/- 40 pixels de l'image d'origine, zoom entre 0,9 et 1,1, zones découvertes remplies avec la couleur moyenne du train). La validation et le test ne sont jamais augmentés, et l'exactitude train finale est mesurée sur les vraies images.
- `cfg.poidsClasses` : `"aucun"` par défaut. Avec `"inverse"`, chaque classe reçoit dans la perte un poids égal à nombre d'images du train / (11 × images de la classe) : une erreur sur rice compte environ 3 fois plus qu'une erreur moyenne, une erreur sur dessert ou soup 0,6 fois. Les poids sont affichés au lancement. Les pertes de l'historique sont alors pondérées et ne se comparent pas à celles des essais sans poids. Testé à l'essai 7, non adopté (voir `suiviCNN.txt`).
- `cfg.dropout` : `0` par défaut (pas de dropout). Avec une valeur entre 0 et 1 (par exemple `0.3`), une couche de dropout est ajoutée entre le global average pooling et la couche dense : pendant l'entraînement, chaque valeur y est mise à zéro avec cette probabilité et les autres sont multipliées par 1/(1 - p). Les colonnes `Lots` de l'historique sont alors mesurées avec le dropout actif. Aucun paramètre ajouté. Testé à l'essai 9 avec `0.3`, sans effet (voir `suiviCNN.txt`).
- `cfg.afficherCourbes = false` : aucun graphique, pour enchaîner des essais sans fenêtre.

Chaque exécution ajoute une ligne dans `built/essais.csv` (réglages, nombre de paramètres, exactitude train et validation, F1 macro, frugalité, durée). Si les colonnes du journal changent, l'ancien fichier est archivé sous `built/essais_ancien_<date>.csv` et un nouveau journal est commencé.

Chaque exécution écrit aussi `built/historique_<essai>_graine<n>.csv`, avec une ligne par epoch (taux d'apprentissage de l'epoch `tauxEpoch`, perte et exactitude sur le train et sur la validation), pour tracer les courbes et faire le diagnostic biais/variance même sans fenêtre graphique. Les colonnes `perteTrainLots` et `exactitudeTrainLots` sont des moyennes des lots de l'epoch, mesurées pendant l'apprentissage : elles ne sont pas égales à l'exactitude train finale du journal, qui reste la référence pour comparer train et validation. Chaque exécution sauvegarde enfin le réseau entraîné dans `built/reseau_<essai>_graine<n>.mat` (environ 740 Ko pour le réseau par défaut) : réseau, réglages, noms des classes, fichiers et vraies classes des images de validation. On peut ainsi refaire des prédictions, la matrice de confusion ou des figures sans réentraîner. Ce fichier binaire reste dans `built/`, hors du dépôt.

Les fichiers propres à un essai portent son nom et sa graine. **Changer `cfg.nomEssai` à chaque nouveau réglage**, sinon les fichiers de l'essai précédent sont écrasés.

### 1.6 (Optionnel) Utiliser le GPU

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
| `CNN_Projet.m` | Pipeline principal : réglages, données, réseau, entraînement, évaluation, export JSON, journal des essais. |
| `evaluerReseau.m` | Fonction commune d'évaluation : exactitude, F1 par classe, F1 macro, nombre de paramètres, score de frugalité, matrice de confusion. À utiliser pour tous les réseaux du groupe afin que les scores soient comparables. |
| `makeCourseProgressOutputFcn.m` | Courbes d'entraînement lissées avec les epochs marquées, reprise du fichier du TP CIFAR-10, avec la courbe « Test set » renommée « Validation set ». |
| `suiviCNN.txt` | Journal de bord du CNN personnel : règles de la démarche, choix de conception justifiés, et pour chaque essai la raison, les réglages, les résultats, l'analyse et la décision prise. Sert de base au rapport. |
| `.gitignore` | Exclut `food-11/`, `built/`, `.DS_Store` et le fichier du sujet du dépôt. |

### Déroulement de `CNN_Projet.m`

1. **Réglages** : bloc `cfg` en tête de script (voir 1.5).
2. **Chargement des données** :
   - `imageDatastore` sur `food-11/train`, avec l'étiquette = nom du sous-dossier ;
   - `imageDatastore` sur `food-11/test` (sans étiquette) ;
   - `rng(cfg.graineDecoupage, "twister")` puis `splitEachLabel(..., 0.8, "randomized")` : 80 % entraînement / 20 % validation, en gardant les proportions de chaque classe (env. 10 637 / 2 659 images), toujours le même découpage ;
   - `augmentedImageDatastore` : redimensionne les images à `cfg.tailleImage`, convertit celles en niveaux de gris en RGB (`ColorPreprocessing="gray2rgb"`) et les fournit au réseau par lots. La normalisation est faite par la couche d'entrée du réseau (centrage sur la moyenne du train). Sans fonction de lecture personnalisée, la lecture est environ 9 fois plus rapide et une epoch environ 4 fois plus rapide (mesuré sur Mac, CPU).
3. **Réseau** : le bloc du TP CIFAR-10 (convolution 3×3 avec padding "same", batch norm, ReLU, max pooling 2×2) répété une fois par valeur de `cfg.filtres` (5 blocs de 16, 32, 64, 96 puis 128 filtres), suivi d'un global average pooling, d'une couche de dropout si `cfg.dropout > 0`, d'une couche dense de 11 sorties et d'un softmax. Le global average pooling réduit chaque carte de caractéristiques à sa moyenne : la couche dense n'a plus que 128 × 11 poids, et le nombre de paramètres (191 787, pénalité de frugalité 0,048) ne dépend pas de la taille des images. Le réseau à une couche du TP en avait env. 350 000, presque tous dans sa couche dense. Si `cfg.afficherCourbes` est vrai, affichage de 25 images d'entraînement et de l'architecture (`analyzeNetwork`).
4. **Entraînement** : `trainnet` avec les réglages de `cfg` (par défaut adam, taux d'apprentissage 0,001 avec calendrier `cosine`, lots de 128, 60 epochs), validation évaluée une fois par epoch. `OutputNetwork="last-iteration"` : on garde le réseau de la dernière epoch, et non celui de la meilleure epoch sur la validation, pour que les scores de validation ne soient pas optimistes et que l'exactitude train corresponde à la fin de l'entraînement. Perte d'entropie croisée, pondérée par classe si `cfg.poidsClasses = "inverse"`. `rng(cfg.graineEntrainement, "twister")` juste avant fixe les poids de départ et l'ordre des images. Si `cfg.afficherCourbes` est vrai, la fenêtre de MATLAB est complétée par les courbes lissées du TP.
5. **Sauvegarde et historique** : juste après l'entraînement, le réseau est sauvegardé dans `built/reseau_<essai>_graine<n>.mat` et l'historique par epoch dans `built/historique_<essai>_graine<n>.csv` (voir 1.5), avant l'évaluation, pour être gardés même si la suite plante.
6. **Évaluation** : exactitude sur le train (pour repérer un sous-apprentissage), puis `evaluerReseau` sur la validation : exactitude, F1 de chaque classe, F1 macro, nombre de paramètres et score de frugalité, avec la matrice de confusion si `cfg.afficherCourbes` est vrai. Le test n'ayant pas d'étiquettes, on prédit seulement ses classes.
7. **Export JSON** : les prédictions du test sont écrites dans `built/predictions_<essai>_graine<n>.json` au format attendu par le serveur (`{"0": "bread", ...}`). Chaque prédiction est associée au nom de son fichier image.
8. **Journal** : une ligne ajoutée dans `built/essais.csv`.

### Notions utiles

- **Train / validation / test** : on entraîne sur *train* et on choisit les hyperparamètres et l'architecture d'après la *validation*. Le *test* n'a pas d'étiquettes : seul le serveur d'évaluation les connaît.
- **Ordre des fichiers de test** : MATLAB les trie par ordre alphabétique (`0, 1, 10, 100, ...`) et non par ordre numérique. Pour le JSON, associez toujours la prédiction au **nom du fichier**, pas à sa position dans la liste.
- **Score de frugalité** : `F1 macro - 0.3 / (1 + 1/R)`, avec **R en millions de paramètres** (c'est la seule convention qui redonne les exemples chiffrés du sujet : 3,5 M de paramètres et F1 = 0,80 donnent 0,57). Pénalité : 0,027 pour 100 000 paramètres, 0,05 pour 200 000, 0,10 pour 500 000, 0,15 pour 1 M, 0,23 pour 3,5 M. On juge les essais sur ce score, pas sur l'exactitude.
- **F1 macro plutôt qu'exactitude** : les classes sont déséquilibrées (rice : 376 images, dessert : 2000). Un réseau qui ne reconnaît jamais le riz perd peu d'exactitude, mais un onzième de son F1 macro, la mesure utilisée par le serveur.
- **Graines** : la graine de découpage (42) donne la même validation à tous les essais et à tous les réseaux du groupe, ce qui rend les comparaisons équitables. La graine d'entraînement (1, 2, 3) fixe les poids de départ et l'ordre des images : relancer un réglage avec plusieurs graines mesure la part de hasard d'un résultat. Pour un réseau d'un autre script, appeler `rng(42, "twister")` juste avant `splitEachLabel(imgDataTrain, 0.8, "randomized")` sur `food-11/train` donne la même validation.

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
- [x] Augmentation de données sur le train (retournement horizontal, translations, zoom), activée par défaut
- [x] Gestion du déséquilibre des classes (rice : 376 images contre 2000 pour dessert et soup) : option `cfg.poidsClasses`, testée à l'essai 7, sans gain sur le score
- [x] Graines aléatoires fixes (`rng`) : découpage train/validation et entraînement reproductibles
- [x] Outils de réglage : réglages regroupés en tête de script, lecture des images accélérée, journal des essais (`built/essais.csv`) et historique par epoch (`built/historique_<essai>_graine<n>.csv`)
- [ ] Réseau n°1 (CNN « maison ») : 5 blocs + global average pooling, 191 787 paramètres. Meilleur résultat à ce stade : frugalité 0,673 sur la validation (calendrier `cosine`, 60 epochs, augmentation de données). Réglage en cours, détail des essais dans `suiviCNN.txt`
- [ ] Réseau n°2 (par ex. transfer learning sur un réseau pré-entraîné léger)
- [x] Évaluation sur la validation : exactitude, F1 par classe, F1 macro, nombre de paramètres, frugalité, matrice de confusion (`evaluerReseau.m`)
- [x] Génération du JSON de prédictions sur le test (`built/predictions_<essai>_graine<n>.json`) et sauvegarde du réseau entraîné (`built/reseau_<essai>_graine<n>.mat`)
- [ ] Rapport

### Problèmes connus

- Aucun pour l'instant.

---

## 5. Conventions de travail

- Ne **jamais** versionner `food-11/` ni d'autres images ou modèles entraînés (`.mat`).
- Commits explicites : on décrit ce qui a changé dans le message, et si une manipulation est nécessaire pour lancer le code, on l'indique ici dans le README.
- Avant de pousser, vérifier que `CNN_Projet.m` s'exécute depuis la racine du dépôt.

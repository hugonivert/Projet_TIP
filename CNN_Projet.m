%% CNN personnel pour le challenge Food-11
% A lancer depuis la racine du dépôt (dossier Projet_TIP), les chemins étant
% relatifs. Fichiers utilisés : evaluerReseau.m (scores du challenge) et
% makeCourseProgressOutputFcn.m (courbes lissées, reprise du TP CIFAR-10).
close all
clear
clc

%% Réglages de l'essai
% Pour un nouvel essai, on ne modifie que ce bloc. Chaque exécution ajoute
% une ligne dans built/essais.csv avec ces réglages et les scores obtenus.
% Valeurs par défaut : meilleur réglage à ce stade (essai 5D de suiviCNN.txt).
cfg.nomEssai           = "aug_5blocs";
cfg.graineDecoupage    = 42;      % ne jamais changer : même validation pour tout le groupe
cfg.graineEntrainement = 1;       % 1 par défaut ; 2 et 3 pour mesurer la variabilité
cfg.tailleImage        = [128 128];
cfg.filtres            = [16 32 64 96 128];   % un bloc de convolution par valeur
cfg.optimiseur         = "adam";
cfg.tauxApprentissage  = 0.001;   % taux de départ
% Calendrier du taux (mot-clé uniquement) : "none" = taux constant ;
% "cosine" = décroît en douceur jusqu'au taux de départ x 1e-5, atteint à la
% dernière epoch (qui ne met alors presque plus à jour les poids, seulement
% les statistiques de la batch norm)
cfg.calendrierTaux     = "cosine";
cfg.augmentation       = true;    % true : retournements, translations et zooms aléatoires sur le train
cfg.tailleLot          = 128;
cfg.nbEpochs           = 40;      % environ 30 min sur le CPU du Mac
cfg.afficherCourbes    = true;    % false pour enchaîner des essais sans fenêtre graphique

% Valeur inattendue : erreur tout de suite plutôt qu'à la fin du script
% (le journal ne pourrait pas être écrit)
mustBeMember(cfg.calendrierTaux, ["none" "cosine"]);

%% Chargement des données
trainFolder = "food-11/train";
testFolder  = "food-11/test";

imgDataTrain = imageDatastore(trainFolder, ...
    IncludeSubfolders=true, ...
    LabelSource="foldernames");
imgDataTest = imageDatastore(testFolder);

% Séparation de train en 80 % train et 20 % validation. La graine fixe donne
% toujours le même découpage, donc la même validation pour tous les essais.
% Le générateur "twister" est imposé : même tirage même si un autre script a
% changé de générateur dans la session MATLAB.
rng(cfg.graineDecoupage, "twister");
[imgDataTrain, imgDataValidation] = splitEachLabel( ...
    imgDataTrain, 0.8, "randomized");

classNames = categories(imgDataTrain.Labels);
inputSize  = [cfg.tailleImage 3];

% augmentedImageDatastore redimensionne les images et convertit celles en
% niveaux de gris en RGB. La normalisation est faite par la couche d'entrée
% du réseau (centrage sur la moyenne du train, augmenté si cfg.augmentation).
% Augmentation de données (train seulement, si cfg.augmentation) : à chaque
% epoch, chaque image est transformée au hasard, ce qui limite
% l'apprentissage par coeur (variance) sans ajouter de paramètre.
%   - retournement horizontal une fois sur deux (un plat retourné reste le
%     même plat) ;
%   - translation de +/- 40 pixels en x et en y (le plat n'est pas toujours
%     centré). Elle s'applique avant le redimensionnement, donc en pixels de
%     l'image d'origine : environ +/- 8 % pour une image de 512 pixels (la
%     majorité), jusqu'à 14 % pour les plus petites (289 pixels) et
%     proportionnellement moins pour les très grandes ;
%   - zoom entre 0.9 et 1.1 (photos prises de plus ou moins près) : 0.9
%     éloigne l'image et découvre ses bords, 1.1 la rapproche et les coupe.
% Les zones découvertes par la translation et le zoom sont remplies avec la
% couleur moyenne des images du train (FillValue, mesurée sur des images
% non augmentées) et non en noir (valeur par défaut) : la validation et le
% test n'ont jamais de bandes noires, et après le centrage de la couche
% d'entrée ces zones valent environ 0, donc n'apportent aucune information.
% Pas de retournement vertical (images peu naturelles), de rotation (coins
% à remplir, angle peu utile pour des photos de plats) ni de changement de
% couleur (indice clé pour la nourriture).
% Reproductibilité : les transformations sont tirées avec le générateur
% aléatoire de la session, donc fixées par cfg.graineEntrainement (ce ne
% serait plus le cas avec une préparation des données en parallèle).
if cfg.augmentation
    augmenteur = imageDataAugmenter( ...
        RandXReflection=true, ...
        RandXTranslation=[-40 40], ...
        RandYTranslation=[-40 40], ...
        RandScale=[0.9 1.1], ...
        FillValue=[146 117 88]);   % couleur moyenne RGB du train
    augTrain = augmentedImageDatastore(cfg.tailleImage, imgDataTrain, ...
        ColorPreprocessing="gray2rgb", DataAugmentation=augmenteur);
else
    augTrain = augmentedImageDatastore(cfg.tailleImage, imgDataTrain, ColorPreprocessing="gray2rgb");
end
% Train sans augmentation : sert à mesurer l'exactitude train finale sur
% les vraies images, comparable à la validation
augTrainEval  = augmentedImageDatastore(cfg.tailleImage, imgDataTrain, ColorPreprocessing="gray2rgb");
augValidation = augmentedImageDatastore(cfg.tailleImage, imgDataValidation, ColorPreprocessing="gray2rgb");
augTest       = augmentedImageDatastore(cfg.tailleImage, imgDataTest, ColorPreprocessing="gray2rgb");

%% Réseau
% Le bloc du TP CIFAR-10 (convolution 3x3, batch norm, ReLU, max pooling)
% est répété une fois par valeur de cfg.filtres. Le padding "same" garde
% la taille des cartes de caractéristiques à la convolution, ce qui permet
% à chaque pooling de la diviser exactement par 2. Avec [16 32 64 96 128],
% une image 128x128x3 arrive en 4x4x128 à la sortie du dernier bloc.
% Le global average pooling remplace chaque carte de caractéristiques par
% sa moyenne. La couche entièrement connectée n'a alors que 128 x 11 poids
% (contre 63 x 63 x 8 x 11 pour le réseau du TP appliqué à des images
% 128x128), et le nombre de paramètres ne dépend plus de la taille des
% images (191 787 avec [16 32 64 96 128]).
% La couche softmax finale donne les probabilités de chaque classe ; la
% fonction de perte est indiquée séparément dans trainnet.
layers = imageInputLayer(inputSize, Name="input");
for b = 1:numel(cfg.filtres)
    layers = [layers
        convolution2dLayer(3, cfg.filtres(b), Padding="same", Name="conv" + b)
        batchNormalizationLayer(Name="batchnorm" + b)
        reluLayer(Name="relu" + b)
        maxPooling2dLayer(2, Stride=2, Name="maxpool" + b)]; %#ok<AGROW>
end
layers = [layers
    globalAveragePooling2dLayer(Name="gap")
    fullyConnectedLayer(numel(classNames), Name="fullyconnected")
    softmaxLayer(Name="softmax")];

if cfg.afficherCourbes
    % Aperçu de 25 images d'entraînement et de l'architecture du réseau
    perm = randperm(numel(imgDataTrain.Files), 25);
    figure
    montage(imgDataTrain.Files(perm), ThumbnailSize=cfg.tailleImage)
    analyzeNetwork(layers)
end

%% Entraînement
% La validation (20 % du train) est évaluée pendant l'entraînement pour
% suivre sa précision et sa perte. Le test n'a pas d'étiquettes : il ne sert
% qu'aux prédictions (export JSON).
% trainnet ignore le dernier lot incomplet : une epoch compte
% floor(nombre d'images / taille de lot) itérations.
% OutputNetwork="last-iteration" : on récupère le réseau de la dernière
% epoch. Sinon, trainnet renverrait celui de l'epoch où la perte de
% validation est la plus basse : les scores de validation seraient un peu
% optimistes (la validation servirait aussi à choisir l'epoch) et
% l'exactitude train ne correspondrait plus à la fin de l'entraînement,
% ce qui fausserait le diagnostic biais/variance. Le nombre d'epochs se
% règle plutôt à partir des courbes d'un essai à taux constant. Avec
% cfg.calendrierTaux = "cosine", tout le calendrier dépend de cfg.nbEpochs :
% l'état à l'epoch k d'un essai de 20 epochs ne correspond pas à un essai
% de k epochs, donc on compare des essais complets.
iterationsParEpoch = floor(numel(imgDataTrain.Files) / cfg.tailleLot);

if cfg.afficherCourbes
    % Fenêtre de MATLAB + courbes lissées du TP (moyenne glissante sur 10 lots).
    % On lui passe le nombre d'images réellement utilisées par epoch pour que
    % les bandes d'epochs tombent au bon endroit.
    courbes = "training-progress";
    suiviEntrainement = makeCourseProgressOutputFcn( ...
        iterationsParEpoch * cfg.tailleLot, cfg.tailleLot, cfg.nbEpochs, 10);
else
    courbes = "none";
    suiviEntrainement = [];
end

options = trainingOptions(cfg.optimiseur, ...
    InitialLearnRate=cfg.tauxApprentissage, ...
    LearnRateSchedule=cfg.calendrierTaux, ...
    MaxEpochs=cfg.nbEpochs, ...
    Shuffle="every-epoch", ...
    MiniBatchSize=cfg.tailleLot, ...
    ValidationData=augValidation, ...
    ValidationFrequency=iterationsParEpoch, ...   % une validation par epoch
    OutputNetwork="last-iteration", ...
    Verbose=false, ...
    Plots=courbes, ...
    OutputFcn=suiviEntrainement, ...
    Metrics="accuracy");
    %ExecutionEnvironment="gpu");

% La graine d'entraînement fixe les poids de départ et l'ordre des images
rng(cfg.graineEntrainement, "twister");
chrono = tic;
[net, info] = trainnet(augTrain, layers, "crossentropy", options);
dureeMinutes = toc(chrono) / 60;
fprintf("Durée de l'entraînement : %.1f min\n", dureeMinutes);

% Les fichiers produits vont dans le dossier "built" (au même niveau que
% food-11), créé automatiquement s'il n'existe pas encore. Les fichiers
% propres à un essai portent son nom et sa graine (par exemple
% reseau_base_4blocs_graine1.mat) : ils ne sont écrasés que si l'on relance
% le même essai avec la même graine. Changer cfg.nomEssai à chaque nouveau
% réglage.
outputFolder = "built";
if ~isfolder(outputFolder)
    mkdir(outputFolder);
end
suffixeEssai = cfg.nomEssai + "_graine" + cfg.graineEntrainement;

%% Sauvegarde du réseau entraîné
% built/reseau_<essai>_graine<n>.mat contient le réseau, les réglages, les
% noms des classes, ainsi que les fichiers et vraies classes des images de
% validation. On peut ainsi refaire des prédictions, la matrice de
% confusion ou des figures sans réentraîner : load, puis recréer un
% augmentedImageDatastore (cfg.tailleImage, ColorPreprocessing="gray2rgb")
% sur les images voulues, puis minibatchpredict et scores2label.
% Fichier binaire : il reste dans built/, hors du dépôt.
fichiersValidation    = imgDataValidation.Files;
etiquettesValidation  = imgDataValidation.Labels;
fichierReseau = fullfile(outputFolder, "reseau_" + suffixeEssai + ".mat");
save(fichierReseau, "net", "cfg", "classNames", "fichiersValidation", "etiquettesValidation");
fprintf("Réseau entraîné sauvegardé dans %s\n", fichierReseau);

%% Historique de l'entraînement
% Une ligne par epoch dans built/historique_<essai>_graine<n>.csv, pour
% tracer les courbes et faire le diagnostic biais/variance même quand le
% script tourne sans fenêtre graphique. Écrit juste après l'entraînement,
% pour être gardé même si la suite du script plante. Exactitudes entre 0
% et 1, comme dans le journal.
% Côté train (colonnes "Lots") : moyenne des lots de l'epoch, mesurée
% pendant l'apprentissage. Deux effets jouent en sens opposés : la moyenne
% mélange des états successifs du réseau (les premiers lots de l'epoch sont
% vus par un réseau moins entraîné, ce qui la tire vers le bas), et la
% batch norm utilise les statistiques de chaque lot au lieu des moyennes
% globales utilisées en prédiction. Elle peut donc s'écarter, dans un sens
% ou dans l'autre, de l'exactitude train finale calculée plus bas, qui
% reste la référence pour comparer train et validation. Avec
% cfg.augmentation, ces colonnes sont en plus mesurées sur des images
% transformées (tâche plus difficile), donc nettement plus basses que
% l'exactitude train finale, calculée sur les vraies images. Côté validation :
% mesure de fin d'epoch, sur le réseau en mode prédiction.
histoTrain = info.TrainingHistory;
histoVal   = info.ValidationHistory;
[groupes, epoch] = findgroups(histoTrain.Epoch);
perteTrainLots      = splitapply(@mean, histoTrain.Loss, groupes);
exactitudeTrainLots = splitapply(@mean, histoTrain.Accuracy, groupes) / 100;
% Taux d'apprentissage de chaque epoch (montre le calendrier). Il est
% constant sur toute l'epoch, on prend donc celui du premier lot.
tauxEpoch           = splitapply(@(t) t(1), histoTrain.LearnRate, groupes);

% La validation de fin d'epoch a lieu à l'itération epoch x iterationsParEpoch.
% Si l'entraînement a été arrêté en cours d'epoch (bouton Stop), la dernière
% validation a lieu à la dernière itération réellement faite.
iterationVal = min(epoch * iterationsParEpoch, histoTrain.Iteration(end));
[~, rangVal] = ismember(iterationVal, histoVal.Iteration);
perteVal      = NaN(size(epoch));
exactitudeVal = NaN(size(epoch));
trouve = rangVal > 0;
perteVal(trouve)      = histoVal.Loss(rangVal(trouve));
exactitudeVal(trouve) = histoVal.Accuracy(rangVal(trouve)) / 100;

historique = table(epoch, tauxEpoch, perteTrainLots, exactitudeTrainLots, ...
    perteVal, exactitudeVal);
fichierHistorique = fullfile(outputFolder, "historique_" + suffixeEssai + ".csv");
writetable(historique, fichierHistorique);
fprintf("Historique de l'entraînement écrit dans %s\n", fichierHistorique);

%% Évaluation
% Précision sur le train : sert à repérer un sous-apprentissage (biais)
trainScores = minibatchpredict(net, augTrainEval);
trainLabelsPred = scores2label(trainScores, classNames);
trainAccuracy = mean(trainLabelsPred == imgDataTrain.Labels);
fprintf("Train-set accuracy: %.2f%%\n", 100 * trainAccuracy);

% Validation : scores du challenge (F1 macro, paramètres, frugalité)
resVal = evaluerReseau(net, augValidation, imgDataValidation.Labels, classNames);
fprintf("Validation-set accuracy: %.2f%%\n", 100 * resVal.exactitude);
fprintf("Validation F1 macro: %.4f\n", resVal.f1Macro);
fprintf("Paramètres: %d | Frugalité: %.4f\n", resVal.nbParametres, resVal.frugalite);
disp(table(string(classNames), resVal.f1ParClasse, VariableNames=["classe" "F1"]))

if cfg.afficherCourbes
    % Matrice de confusion (code du TP) : quelles classes sont confondues
    figure
    cm = confusionchart(imgDataValidation.Labels, resVal.predites);
    cm.ColumnSummary = "column-normalized";   % précision par classe prédite
    cm.RowSummary    = "row-normalized";      % rappel par vraie classe
    cm.Title = sprintf("Validation : F1 macro %.3f | frugalité %.3f", ...
        resVal.f1Macro, resVal.frugalite);
end

%% Prédictions sur le test
% Les vraies classes ne sont pas fournies (seul le serveur d'évaluation les
% connaît), donc pas de score ici.
testScores = minibatchpredict(net, augTest);
testLabelsPred = scores2label(testScores, classNames);

% Export JSON pour le serveur d'évaluation : { "0": "bread", "1": "rice", ... }
% Clé = nom du fichier sans extension, valeur = classe prédite (= nom du
% dossier de train, déjà en minuscules). On associe via le nom de fichier et
% non la position, car les fichiers sont triés par ordre alphabétique
% (0, 1, 10, 100, ...).
[~, testIds] = fileparts(imgDataTest.Files);
predictions = containers.Map(testIds, cellstr(testLabelsPred));
jsonFile = fullfile(outputFolder, "predictions_" + suffixeEssai + ".json");
writelines(jsonencode(predictions, PrettyPrint=true), jsonFile);
fprintf("Prédictions du test écrites dans %s (%d images)\n", jsonFile, predictions.Count);

%% Journal des essais
% Une ligne par exécution dans built/essais.csv (créé au premier essai)
texteAugmentation = "non";
if cfg.augmentation
    texteAugmentation = "oui";
end
ligne = table(string(datetime("now", Format="yyyy-MM-dd HH:mm")), cfg.nomEssai, ...
    cfg.graineDecoupage, cfg.graineEntrainement, ...
    string(sprintf("%dx%d", cfg.tailleImage(1), cfg.tailleImage(2))), ...
    strjoin(string(cfg.filtres), "-"), cfg.optimiseur, cfg.tauxApprentissage, ...
    cfg.calendrierTaux, texteAugmentation, cfg.tailleLot, cfg.nbEpochs, ...
    resVal.nbParametres, trainAccuracy, resVal.exactitude, resVal.f1Macro, ...
    resVal.frugalite, dureeMinutes, ...
    VariableNames=["date" "essai" "graineDecoupage" "graineEntrainement" ...
    "taille" "filtres" "optimiseur" "tauxApprentissage" "calendrierTaux" "augmentation" "tailleLot" "epochs" ...
    "parametres" "exactitudeTrain" "exactitudeVal" "f1Val" "frugalite" "dureeMin"]);
journal = fullfile(outputFolder, "essais.csv");

% Si les colonnes ont changé depuis la création du journal, on ne mélange
% pas des lignes de formats différents : l'ancien journal est archivé sous
% un nom daté et un nouveau essais.csv est commencé
if isfile(journal)
    lignesJournal = readlines(journal);
    colonnesExistantes = split(lignesJournal(1), ",")';
    if ~isequal(colonnesExistantes, string(ligne.Properties.VariableNames))
        archive = fullfile(outputFolder, "essais_ancien_" + ...
            string(datetime("now", Format="yyyyMMdd_HHmmss")) + ".csv");
        movefile(journal, archive);
        warning("Les colonnes du journal ont changé : l'ancien journal est archivé dans %s.", archive);
    end
end
writetable(ligne, journal, WriteMode="append");
fprintf("Essai ajouté au journal %s\n", journal);

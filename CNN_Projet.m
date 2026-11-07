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
cfg.nomEssai           = "tp_1couche";
cfg.graineDecoupage    = 42;      % ne jamais changer : même validation pour tout le groupe
cfg.graineEntrainement = 1;       % 1 par défaut ; 2 et 3 pour mesurer la variabilité
cfg.tailleImage        = [128 128];
cfg.optimiseur         = "sgdm";
cfg.tauxApprentissage  = 0.001;
cfg.tailleLot          = 500;
cfg.nbEpochs           = 16;
cfg.afficherCourbes    = true;    % false pour enchaîner des essais sans fenêtre graphique

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
% du réseau (centrage sur la moyenne du train).
augTrain      = augmentedImageDatastore(cfg.tailleImage, imgDataTrain, ColorPreprocessing="gray2rgb");
augValidation = augmentedImageDatastore(cfg.tailleImage, imgDataValidation, ColorPreprocessing="gray2rgb");
augTest       = augmentedImageDatastore(cfg.tailleImage, imgDataTest, ColorPreprocessing="gray2rgb");

%% Réseau
% Réseau de départ du TP CIFAR-10 (une couche de convolution). La couche
% softmax finale donne les probabilités de chaque classe ; la fonction de
% perte est indiquée séparément dans trainnet.
layers = [
    imageInputLayer(inputSize, Name="input")
    convolution2dLayer(3, 8, Name="conv")
    batchNormalizationLayer(Name="batchnorm")
    reluLayer(Name="relu")
    maxPooling2dLayer(2, Stride=2, Name="maxpooling")
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
    MaxEpochs=cfg.nbEpochs, ...
    Shuffle="every-epoch", ...
    MiniBatchSize=cfg.tailleLot, ...
    ValidationData=augValidation, ...
    ValidationFrequency=iterationsParEpoch, ...   % une validation par epoch
    Verbose=false, ...
    Plots=courbes, ...
    OutputFcn=suiviEntrainement, ...
    Metrics="accuracy");
    %ExecutionEnvironment="gpu");

% La graine d'entraînement fixe les poids de départ et l'ordre des images
rng(cfg.graineEntrainement, "twister");
chrono = tic;
net = trainnet(augTrain, layers, "crossentropy", options);
dureeMinutes = toc(chrono) / 60;
fprintf("Durée de l'entraînement : %.1f min\n", dureeMinutes);

%% Évaluation
% Précision sur le train : sert à repérer un sous-apprentissage (biais)
trainScores = minibatchpredict(net, augTrain);
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

% Les fichiers produits vont dans le dossier "built" (au même niveau que
% food-11), créé automatiquement s'il n'existe pas encore
outputFolder = "built";
if ~isfolder(outputFolder)
    mkdir(outputFolder);
end

% Export JSON pour le serveur d'évaluation : { "0": "bread", "1": "rice", ... }
% Clé = nom du fichier sans extension, valeur = classe prédite (= nom du
% dossier de train, déjà en minuscules). On associe via le nom de fichier et
% non la position, car les fichiers sont triés par ordre alphabétique
% (0, 1, 10, 100, ...).
[~, testIds] = fileparts(imgDataTest.Files);
predictions = containers.Map(testIds, cellstr(testLabelsPred));
jsonFile = fullfile(outputFolder, "predictions.json");
writelines(jsonencode(predictions, PrettyPrint=true), jsonFile);
fprintf("Prédictions du test écrites dans %s (%d images)\n", jsonFile, predictions.Count);

%% Journal des essais
% Une ligne par exécution dans built/essais.csv (créé au premier essai)
ligne = table(string(datetime("now", Format="yyyy-MM-dd HH:mm")), cfg.nomEssai, ...
    cfg.graineDecoupage, cfg.graineEntrainement, ...
    string(sprintf("%dx%d", cfg.tailleImage(1), cfg.tailleImage(2))), ...
    cfg.optimiseur, cfg.tauxApprentissage, cfg.tailleLot, cfg.nbEpochs, ...
    resVal.nbParametres, trainAccuracy, resVal.exactitude, resVal.f1Macro, ...
    resVal.frugalite, dureeMinutes, ...
    VariableNames=["date" "essai" "graineDecoupage" "graineEntrainement" ...
    "taille" "optimiseur" "tauxApprentissage" "tailleLot" "epochs" ...
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

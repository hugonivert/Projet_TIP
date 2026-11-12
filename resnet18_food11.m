%% ResNet-18 pré-entraîné pour le challenge Food-11 (transfer learning)
% A lancer depuis la racine du dépôt (dossier Projet_TIP), les chemins étant
% relatifs. Fichiers utilisés : evaluerReseau.m (scores du challenge) et
% makeCourseProgressOutputFcn.m (courbes lissées, reprise du TP CIFAR-10).
% Script de départ : resnet34_food11.m (Nathan), repris pour que le
% ResNet-18 soit évalué exactement comme le CNN personnel (CNN_Projet.m) :
% même découpage train/validation, même augmentation, mêmes scores.
%
% Transfer learning : on part d'un ResNet-18 déjà entraîné sur ImageNet
% (1,2 million d'images, 1000 classes). Ses couches ont appris à repérer
% des bords, des textures et des formes qui servent aussi pour d'autres
% images. Sa dernière couche (1000 classes) est remplacée par une couche à
% 11 classes, puis tout le réseau est réentraîné avec un taux
% d'apprentissage faible (fine-tuning) : il s'adapte à Food-11 sans
% effacer ce qu'il a appris sur ImageNet.
%
% Prérequis : MATLAB R2024b ou plus récent (testé avec R2026b), Deep
% Learning Toolbox et module "Deep Learning Toolbox Model for ResNet-18
% Network" (menu Home, Add-Ons). Le GPU est utilisé automatiquement s'il est disponible
% (Parallel Computing Toolbox et carte NVIDIA : canUseGPU renvoie 1).
close all
clear
clc

%% Réglages de l'essai
% Pour un nouvel essai, on ne modifie que ce bloc. Chaque exécution complète
% ajoute une ligne dans built/essais_resnet18.csv avec ces réglages et les
% scores obtenus. La taille des images n'est pas un réglage : elle est
% imposée par le réseau pré-entraîné (224x224).
cfg.nomEssai           = "resnet18_base_15ep";
cfg.graineDecoupage    = 42;      % ne jamais changer : même validation pour tout le groupe
cfg.graineEntrainement = 1;       % 1 par défaut ; 2 et 3 pour mesurer la variabilité
cfg.optimiseur         = "adam";
cfg.tauxApprentissage  = 3e-4;    % faible : on ajuste des poids déjà bons
% Calendrier du taux (mot-clé uniquement) : "none" = taux constant ;
% "cosine" = décroît en douceur jusqu'au taux de départ x 1e-5, atteint à la
% dernière epoch (même calendrier que le CNN personnel)
cfg.calendrierTaux     = "cosine";
% Augmentation du train : "aucune" ; "base" = retournement horizontal,
% translations et zoom (mêmes réglages que le CNN personnel)
cfg.augmentation       = "base";
cfg.tailleLot          = 64;      % 32 si la mémoire du GPU ne suffit pas (noté dans le journal)
cfg.nbEpochs           = 15;
% Test de bon fonctionnement : true = 20 images par classe pour le train et
% la validation, 50 images de test, 2 epochs. Moins d'une minute, même sans
% GPU. Les fichiers produits finissent par "_rapide" et le journal n'est
% pas modifié. Les scores obtenus n'ont aucune valeur.
cfg.essaiRapide        = false;
cfg.afficherCourbes    = true;    % false pour enchaîner des essais sans fenêtre graphique

% Valeur inattendue : erreur tout de suite plutôt qu'à la fin du script
% (le journal ne pourrait pas être écrit). Les réglages texte doivent être
% un seul texte ; on les convertit en string (un texte entre apostrophes,
% de type char, ferait échouer l'écriture du journal).
for champ = ["nomEssai" "optimiseur" "calendrierTaux" "augmentation"]
    mustBeTextScalar(cfg.(champ));
    cfg.(champ) = string(cfg.(champ));
end
mustBeMember(cfg.calendrierTaux, ["none" "cosine"]);
mustBeMember(cfg.augmentation, ["aucune" "base"]);
for champ = ["essaiRapide" "afficherCourbes"]
    validateattributes(cfg.(champ), ["logical" "numeric"], "scalar");
    cfg.(champ) = logical(cfg.(champ));
end

%% Chargement des données
trainFolder = "food-11/train";
testFolder  = "food-11/test";

imgDataTrain = imageDatastore(trainFolder, ...
    IncludeSubfolders=true, ...
    LabelSource="foldernames");
imgDataTest = imageDatastore(testFolder);

% Séparation de train en 80 % train et 20 % validation, exactement comme
% dans CNN_Projet.m : même graine, même générateur, donc même validation.
rng(cfg.graineDecoupage, "twister");
[imgDataTrain, imgDataValidation] = splitEachLabel( ...
    imgDataTrain, 0.8, "randomized");

classNames = categories(imgDataTrain.Labels);

%% Empreinte de la validation
% Pour comparer les scores de ce réseau à ceux du CNN personnel, les deux
% doivent être évalués sur les mêmes images. Une autre version de MATLAB
% pourrait découper autrement malgré la même graine : on écrit donc la
% liste des images de validation (chemin "classe/fichier.jpg", triée) dans
% built/validation_fichiers.txt. Deux fichiers identiques prouvent que la
% validation est la même. Le nombre d'images par classe est aussi affiché.
% Toujours calculée sur la validation complète, même en essai rapide.
outputFolder = "built";
if ~isfolder(outputFolder)
    mkdir(outputFolder);
end
[dossiersValidation, nomsValidation, extensionsValidation] = fileparts(imgDataValidation.Files);
[~, classesValidation] = fileparts(dossiersValidation);
listeValidation = sort(string(classesValidation) + "/" + string(nomsValidation) ...
    + string(extensionsValidation));
fichierEmpreinte = fullfile(outputFolder, "validation_fichiers.txt");
writelines(listeValidation, fichierEmpreinte, LineEnding="\n");   % même fichier sous Windows, macOS et Linux
disp(table(string(classNames), countcats(imgDataValidation.Labels), ...
    VariableNames=["classe" "imagesValidation"]))
fprintf("Validation : %d images, liste écrite dans %s\n", ...
    numel(listeValidation), fichierEmpreinte);

% Essai rapide : on garde les 20 premières images de chaque classe (train
% et validation) et les 50 premières images de test
if cfg.essaiRapide
    imgDataTrain      = splitEachLabel(imgDataTrain, 20);
    imgDataValidation = splitEachLabel(imgDataValidation, 20);
    imgDataTest       = subset(imgDataTest, 1:50);
    cfg.nbEpochs      = 2;   % au moins 2 : avec "cosine" et 1 seule epoch, le taux serait quasi nul
    cfg.nomEssai      = cfg.nomEssai + "_rapide";
end

%% Réseau
% imagePretrainedNetwork charge le ResNet-18 entraîné sur ImageNet et
% remplace sa dernière couche entièrement connectée (1000 sorties) par une
% couche à 11 sorties, aux poids tirés au hasard. Toutes les autres couches
% gardent leurs poids ImageNet. Le réseau compte 11 186 955 paramètres,
% soit une pénalité de frugalité de 0,275 (contre 0,048 pour le CNN
% personnel).
% ResNet ("Residual Network") : une suite de blocs de 2 convolutions 3x3,
% chacun doublé d'un raccourci qui ajoute son entrée à sa sortie (après une
% convolution 1x1 pour 3 des 8 raccourcis, quand la taille change), ce qui
% permet d'entraîner des réseaux profonds. Le "18" compte les couches à
% poids du chemin principal : 17 convolutions et la couche entièrement
% connectée. MATLAB ne fournit pas de ResNet-34 pré-entraîné (choix de départ
% de Nathan), et un ResNet-34 entraîné de zéro (environ 21 millions de
% paramètres pour 10 600 images) serait trop pénalisé par la frugalité.
% Les poids de départ de la nouvelle couche sont tirés au moment du
% chargement : la graine d'entraînement est donc fixée juste avant, comme
% pour tous les poids de départ du CNN personnel.
rng(cfg.graineEntrainement, "twister");
net = imagePretrainedNetwork("resnet18", NumClasses=numel(classNames));
inputSize   = net.Layers(1).InputSize;   % [224 224 3]
tailleImage = inputSize(1:2);
% La fonction de perte "crossentropy" attend des probabilités en sortie
assert(isa(net.Layers(end), "nnet.cnn.layer.SoftmaxLayer"), ...
    "La dernière couche du ResNet-18 devrait être un softmax.");

%% Préparation des images
% augmentedImageDatastore redimensionne les images en 224x224 et convertit
% celles en niveaux de gris en RGB. Les pixels restent entre 0 et 255 :
% c'est ce qu'attend la couche d'entrée du ResNet-18, qui les normalise
% elle-même avec la moyenne et l'écart type d'ImageNet.
% Augmentation "base" (train seulement) : mêmes réglages que dans
% CNN_Projet.m, où leur choix est expliqué (retournement horizontal une fois
% sur deux, translation de +/- 40 pixels de l'image d'origine, zoom entre
% 0.9 et 1.1, zones découvertes remplies avec la couleur moyenne du train).
% Les transformations sont tirées avec le générateur aléatoire de la
% session, donc fixées par cfg.graineEntrainement.
if cfg.augmentation == "base"
    augmenteur = imageDataAugmenter( ...
        RandXReflection=true, ...
        RandXTranslation=[-40 40], ...
        RandYTranslation=[-40 40], ...
        RandScale=[0.9 1.1], ...
        FillValue=[146 117 88]);   % couleur moyenne RGB du train
    augTrain = augmentedImageDatastore(tailleImage, imgDataTrain, ...
        ColorPreprocessing="gray2rgb", DataAugmentation=augmenteur);
else
    augTrain = augmentedImageDatastore(tailleImage, imgDataTrain, ColorPreprocessing="gray2rgb");
end
% Train sans augmentation : sert à mesurer l'exactitude train finale sur
% les vraies images, comparable à la validation
augTrainEval  = augmentedImageDatastore(tailleImage, imgDataTrain, ColorPreprocessing="gray2rgb");
augValidation = augmentedImageDatastore(tailleImage, imgDataValidation, ColorPreprocessing="gray2rgb");
augTest       = augmentedImageDatastore(tailleImage, imgDataTest, ColorPreprocessing="gray2rgb");

%% Entraînement
% Mêmes choix que pour le CNN personnel (voir CNN_Projet.m) : une
% validation par epoch, réseau de la dernière epoch (OutputNetwork =
% "last-iteration") pour ne pas utiliser la validation à choisir l'epoch.
% ResetInputNormalization=false : on garde la normalisation d'ImageNet de
% la couche d'entrée, celle avec laquelle les poids pré-entraînés ont été
% appris (sinon trainnet la recalculerait sur les images de Food-11).
% Le GPU est utilisé automatiquement s'il est disponible.
iterationsParEpoch = floor(numel(imgDataTrain.Files) / cfg.tailleLot);

if cfg.afficherCourbes
    % Fenêtre de MATLAB + courbes lissées du TP (moyenne glissante sur 10 lots)
    courbes = "training-progress";
    suiviEntrainement = makeCourseProgressOutputFcn( ...
        iterationsParEpoch * cfg.tailleLot, cfg.tailleLot, cfg.nbEpochs, 10);
else
    courbes = "none";
    suiviEntrainement = [];
end

options = trainingOptions(cfg.optimiseur, ...
    L2Regularization=1e-4, ...
    InitialLearnRate=cfg.tauxApprentissage, ...
    LearnRateSchedule=cfg.calendrierTaux, ...
    MaxEpochs=cfg.nbEpochs, ...
    Shuffle="every-epoch", ...
    MiniBatchSize=cfg.tailleLot, ...
    ValidationData=augValidation, ...
    ValidationFrequency=iterationsParEpoch, ...   % une validation par epoch
    OutputNetwork="last-iteration", ...
    ResetInputNormalization=false, ...
    Verbose=false, ...
    Plots=courbes, ...
    OutputFcn=suiviEntrainement, ...
    Metrics="accuracy");

% Machine utilisée, notée dans le journal : les durées ne se comparent
% qu'entre essais faits sur la même machine. Sur GPU, deux exécutions
% identiques peuvent donner des résultats très légèrement différents.
if canUseGPU
    carteGraphique = gpuDevice;
    machine = "gpu " + string(carteGraphique.Name);
else
    machine = "cpu";
end
versionMatlab = string(version("-release"));
fprintf("Machine : %s, MATLAB %s\n", machine, versionMatlab);

% La graine d'entraînement est fixée à nouveau ici : l'ordre des images et
% les transformations de l'augmentation partent d'un état connu
rng(cfg.graineEntrainement, "twister");
chrono = tic;
[net, info] = trainnet(augTrain, net, "crossentropy", options);
dureeMinutes = toc(chrono) / 60;
fprintf("Durée de l'entraînement : %.1f min\n", dureeMinutes);

% Fichiers propres à l'essai : nom de l'essai et graine (par exemple
% reseau_resnet18_base_15ep_graine1.mat), dans le dossier built
suffixeEssai = cfg.nomEssai + "_graine" + cfg.graineEntrainement;

%% Sauvegarde du réseau entraîné
% Même contenu que pour le CNN personnel : réseau, réglages, noms des
% classes, fichiers et vraies classes de la validation. Fichier binaire
% (environ 45 Mo) : il reste dans built/, hors du dépôt.
fichiersValidation   = imgDataValidation.Files;
etiquettesValidation = imgDataValidation.Labels;
fichierReseau = fullfile(outputFolder, "reseau_" + suffixeEssai + ".mat");
save(fichierReseau, "net", "cfg", "classNames", "fichiersValidation", "etiquettesValidation");
fprintf("Réseau entraîné sauvegardé dans %s\n", fichierReseau);

%% Historique de l'entraînement
% Une ligne par epoch dans built/historique_<essai>_graine<n>.csv, même
% format que pour le CNN personnel (voir CNN_Projet.m pour la lecture des
% colonnes "Lots", mesurées pendant l'apprentissage sur des images
% augmentées). Exactitudes entre 0 et 1.
histoTrain = info.TrainingHistory;
histoVal   = info.ValidationHistory;
[groupes, epoch] = findgroups(histoTrain.Epoch);
perteTrainLots      = splitapply(@mean, histoTrain.Loss, groupes);
exactitudeTrainLots = splitapply(@mean, histoTrain.Accuracy, groupes) / 100;
tauxEpoch           = splitapply(@(t) t(1), histoTrain.LearnRate, groupes);

% La validation de fin d'epoch a lieu à l'itération epoch x iterationsParEpoch
% (ou à la dernière itération faite si l'entraînement a été arrêté)
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

% Matrice de confusion de la validation en CSV (lignes = vraies classes,
% colonnes = classes prédites) : permet l'analyse par classe et la figure
% du rapport sans avoir besoin du réseau sauvegardé
matriceConfusion = array2table(resVal.matriceConfusion, ...
    VariableNames=classNames, RowNames=classNames);
matriceConfusion.Properties.DimensionNames{1} = 'vraieClasse';   % titre de la 1re colonne
fichierConfusion = fullfile(outputFolder, "confusion_" + suffixeEssai + ".csv");
writetable(matriceConfusion, fichierConfusion, WriteRowNames=true);
fprintf("Matrice de confusion écrite dans %s\n", fichierConfusion);

if cfg.afficherCourbes
    % Matrice de confusion (code du TP) : quelles classes sont confondues
    figure
    cm = confusionchart(imgDataValidation.Labels, resVal.predites);
    cm.ColumnSummary = "column-normalized";   % précision par classe prédite
    cm.RowSummary    = "row-normalized";      % rappel par vraie classe
    cm.Title = sprintf("ResNet-18, validation : F1 macro %.3f | frugalité %.3f", ...
        resVal.f1Macro, resVal.frugalite);
end

%% Prédictions sur le test
% Export JSON pour le serveur d'évaluation : { "0": "bread", "1": "rice", ... }
% Clé = nom du fichier sans extension, valeur = classe prédite.
testScores = minibatchpredict(net, augTest);
testLabelsPred = scores2label(testScores, classNames);
[~, testIds] = fileparts(imgDataTest.Files);
predictions = containers.Map(testIds, cellstr(testLabelsPred));
jsonFile = fullfile(outputFolder, "predictions_" + suffixeEssai + ".json");
writelines(jsonencode(predictions, PrettyPrint=true), jsonFile);
fprintf("Prédictions du test écrites dans %s (%d images)\n", jsonFile, predictions.Count);

%% Journal des essais
% Une ligne par exécution complète dans built/essais_resnet18.csv, séparé
% de built/essais.csv (CNN personnel) car les colonnes diffèrent. Les
% essais rapides n'y sont pas ajoutés.
if cfg.essaiRapide
    fprintf("Essai rapide : journal non modifié.\n");
else
    ligne = table(string(datetime("now", Format="yyyy-MM-dd HH:mm")), cfg.nomEssai, ...
        cfg.graineDecoupage, cfg.graineEntrainement, ...
        string(sprintf("%dx%d", tailleImage(1), tailleImage(2))), ...
        cfg.optimiseur, cfg.tauxApprentissage, cfg.calendrierTaux, cfg.augmentation, ...
        cfg.tailleLot, cfg.nbEpochs, ...
        resVal.nbParametres, trainAccuracy, resVal.exactitude, resVal.f1Macro, ...
        resVal.frugalite, dureeMinutes, machine, versionMatlab, ...
        VariableNames=["date" "essai" "graineDecoupage" "graineEntrainement" ...
        "taille" "optimiseur" "tauxApprentissage" "calendrierTaux" "augmentation" ...
        "tailleLot" "epochs" "parametres" "exactitudeTrain" "exactitudeVal" ...
        "f1Val" "frugalite" "dureeMin" "machine" "versionMatlab"]);
    journal = fullfile(outputFolder, "essais_resnet18.csv");

    % Si les colonnes ont changé depuis la création du journal, l'ancien
    % journal est archivé sous un nom daté et un nouveau est commencé
    if isfile(journal)
        lignesJournal = readlines(journal);
        colonnesExistantes = split(lignesJournal(1), ",")';
        if ~isequal(colonnesExistantes, string(ligne.Properties.VariableNames))
            archive = fullfile(outputFolder, "essais_resnet18_ancien_" + ...
                string(datetime("now", Format="yyyyMMdd_HHmmss")) + ".csv");
            movefile(journal, archive);
            warning("Les colonnes du journal ont changé : l'ancien journal est archivé dans %s.", archive);
        end
    end
    writetable(ligne, journal, WriteMode="append");
    fprintf("Essai ajouté au journal %s\n", journal);
end

%% SqueezeNet sur Food-11 : matrice de confusion + frugalité
% Script de Nathan, complété pour que le SqueezeNet soit évalué exactement
% comme le CNN personnel (CNN_Projet.m) et le ResNet-18
% (resnet18_food11.m) : même découpage train/validation, même
% augmentation, mêmes scores (evaluerReseau.m), mêmes fichiers de sortie.
% A lancer depuis la racine du dépôt (dossier Projet_TIP).
close all
clear
clc

%% ---------------- Paramètres ----------------
% SqueezeNet pré-entraîné sur ImageNet (transfer learning)

trainFolder   = "food-11/train";
testFolder    = "food-11/test";
inputSize     = [227 227 3];   % taille d'entrée native de SqueezeNet
miniBatchSize = 64;
maxEpochs     = 20;
learnRate     = 3e-4;

% Réglages ajoutés pour comparer les réseaux à égalité. Chaque exécution
% complète ajoute une ligne dans built/essais_squeezenet.csv.
nomEssai           = "squeezenet_base_20ep";   % sert à nommer les fichiers produits
graineDecoupage    = 42;   % ne jamais changer : même validation pour tout le groupe
graineEntrainement = 1;    % 1 par défaut ; 2 et 3 pour mesurer la variabilité
% Calendrier du taux : "none" = taux constant ; "cosine" = décroît en
% douceur jusqu'au taux de départ x 1e-5, atteint à la dernière epoch
% (même calendrier que les autres réseaux)
calendrierTaux     = "cosine";
% Augmentation du train : "aucune" ou "base" (mêmes réglages que les
% autres réseaux : retournement horizontal, translations et zoom)
augmentation       = "base";
% Test de bon fonctionnement : true = 20 images par classe pour le train et
% la validation, 50 images de test, 2 epochs. Les fichiers produits
% finissent par "_rapide" et le journal n'est pas modifié. Les scores
% obtenus n'ont aucune valeur.
essaiRapide        = false;
afficherCourbes    = true;   % false pour lancer le script sans fenêtre graphique

mustBeMember(calendrierTaux, ["none" "cosine"]);
mustBeMember(augmentation, ["aucune" "base"]);

%% ---------------- Données ----------------
% Pas de fonction de lecture personnalisée (ReadFcn) : la lecture est bien
% plus rapide (environ 9 fois, mesuré sur le CNN personnel), et l'augmentation s'applique sur l'image
% d'origine, comme pour les autres réseaux. augmentedImageDatastore
% redimensionne en 227x227 et convertit les images en niveaux de gris en
% RGB. Les pixels restent entre 0 et 255, ce qu'attend la couche d'entrée
% de SqueezeNet (elle soustrait elle-même la moyenne d'ImageNet).
imgDataTrain = imageDatastore(trainFolder, ...
    IncludeSubfolders=true, ...
    LabelSource="foldernames");
imgDataTest = imageDatastore(testFolder);

% Graine fixe juste avant le découpage, comme dans CNN_Projet.m : même
% validation que les autres réseaux, à chaque lancement
rng(graineDecoupage, "twister");
[imgDataTrain, imgDataValidation] = splitEachLabel(imgDataTrain, 0.8, "randomized");

classNames = categories(imgDataTrain.Labels);
numClasses = numel(classNames);

% Empreinte de la validation : liste triée des images de validation
% ("classe/fichier.jpg") dans built/validation_fichiers.txt. Elle doit être
% identique à celle du ResNet-18 : c'est la preuve que les réseaux sont
% comparés sur les mêmes images. Toujours calculée sur la validation
% complète, même en essai rapide.
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

% Essai rapide : les 20 premières images de chaque classe (train et
% validation) et les 50 premières images de test
if essaiRapide
    imgDataTrain      = splitEachLabel(imgDataTrain, 20);
    imgDataValidation = splitEachLabel(imgDataValidation, 20);
    imgDataTest       = subset(imgDataTest, 1:50);
    maxEpochs         = 2;   % au moins 2 : avec "cosine" et 1 seule epoch, le taux serait quasi nul
    nomEssai          = nomEssai + "_rapide";
end

% Augmentation "base" (train seulement), mêmes réglages que dans
% CNN_Projet.m, où leur choix est expliqué : retournement horizontal une
% fois sur deux, translation de +/- 40 pixels de l'image d'origine, zoom
% entre 0.9 et 1.1, zones découvertes remplies avec la couleur moyenne du
% train. Transformations tirées avec le générateur aléatoire de la
% session, donc fixées par graineEntrainement.
if augmentation == "base"
    augmenteur = imageDataAugmenter( ...
        RandXReflection=true, ...
        RandXTranslation=[-40 40], ...
        RandYTranslation=[-40 40], ...
        RandScale=[0.9 1.1], ...
        FillValue=[146 117 88]);   % couleur moyenne RGB du train
    augTrain = augmentedImageDatastore(inputSize(1:2), imgDataTrain, ...
        ColorPreprocessing="gray2rgb", DataAugmentation=augmenteur);
else
    augTrain = augmentedImageDatastore(inputSize(1:2), imgDataTrain, ColorPreprocessing="gray2rgb");
end
% Train sans augmentation : sert à mesurer l'exactitude train finale sur
% les vraies images, comparable à la validation
augTrainEval  = augmentedImageDatastore(inputSize(1:2), imgDataTrain, ColorPreprocessing="gray2rgb");
augValidation = augmentedImageDatastore(inputSize(1:2), imgDataValidation, ColorPreprocessing="gray2rgb");
augTest       = augmentedImageDatastore(inputSize(1:2), imgDataTest, ColorPreprocessing="gray2rgb");

%% ---------------- Réseau ----------------
% Les poids de départ de la nouvelle couche conv10 sont tirés à la
% construction du réseau : la graine d'entraînement est fixée juste avant
rng(graineEntrainement, "twister");
% SqueezeNet pré-entraîné (DAGNetwork) -> on remplace la tête de classification
lgraph = layerGraph(squeezenet);
lgraph = removeLayers(lgraph, "ClassificationLayer_predictions");
lgraph = replaceLayer(lgraph, "conv10", ...
    convolution2dLayer(1, numClasses, Name="conv10", ...
        WeightLearnRateFactor=10, BiasLearnRateFactor=10));
net = dlnetwork(lgraph);
modelName = "SqueezeNet pretrained";

% Sécurité : crossentropy attend des probabilités -> softmax en sortie
if ~isa(net.Layers(end), "nnet.cnn.layer.SoftmaxLayer")
    lastName = net.Layers(end).Name;
    net = addLayers(net, softmaxLayer(Name="softmax_out"));
    net = connectLayers(net, lastName, "softmax_out");
end

% Aplatir la sortie 1x1xK en vecteur K (sortie N x K pour minibatchpredict)
lastName = net.Layers(end).Name;
net = addLayers(net, flattenLayer(Name="flatten_out"));
net = connectLayers(net, lastName, "flatten_out");

% addLayers rend le réseau non initialisé : on l'initialise avant l'entraînement
net = initialize(net);

%% ---------------- Entraînement ----------------
% Mêmes choix que pour les autres réseaux :
% - une validation par epoch (historique avec une ligne par epoch) ;
% - OutputNetwork="last-iteration" : réseau de la dernière epoch, et non
%   celui de la meilleure epoch sur la validation (sinon les scores de
%   validation seraient un peu optimistes) ;
% - ResetInputNormalization=false : on garde la moyenne d'ImageNet de la
%   couche d'entrée, celle avec laquelle les poids ont été appris (sinon
%   trainnet la recalculerait sur Food-11).
% Le GPU est utilisé automatiquement s'il est disponible. Sur GPU, deux
% exécutions identiques peuvent donner des résultats très légèrement
% différents.
iterationsParEpoch = floor(numel(imgDataTrain.Files) / miniBatchSize);
if afficherCourbes
    courbes = "training-progress";
else
    courbes = "none";
end

options = trainingOptions("adam", ...
    L2Regularization=1e-4, ...
    InitialLearnRate=learnRate, ...
    LearnRateSchedule=calendrierTaux, ...
    MaxEpochs=maxEpochs, ...
    MiniBatchSize=miniBatchSize, ...
    Shuffle="every-epoch", ...
    ValidationData=augValidation, ...
    ValidationFrequency=iterationsParEpoch, ...   % une validation par epoch
    OutputNetwork="last-iteration", ...
    ResetInputNormalization=false, ...
    Verbose=false, ...
    Plots=courbes, ...
    Metrics="accuracy");

% Machine utilisée, notée dans le journal : les durées ne se comparent
% qu'entre essais faits sur la même machine
if canUseGPU
    carteGraphique = gpuDevice;
    machine = "gpu " + string(carteGraphique.Name);
else
    machine = "cpu";
end
versionMatlab = string(version("-release"));
fprintf("Machine : %s, MATLAB %s\n", machine, versionMatlab);

% La graine d'entraînement est fixée à nouveau : l'ordre des images et les
% transformations de l'augmentation partent d'un état connu
rng(graineEntrainement, "twister");
chrono = tic;
[net, info] = trainnet(augTrain, net, "crossentropy", options);
dureeMinutes = toc(chrono) / 60;
fprintf("Durée de l'entraînement : %.1f min\n", dureeMinutes);

% Fichiers propres à l'essai : nom de l'essai et graine (par exemple
% reseau_squeezenet_base_20ep_graine1.mat), dans le dossier built
suffixeEssai = nomEssai + "_graine" + graineEntrainement;

% Réseau entraîné (fichier binaire, environ 3 Mo, hors du dépôt) avec les
% réglages, les noms des classes, et les fichiers et vraies classes de la
% validation
reglages = struct(nomEssai=nomEssai, graineDecoupage=graineDecoupage, ...
    graineEntrainement=graineEntrainement, learnRate=learnRate, ...
    calendrierTaux=calendrierTaux, augmentation=augmentation, ...
    miniBatchSize=miniBatchSize, maxEpochs=maxEpochs, inputSize=inputSize);
fichiersValidation   = imgDataValidation.Files;
etiquettesValidation = imgDataValidation.Labels;
fichierReseau = fullfile(outputFolder, "reseau_" + suffixeEssai + ".mat");
save(fichierReseau, "net", "reglages", "classNames", "fichiersValidation", "etiquettesValidation");
fprintf("Réseau entraîné sauvegardé dans %s\n", fichierReseau);

% Historique : une ligne par epoch dans built/historique_<essai>_graine<n>.csv,
% même format que pour les autres réseaux (voir CNN_Projet.m pour la
% lecture des colonnes "Lots", mesurées pendant l'apprentissage sur des
% images augmentées). Exactitudes entre 0 et 1.
histoTrain = info.TrainingHistory;
histoVal   = info.ValidationHistory;
[groupes, epoch] = findgroups(histoTrain.Epoch);
perteTrainLots      = splitapply(@mean, histoTrain.Loss, groupes);
exactitudeTrainLots = splitapply(@mean, histoTrain.Accuracy, groupes) / 100;
tauxEpoch           = splitapply(@(t) t(1), histoTrain.LearnRate, groupes);
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

%}
%% ---------------- Évaluation (evaluerReseau) ----------------
% Exactitude sur le train (images non augmentées) : sert à repérer un
% sous-apprentissage (biais) ou un écart train/validation (variance)
trainLabelsPred = scores2label(minibatchpredict(net, augTrainEval), classNames);
trainAccuracy = mean(trainLabelsPred == imgDataTrain.Labels);

YVal = imgDataValidation.Labels;
res  = evaluerReseau(net, augValidation, YVal, classNames);

fprintf("%s\n", modelName);
fprintf("  Exactitude train      : %.2f %%\n", 100 * trainAccuracy);
fprintf("  Exactitude (accuracy) : %.2f %%\n", 100 * res.exactitude);
fprintf("  F1 macro              : %.4f\n", res.f1Macro);
fprintf("  Paramètres            : %d (%.3f M)\n", res.nbParametres, res.nbParametres / 1e6);
fprintf("  Frugalité             : %.4f\n", res.frugalite);
disp(table(string(classNames), res.f1ParClasse, VariableNames=["classe" "F1"]))

%% ---------------- Matrice de confusion ----------------
% En CSV (lignes = vraies classes, colonnes = classes prédites), pour
% l'analyse par classe et la figure du rapport
matriceConfusion = array2table(res.matriceConfusion, ...
    VariableNames=classNames, RowNames=classNames);
matriceConfusion.Properties.DimensionNames{1} = 'vraieClasse';   % titre de la 1re colonne
fichierConfusion = fullfile(outputFolder, "confusion_" + suffixeEssai + ".csv");
writetable(matriceConfusion, fichierConfusion, WriteRowNames=true);
fprintf("Matrice de confusion écrite dans %s\n", fichierConfusion);

if afficherCourbes
    figure
    confusionchart(res.matriceConfusion, classNames, ...
        Title=sprintf("%s - matrice de confusion (validation)", modelName), ...
        RowSummary="row-normalized", ...
        ColumnSummary="column-normalized");
end

%% ---------------- Prédictions test (JSON) ----------------
testScores     = minibatchpredict(net, augTest);
testLabelsPred = scores2label(testScores, classNames);

[~, testIds] = fileparts(imgDataTest.Files);
predictions = containers.Map(testIds, cellstr(testLabelsPred));

jsonFile = fullfile(outputFolder, "predictions_" + suffixeEssai + ".json");
writelines(jsonencode(predictions, PrettyPrint=true), jsonFile);
fprintf("Prédictions du test écrites dans %s (%d images)\n", jsonFile, predictions.Count);

%% ---------------- Journal des essais ----------------
% Une ligne par exécution complète dans built/essais_squeezenet.csv, mêmes
% colonnes que built/essais_resnet18.csv. Les essais rapides n'y sont pas
% ajoutés.
if essaiRapide
    fprintf("Essai rapide : journal non modifié.\n");
else
    ligne = table(string(datetime("now", Format="yyyy-MM-dd HH:mm")), nomEssai, ...
        graineDecoupage, graineEntrainement, ...
        string(sprintf("%dx%d", inputSize(1), inputSize(2))), ...
        "adam", learnRate, calendrierTaux, augmentation, ...
        miniBatchSize, maxEpochs, ...
        res.nbParametres, trainAccuracy, res.exactitude, res.f1Macro, ...
        res.frugalite, dureeMinutes, machine, versionMatlab, ...
        VariableNames=["date" "essai" "graineDecoupage" "graineEntrainement" ...
        "taille" "optimiseur" "tauxApprentissage" "calendrierTaux" "augmentation" ...
        "tailleLot" "epochs" "parametres" "exactitudeTrain" "exactitudeVal" ...
        "f1Val" "frugalite" "dureeMin" "machine" "versionMatlab"]);
    journal = fullfile(outputFolder, "essais_squeezenet.csv");

    % Si les colonnes ont changé depuis la création du journal, l'ancien
    % journal est archivé sous un nom daté et un nouveau est commencé
    if isfile(journal)
        lignesJournal = readlines(journal);
        colonnesExistantes = split(lignesJournal(1), ",")';
        if ~isequal(colonnesExistantes, string(ligne.Properties.VariableNames))
            archive = fullfile(outputFolder, "essais_squeezenet_ancien_" + ...
                string(datetime("now", Format="yyyyMMdd_HHmmss")) + ".csv");
            movefile(journal, archive);
            warning("Les colonnes du journal ont changé : l'ancien journal est archivé dans %s.", archive);
        end
    end
    writetable(ligne, journal, WriteMode="append");
    fprintf("Essai ajouté au journal %s\n", journal);
end

%% SqueezeNet sur Food-11 : matrice de confusion + frugalité
close all
clear
clc

%% ---------------- Paramètres ----------------
% "scratch"    : SqueezeNet, poids aléatoires (imagePretrainedNetwork avec Weights="none")
% "pretrained" : SqueezeNet pré-entraîné ImageNet
mode = "pretrained";

trainFolder   = "food-11/train";
testFolder    = "food-11/test";
inputSize     = [227 227 3];   % taille d'entrée native de SqueezeNet
miniBatchSize = 64;
maxEpochs     = 20;

% La couche d'entrée de SqueezeNet attend des pixels dans [0,255]
scale = 255;

%% ---------------- Données ----------------
readFcn = @(f) readFoodImage(f, inputSize(1:2), scale);

imgDataTrain = imageDatastore(trainFolder, ...
    IncludeSubfolders=true, ...
    LabelSource="foldernames", ...
    ReadFcn=readFcn);
imgDataTest = imageDatastore(testFolder, ReadFcn=readFcn);

[imgDataTrain, imgDataValidation] = splitEachLabel(imgDataTrain, 0.8, "randomized");

augTrain      = augmentedImageDatastore(inputSize, imgDataTrain);
augValidation = augmentedImageDatastore(inputSize, imgDataValidation);
augTest       = augmentedImageDatastore(inputSize, imgDataTest);

classNames = categories(imgDataTrain.Labels);
numClasses = numel(classNames);

%% ---------------- Réseau ----------------
switch mode
    case "scratch"
        net = imagePretrainedNetwork("squeezenet", Weights="none", NumClasses=numClasses);
        modelName = "SqueezeNet (scratch)";
        learnRate = 5e-4;
    case "pretrained"
        % SqueezeNet pré-entraîné (DAGNetwork) -> on remplace la tête de classification
        lgraph = layerGraph(squeezenet);
        lgraph = removeLayers(lgraph, "ClassificationLayer_predictions");
        lgraph = replaceLayer(lgraph, "conv10", ...
            convolution2dLayer(1, numClasses, Name="conv10", ...
                WeightLearnRateFactor=10, BiasLearnRateFactor=10));
        net = dlnetwork(lgraph);
        modelName = "SqueezeNet pretrained";
        learnRate = 3e-4;
end

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
options = trainingOptions("adam", ...
    InitialLearnRate=learnRate, ...
    MaxEpochs=maxEpochs, ...
    MiniBatchSize=miniBatchSize, ...
    Shuffle="every-epoch", ...
    ValidationData=augValidation, ...
    ValidationFrequency=50, ...
    Verbose=false, ...
    Plots="training-progress", ...
    Metrics="accuracy");

net = trainnet(augTrain, net, "crossentropy", options);

%}
%% ---------------- Évaluation (evaluerReseau) ----------------
YVal = imgDataValidation.Labels;
res  = evaluerReseau(net, augValidation, YVal, classNames);

fprintf("%s\n", modelName);
fprintf("  Exactitude (accuracy) : %.2f %%\n", 100 * res.exactitude);
fprintf("  F1 macro              : %.4f\n", res.f1Macro);
fprintf("  Paramètres            : %d (%.3f M)\n", res.nbParametres, res.nbParametres / 1e6);
fprintf("  Frugalité             : %.4f\n", res.frugalite);

%% ---------------- Matrice de confusion ----------------
figure
confusionchart(res.matriceConfusion, classNames, ...
    Title=sprintf("%s - matrice de confusion (validation)", modelName), ...
    RowSummary="row-normalized", ...
    ColumnSummary="column-normalized");

%% ---------------- Prédictions test (JSON) ----------------
testScores     = minibatchpredict(net, augTest);
testLabelsPred = scores2label(testScores, classNames);

[~, testIds] = fileparts(imgDataTest.Files);
predictions = containers.Map(testIds, cellstr(testLabelsPred));

outputFolder = "built";
if ~isfolder(outputFolder)
    mkdir(outputFolder);
end
jsonFile = fullfile(outputFolder, "predictions_squeezenet.json");
writelines(jsonencode(predictions, PrettyPrint=true), jsonFile);
fprintf("Prédictions du test écrites dans %s (%d images)\n", jsonFile, predictions.Count);

%% ---------------- Fonction locale ----------------
function I = readFoodImage(filename, sz, scale)
    I = imread(filename);
    if size(I,3) == 1
        I = repmat(I,1,1,3);
    elseif size(I,3) == 4
        I = I(:,:,1:3);
    end
    I = imresize(I, sz);
    I = im2single(I) * scale;
end

%% ResNet-34 sur Food-11 : matrice de confusion + frugalité
close all
clear
clc

%% ---------------- Paramètres ----------------
% "scratch"      : vrai ResNet-34, poids aléatoires (resnetNetwork, R2024a+)
% "pretrained18" : ResNet-18 pré-entraîné ImageNet (repli pré-entraîné, car
%                  MATLAB ne fournit pas de ResNet-34 pré-entraîné)
mode = "scratch";

trainFolder   = "food-11/train";
testFolder    = "food-11/test";
inputSize     = [128 128 3];
miniBatchSize = 64;
maxEpochs     = 20;
w             = 0.3;     % poids de la pénalité de frugalité

% Les poids ImageNet attendent des pixels dans [0,255] ; le réseau "scratch" utilise [0,1]
if mode == "pretrained18"
    scale = 255;
else
    scale = 1;
end

%% ---------------- Données (même base que le CNN d'origine) ----------------
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
        % ResNet-34 : blocs basiques (2 conv 3x3), [3 4 6 3] blocs par étage
        net = resnetNetwork(inputSize, numClasses, ...
            BottleneckType="none", ...
            StackDepth=[3 4 6 3], ...
            NumFilters=[64 128 256 512]);
        modelName = "ResNet-34";
        learnRate = 1e-3;
    case "pretrained18"
        net = imagePretrainedNetwork("resnet18", NumClasses=numClasses);
        modelName = "ResNet-18 pretrained";
        learnRate = 3e-4;
end

% Sécurité : crossentropy attend des probabilités -> softmax en sortie
if ~isa(net.Layers(end), "nnet.cnn.layer.SoftmaxLayer")
    lastName = net.Layers(end).Name;
    net = addLayers(net, softmaxLayer(Name="softmax_out"));
    net = connectLayers(net, lastName, "softmax_out");
end

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

%% ---------------- Évaluation sur la validation ----------------
% Le jeu de test n'a pas de labels : score et matrice de confusion
% sont calculés sur les 20 % de validation.
valScores = minibatchpredict(net, augValidation);
YPred = scores2label(valScores, classNames);
YVal  = imgDataValidation.Labels;

P = mean(YPred == YVal);                 % score de prédiction (accuracy, entre 0 et 1)
fprintf("Validation-set accuracy (P): %.2f%%\n", 100 * P);

%% ---------------- Matrice de confusion ----------------
figure
confusionchart(YVal, YPred, ...
    Title=sprintf("%s - matrice de confusion (validation)", modelName), ...
    RowSummary="row-normalized", ...
    ColumnSummary="column-normalized");

%% ---------------- Frugalité : Frug = P - w / (1 + 1/R) ----------------
R = sum(cellfun(@numel, net.Learnables.Value));   % nombre de paramètres
Frug = P - w / (1 + 1/R);
fprintf("%s : R = %d paramètres, P = %.4f, Frug = %.4f\n", modelName, R, P, Frug);

%% ---------------- Prédictions test (JSON) ----------------
testScores     = minibatchpredict(net, augTest);
testLabelsPred = scores2label(testScores, classNames);

[~, testIds] = fileparts(imgDataTest.Files);
predictions = containers.Map(testIds, cellstr(testLabelsPred));

outputFolder = "built";
if ~isfolder(outputFolder)
    mkdir(outputFolder);
end
jsonFile = fullfile(outputFolder, "predictions_resnet34.json");
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

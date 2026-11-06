%% LeNet-5 sur Food-11 : matrice de confusion + frugalité
close all
clear
clc

%% ---------------- Paramètres ----------------
trainFolder   = "food-11/train";
testFolder    = "food-11/test";
inputSize     = [32 32 3];   % LeNet-5 d'origine : 32x32x1 ; ici on garde les 3 canaux RGB
miniBatchSize = 128;
maxEpochs     = 40;
learnRate     = 0.01;
w             = 0.3;         % poids de la pénalité de frugalité

%% ---------------- Données (même base que le CNN d'origine) ----------------
readFcn = @(f) readFoodImage(f, inputSize(1:2));

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

%% ---------------- Réseau LeNet-5 ----------------
% C1 -> S2 -> C3 -> S4 -> C5 -> F6 -> sortie (tanh + average pooling comme en 1998)
layers = [
    imageInputLayer(inputSize, Name="input")
    convolution2dLayer(5, 6,  Name="C1")             % 32x32 -> 28x28x6
    tanhLayer(Name="tanh1")
    averagePooling2dLayer(2, Stride=2, Name="S2")    % -> 14x14x6
    convolution2dLayer(5, 16, Name="C3")             % -> 10x10x16
    tanhLayer(Name="tanh3")
    averagePooling2dLayer(2, Stride=2, Name="S4")    % -> 5x5x16
    fullyConnectedLayer(120, Name="C5")
    tanhLayer(Name="tanh5")
    fullyConnectedLayer(84, Name="F6")
    tanhLayer(Name="tanh6")
    fullyConnectedLayer(numClasses, Name="output")
    softmaxLayer(Name="softmax")];

%% ---------------- Entraînement ----------------
options = trainingOptions("sgdm", ...
    InitialLearnRate=learnRate, ...
    MaxEpochs=maxEpochs, ...
    MiniBatchSize=miniBatchSize, ...
    Shuffle="every-epoch", ...
    ValidationData=augValidation, ...
    ValidationFrequency=50, ...
    Verbose=false, ...
    Plots="training-progress", ...
    Metrics="accuracy");

net = trainnet(augTrain, layers, "crossentropy", options);

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
    Title="LeNet-5 - matrice de confusion (validation)", ...
    RowSummary="row-normalized", ...
    ColumnSummary="column-normalized");

%% ---------------- Frugalité : Frug = P - w / (1 + 1/R) ----------------
R = sum(cellfun(@numel, net.Learnables.Value));   % nombre de paramètres
Frug = P - w / (1 + 1/R);
fprintf("LeNet-5 : R = %d paramètres, P = %.4f, Frug = %.4f\n", R, P, Frug);

%% ---------------- Prédictions test (JSON) ----------------
testScores     = minibatchpredict(net, augTest);
testLabelsPred = scores2label(testScores, classNames);

[~, testIds] = fileparts(imgDataTest.Files);
predictions = containers.Map(testIds, cellstr(testLabelsPred));

outputFolder = "built";
if ~isfolder(outputFolder)
    mkdir(outputFolder);
end
jsonFile = fullfile(outputFolder, "predictions_lenet5.json");
writelines(jsonencode(predictions, PrettyPrint=true), jsonFile);
fprintf("Prédictions du test écrites dans %s (%d images)\n", jsonFile, predictions.Count);

%% ---------------- Fonction locale ----------------
function I = readFoodImage(filename, sz)
    I = imread(filename);
    if size(I,3) == 1
        I = repmat(I,1,1,3);
    elseif size(I,3) == 4
        I = I(:,:,1:3);
    end
    I = imresize(I, sz);
    I = im2single(I);
end

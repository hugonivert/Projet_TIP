%Fonction pour redimensionner et normaliser les images
function I = readFoodImage(filename)

    I = imread(filename);

    % Conversion en RGB si nécessaire
    if size(I,3) == 1
        I = repmat(I,1,1,3);
    elseif size(I,3) == 4
        I = I(:,:,1:3);
    end

    % Redimensionnement
    I = imresize(I,[128 128]);

    % Normalisation entre 0 et 1
    I = im2single(I);

end
% Chargement des données 
close all
clear
clc
trainFolder = "food-11/train";
testFolder = "food-11/test";
%Chargement des images d'entraînement et de test
imgDataTrain = imageDatastore(trainFolder, ...
    IncludeSubfolders=true, ...
    LabelSource="foldernames", ...
    ReadFcn=@readFoodImage);

imgDataTest = imageDatastore(testFolder, ...
    ReadFcn=@readFoodImage);

% Séparation de train en 80% train et 20% validation
[imgDataTrain, imgDataValidation] = splitEachLabel( ...
    imgDataTrain, 0.8, "randomized");

inputSize = [128 128 3];

augTrain = augmentedImageDatastore(inputSize, imgDataTrain);
augValidation = augmentedImageDatastore(inputSize, imgDataValidation);
augTest = augmentedImageDatastore(inputSize, imgDataTest);

% Display a random sample of training images.
perm = randperm(numel(imgDataTrain.Files), 25);
subset = readall(subset(imgDataTrain,perm));
montage(subset)

% Define the network. The final softmax layer produces class probabilities.
% The classification loss is specified separately in trainnet.
layers = [
    imageInputLayer([128 128 3], Name="input")
    convolution2dLayer(3, 8, Name="conv")
    batchNormalizationLayer(Name="batchnorm")
    reluLayer(Name="relu")
    maxPooling2dLayer(2, Stride=2, Name="maxpooling")
    fullyConnectedLayer(11, Name="fullyconnected")
    softmaxLayer(Name="softmax")];

% Inspect the network architecture.
analyzeNetwork(layers)

% The validation split (20% of train) is supplied through ValidationData so
% its held-out accuracy and loss can be plotted during training. The official
% test set has no labels: it is only used for predictions (JSON export).
% Train the network using the R2024a dlnetwork workflow.
miniBatchSize = 500;
% Keep MATLAB's progress window for device/status information, and
% add a companion plot with smoothed curves and epoch bands.
maxEpochs = 16;

%Pour l'instant on enlève la partie smooth window ...

options = trainingOptions("sgdm", ...
    InitialLearnRate=0.001, ...
    MaxEpochs=maxEpochs, ...
    Shuffle="every-epoch", ...
    MiniBatchSize=miniBatchSize, ...
    ValidationData=augValidation, ...
    ValidationFrequency=50, ...
    Verbose=false, ...
    Plots="training-progress", ...
    Metrics="accuracy")
    %ExecutionEnvironment="gpu");

net = trainnet(augTrain, layers, "crossentropy", options);

% Evaluate the final network on the training and validation sets.
classNames = categories(imgDataTrain.Labels);

trainScores = minibatchpredict(net, augTrain);
trainLabelsPred = scores2label(trainScores, classNames);
trainAccuracy = mean(trainLabelsPred == imgDataTrain.Labels);
fprintf("Train-set accuracy: %.2f%%\n", 100 * trainAccuracy);

validationScores = minibatchpredict(net, augValidation);
validationLabelsPred = scores2label(validationScores, classNames);
validationAccuracy = mean(validationLabelsPred == imgDataValidation.Labels);
fprintf("Validation-set accuracy: %.2f%%\n", 100 * validationAccuracy);

% Prédictions sur le test : les vraies classes ne sont pas fournies
% (seul le serveur d'évaluation les connaît), donc pas de précision ici.
testScores = minibatchpredict(net, augTest);
testLabelsPred = scores2label(testScores, classNames);

% Export JSON pour le serveur d'évaluation : { "0": "bread", "1": "rice", ... }
% Clé = nom du fichier sans extension, valeur = classe prédite (= nom du
% dossier de train, déjà en minuscules). On associe via le nom de fichier et
% non la position, car les fichiers sont triés par ordre alphabétique
% (0, 1, 10, 100, ...).
[~, testIds] = fileparts(imgDataTest.Files);
predictions = containers.Map(testIds, cellstr(testLabelsPred));

% Le JSON est écrit dans le dossier "built" (au même niveau que food-11),
% créé automatiquement s'il n'existe pas encore
outputFolder = "built";
if ~isfolder(outputFolder)
    mkdir(outputFolder);
end
jsonFile = fullfile(outputFolder, "predictions.json");
writelines(jsonencode(predictions, PrettyPrint=true), jsonFile);
fprintf("Prédictions du test écrites dans %s (%d images)\n", jsonFile, predictions.Count);

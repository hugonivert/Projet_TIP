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
trainFolder = "train";
testFolder = "test";
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

% The official test set is supplied through ValidationData so its
% held-out accuracy and loss can be plotted during training. The custom plot
% labels these curves "Test set".
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

% Evaluate the final network on the complete training and sets.
trainScores = minibatchpredict(net, augTrain);
trainLabelsPred = scores2label(trainScores, categories(imgDataTrain.Labels));
trainAccuracy = mean(trainLabelsPred == imgDataTrain.Labels);
fprintf("Train-set accuracy: %.2f%%\n", 100 * trainAccuracy);

testScores = minibatchpredict(net, imgDataTest);
testLabelsPred = scores2label(testScores, categories(labelsTrain));
testAccuracy = mean(testLabelsPred == labelsTest);
fprintf("Test-set accuracy: %.2f%%\n", 100 * testAccuracy);

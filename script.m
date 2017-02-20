

% load MNIST data
load('data4students.mat')


%if no validation exists then
% val_x = [];
% val_y = [];

    inputSize = size(datasetInputs{1},2);
    outputSize = size(datasetTargets{1},2); % in case of classification it should be equal to the number of classes

     hiddenActivationFunctions = {'ReLu','softmax'};
%     hiddenLayers = [100 100 100 outputSize]; 
        hiddenLayers = [900 outputSize]; 

%    for i= 1:size(datasetInputs{1},1) 
%     
%        im = reshape(datasetInputs{1}(i,:),30,30);
%        rotated = imrotate(im,90);
%        datasetInputs{1}(end + i,:) = reshape(rotated,1,900); 
%        datasetTargets{1}(end + i,:) = datasetTargets{1}(i,:); 
% 
%        
%    end 
   
% parameters used for visualisation of first layer weights
visParams.noExamplesPerSubplot = 50; % number of images to show per row
visParams.noSubplots = floor(hiddenLayers(1) / visParams.noExamplesPerSubplot);
visParams.col = 30;% number of image columns
visParams.row = 30;% number of image rows 

inputActivationFunction = 'linear'; %sigm for binary inputs, linear for continuous input

% normalise data
% we assume that data are images so each image is z-normalised. If other
% types of data are used then each feature should be z-normalised on the
% training set and then mean and standard deviation should be applied to
% validation and test sets.
datasetInputs{1} = normaliseData(inputActivationFunction, datasetInputs{1}, []);
datasetInputs{2} = normaliseData(inputActivationFunction, datasetInputs{2}, []);
datasetInputs{3} = normaliseData(inputActivationFunction, datasetInputs{3}, []);

%initialise NN params
nn = paramsNNinit(hiddenLayers, hiddenActivationFunctions);

% Set some NN params
%-----
nn.epochs = 100;

% set initial learning rate
nn.trParams.lrParams.initialLR = 0.05; 
% set the threshold after which the learning rate will decrease (if type
% = 1 or 2)
nn.trParams.lrParams.lrEpochThres = 10;
% set the learning rate update policy (check manual)
% 1 = initialLR*lrEpochThres / max(lrEpochThres, T), 2 = scaling, 3 = lr / (1 + currentEpoch/lrEpochThres)
nn.trParams.lrParams.schedulingType = 1;

nn.trParams.momParams.schedulingType = 1;
%set the epoch where the learning will begin to increase
nn.trParams.momParams.momentumEpochLowerThres = 10;
%set the epoch where the learning will reach its final value (usually 0.9)
nn.trParams.momParams.momentumEpochUpperThres = 30;
nn.trParams.momParams.finalMomentum = 0.9;


% set weight constraints
nn.weightConstraints.weightPenaltyL1 = 0;
nn.weightConstraints.weightPenaltyL2 = 0;
%0.02
nn.weightConstraints.maxNormConstraint = 3;

% show diagnostics to monnitor training  
nn.diagnostics = 1;
% show diagnostics every "showDiagnostics" epochs
nn.showDiagnostics = 5;

% show training and validation loss plot
nn.showPlot = 1;

% use bernoulli dropout
nn.dropoutParams.dropoutType = 1;

% if 1 then early stopping is used
nn.earlyStopping = 0;
nn.max_fail = 10;

nn.type = 2;

% set the type of weight initialisation (check manual for details)
nn.weightInitParams.type = 9;

% set training method
% 1: SGD, 2: SGD with momentum, 3: SGD with nesterov momentum, 4: Adagrad, 5: Adadelta,
% 6: RMSprop, 7: Adam
nn.trainingMethod = 2;
%-----------

% initialise weights
[W, biases] = initWeights(inputSize, nn.weightInitParams, hiddenLayers, hiddenActivationFunctions);

nn.W = W;
nn.biases = biases;

% if dropout is used then use max-norm constraint and a
%high learning rate + momentum with scheduling
% see the function below for suggested values
% nn = useSomeDefaultNNparams(nn);

  [nn, Lbatch, L_train, L_val, clsfError_train, clsfError_val]  = trainNN(nn, datasetInputs{1}, datasetTargets{1}, datasetInputs{2}, datasetTargets{2});


nn = prepareNet4Testing(nn);

% visualise weights of first layer
figure()
visualiseHiddenLayerWeights(nn.W{1},visParams.col,visParams.row,visParams.noSubplots);

[stats, output, e, L] = evaluateNNperformance( nn, datasetInputs{3}, datasetTargets{3});
 





%% TFDNN


clear all;
close all;
clc;


% LOAD DATASET
load Xtrain
load Ytrain
load Xtest
load Ytest
% load Xval
% load Yval


% DATA STANDARDIZATION
temp_mn = ones(length(Xtrain),1);
temp_sd = ones(length(Xtrain),1);
numSeqTr = length(Xtrain);
numSeqTs = length(Xtest);
for q = 1:numSeqTr
    temp_mn(q,1) = mean(Xtrain{q,1});
    temp_sd(q,1) = std(Xtrain{q,1});
end
mn = mean(temp_mn);
sd = std(temp_sd);
for q = 1:numSeqTr
    Xtrain{q,1}(1,:) = (Xtrain{q,1}(1,:) - mn)/sd;
end
for r = 1:numSeqTs
    % Xval{r,1}(1,:) = (Xval{r,1}(1,:) - mn)/sd;
    Xtest{r,1}(1,:) = (Xtest{r,1}(1,:) - mn)/sd;
end


% SORT DATA BY SEQUENCE LENGTH
numObservations = numel(Xtrain);
for i = 1:numObservations
    sequence = Xtrain{i};
    sequenceLengths(i) = size(sequence,2);
end
[sequenceLengths, idx] = sort(sequenceLengths);
Xtrain = Xtrain(idx);
Ytrain = Ytrain(idx,1:3);
% figure
% bar(sequenceLengths)
% xlabel("Sequence")
% ylabel("Length")
% title("Sorted Data")


% DEFUZZIFICATION
Ytest_df = zeros(numSeqTs,1);
Ytest_max = max(Ytest,[],2);
for s = 1:length(Ytest)
    for t = 1:3
        if Ytest(s,t) == Ytest_max(s,1)
            Ytest_df(s,1) = t;
        end
    end
end


% PARAMETERS' SETTING
nApp = 20;  % number of appliances
nSeqApp = numSeqTs/nApp;  % number of sequences per appliance
numInputs = 1;  % number of input sequences
filterSize = 1;  % filter size of 1-D randomized convolutional layer
numFilters = 5;  % number of filter of 1-D randomized convolutional layer
numHiddenUnits = 30;  % number of hidden units of LSTM layer
mBatch = 4;  % mini-batch size
learnRate = 0.01;  % learning rate
numClasses = 3;  % number of classes
nRuns = 10;  % number of runs
best_RMSE_mean = Inf;
best_RMSE_std = Inf;
best_numHiddenUnits = 0;
best_numFilters = 0;
best_learnRate = 0;


% GRID SEARCH
for idx1 = 1:numel(learnRate)
    for idx2 = 1:numel(numHiddenUnits)
        for idx3 = 1:numel(numFilters)
            for idx4 = 1:numel(mBatch)


                % LAYERS
                lgraph = layerGraph(sequenceInputLayer(numInputs, 'Name', 'Input'));
                lgraph = addLayers(lgraph, convolution1dLayer(filterSize, numFilters(idx3), 'Name', 'conv_1'));
                lgraph = connectLayers(lgraph, 'Input', 'conv_1');
                lgraph = addLayers(lgraph, lstmLayer(numHiddenUnits(idx2), 'OutputMode', 'last', 'Name', 'lstm_1'));
                lgraph = connectLayers(lgraph, 'conv_1', 'lstm_1');
    %             lgraph = addLayers(lgraph, lstmLayer(numHiddenUnits(idx2), 'OutputMode', 'last', 'Name', 'lstm_2'));
    %             lgraph = connectLayers(lgraph, 'lstm_1', 'lstm_2');
                lgraph = addLayers(lgraph, fullyConnectedLayer(numClasses, 'Name', 'fc'));
                lgraph = connectLayers(lgraph, 'lstm_1', 'fc'); 
                start_D=0;
                for kk=1:numClasses
                    W = zeros(1,numClasses);
                    W(1,start_D+1) = 1;
                    B = 0;
                    start_D = start_D + 1;
                    lgraph = addLayers(lgraph, fullyConnectedLayer(1, 'Name', sprintf('Mask_%d',kk), 'Weights', W, 'Bias', B, 'WeightLearnRateFactor', 0, 'BiasLearnRateFactor', 0));
                    lgraph = connectLayers(lgraph, 'fc', sprintf('Mask_%d',kk));
                    lgraph = addLayers(lgraph, sigmoidLayer('Name', sprintf('sgm_%d',kk)));
                    lgraph = connectLayers(lgraph, sprintf('Mask_%d',kk), sprintf('sgm_%d',kk));
                end
                lgraph = addLayers(lgraph, concatenationLayer(1, numClasses, 'Name', 'Concat'));
                for kk=1:numClasses
                    lgraph = connectLayers(lgraph, sprintf('sgm_%d',kk), sprintf('Concat/in%d',kk));
                end
                lgraph = addLayers(lgraph, regressionLayer('Name','rmse'));
                lgraph = connectLayers(lgraph, 'Concat', 'rmse');
                % plot(lgraph)
            
                
                % TRAINING OPTIONS
                options = trainingOptions('adam',...
                    'GradientDecayFactor',0.9,...
                    'SquaredGradientDecayFactor',0.999,...
                    'Epsilon',1.0e-08,...
                    'InitialLearnRate',learnRate(idx1),...
                    'LearnRateSchedule','none',...
                    'LearnRateDropPeriod',50,...
                    'LearnRateDropFactor',0.4,...
                    'L2Regularization',1.0e-04,...
                    'GradientThresholdMethod','l2norm',...
                    'GradientThreshold',Inf,...
                    'MaxEpochs',100,...
                    'MiniBatchSize',mBatch(idx4),...
                    'Verbose',0,...
                    'Shuffle','never',...
                    'ExecutionEnvironment','auto',...
                    'Plots','none');
                

                % RUNS WITH DIFFERENT SEEDS
                u = 1;
                v = 3;
                for p = 1:nRuns
                    rng(p,'threefry');
                    gpurng(p,'threefry');
                    fprintf('Run = %d;\n', p)
    
                    % Training
                    start = tic;
                    net = trainNetwork(Xtrain,Ytrain,lgraph,options);
                    stop(1,p) = toc(start);
    
                    % Inference
                    Ypred = predict(net,Xtest,'MiniBatchSize',mBatch(idx4));
                    Ypred_app(:,u:v) = Ypred;
                    u = u + 3;
                    v = v + 3;
                    
                    % Data Denormalization (unused in this case)
                    % YPred = (YPred(1,:)*sd) + mn;
        
                    % Root Mean Squared Error
                    RMSE_app(:,p) = sqrt(mean((Ypred - Ytest).^2, 2));
                    RMSE_mn(1,p) = mean(RMSE_app(:,p));
    
                    % Defuzzification
                    Ypred_max = max(Ypred,[],2);
                    for s = 1:length(Ytest)
                        for t = 1:3
                            if Ypred(s,t) == Ypred_max(s,1)
                                Ypred_df(s,1) = t;
                            end
                        end
                    end

                    % Accuracy
                    Acc_app(:,p) = sum(Ypred_df == Ytest_df, 2);
                    Acc_mn(1,p) = sum(Acc_app(:,p))/numel(Ytest_df);
                    
                end
                
    
                % PERFORMANCE EVALUATION
                RMSE_app_mn = mean(RMSE_app,2);
                RMSE_app_std = std(RMSE_app,[],2);
                RMSE_all_mn = mean(RMSE_mn);
                RMSE_all_std = std(RMSE_mn);
                Acc_app_mn = mean(Acc_app,2);
                Acc_app_std = std(Acc_app,[],2);
                Acc_all_test = mean(Acc_mn);
                Acc_all_std = std(Acc_mn);
                time = sum(stop);
                if RMSE_all_mn < best_RMSE_mean
                    best_RMSE_mean = RMSE_all_mn;
                    final_RMSE_std = RMSE_all_std;
                    final_Acc_mean = Acc_all_test;
                    final_Acc_std = Acc_all_std;
                    final_time = time;
                    best_learnRate = learnRate(idx1);
                    best_numHiddenUnits = numHiddenUnits(idx2);
                    best_numFilters = numFilters(idx3);
                    best_mBatch = mBatch(idx4);
                end
            end
        end
    end
end


% GET PREDICTED LABEL PER APPLIANCE
idx11 = [1 4 7 10 13 16 19 22 25 28];  % Depend on 'nRuns'; (i.e. if nRuns=3 -> idx11=[1 4 7]) 
idx22 = idx11 + 1;
idx33 = idx22 + 1;
Class_1(:,1:nRuns) = Ypred_app(:,idx11);
Class_2(:,1:nRuns) = Ypred_app(:,idx22);
Class_3(:,1:nRuns) = Ypred_app(:,idx33);
% Averaged by runs
Class_1 = mean(Class_1,2);
Class_2 = mean(Class_2,2);
Class_3 = mean(Class_3,2);
% Averaged by appliance
u = 1;
v = nSeqApp;
for zz = 1:nApp
    Avg_Class_1(zz,1) = mean(Class_1(u:v));
    Avg_Class_2(zz,1) = mean(Class_2(u:v));
    Avg_Class_3(zz,1) = mean(Class_3(u:v));
    u = u + nSeqApp;
    v = v + nSeqApp;
end
Avg_Class = [Avg_Class_1, Avg_Class_2, Avg_Class_3];


% PRINT SOME INFO
fprintf('best_RMSE_mean = %.4f; final_RMSE_std = %.3f; final_Acc_mean = %.4f; final_Acc_std = %.3f; final_time = %.3f; best_learnRate = %.4f; best_numHiddenUnits = %d; best_numFilters = %d; best_miniBatch = %d;\n', best_RMSE_mean, final_RMSE_std, final_Acc_mean, final_Acc_std, final_time, best_learnRate, best_numHiddenUnits, best_numFilters, best_mBatch);

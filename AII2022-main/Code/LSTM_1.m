%% LSTM-1


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


% DEFUZZIFICATION AND DATA PRE-PROCESSING
Ytest_df = zeros(numSeqTs,1);
Ytest_max = max(Ytest,[],2);
for s = 1:length(Ytest)
    for t = 1:3
        if Ytest(s,t) == Ytest_max(s,1)
            Ytest_df(s,1) = t;
        end
    end
end
Ytrain_df = zeros(numSeqTr,1);
Ytrain_max = max(Ytrain,[],2);
for s = 1:length(Ytrain)
    for t = 1:3
        if Ytrain(s,t) == Ytrain_max(s,1)
            Ytrain_df(s,1) = t;
        end
    end
end
Ytest_df = categorical(Ytest_df);
Ytrain_df = categorical(Ytrain_df);


% PARAMETERS' SETTING
numInputs = 1;  % number of input sequences
numHiddenUnits = 20;  % number of hidden units of LSTM layer
learnRate = 0.006;  % learning rate
mBatch = 5;  % mini-batch size
numClasses = 3;  % number of classes
nRuns = 10;  % number of runs
best_Acc_mean = Inf;
best_Acc_std = Inf;
best_numHiddenUnits = 0;
best_numFilters = 0;
best_learnRate = 0;


% GRID SEARCH
for idx1 = 1:numel(learnRate)
    for idx2 = 1:numel(numHiddenUnits)
            

        % LAYERS
        layers = [
            sequenceInputLayer(numInputs)
            lstmLayer(numHiddenUnits(idx2),'OutputMode','last','name','lstm1')
            fullyConnectedLayer(numClasses)
            softmaxLayer()
            classificationLayer()];
        

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
            'MiniBatchSize',mBatch,...
            'Verbose',0,...
            'Shuffle','never',...
            'ExecutionEnvironment','auto',...
            'Plots','none');
        

        % RUNS WITH DIFFERENT SEEDS
        for p = 1:nRuns
            rng(p,'threefry');
            gpurng(p,'threefry');
            fprintf('Run = %d;\n', p)

            % Training
            start = tic;
            net = trainNetwork(Xtrain,Ytrain_df,layers,options);
            stop(1,p) = toc(start);

            % Inference
            Ypred = classify(net,Xtest,'MiniBatchSize',mBatch);

            % Accuracy
            Acc_app(:,p) = sum(Ypred == Ytest_df, 2);
            Acc_mn(1,p) = sum(Acc_app(:,p))/numel(Ytest_df);
        end

        
        % PERFORMANCE EVALUATION
        Acc_all_mn = mean(Acc_mn);
        Acc_all_std = std(Acc_mn);
        Acc_app_mn = mean(Acc_app,2);
        Acc_app_std = std(Acc_app,[],2);
        time = sum(stop);
        if Acc_all_mn < best_Acc_mean
            final_Acc_mean = Acc_all_mn;
            final_Acc_std = Acc_all_std;
            final_time = time;
            best_learnRate = learnRate(idx1);
            best_numHiddenUnits = numHiddenUnits(idx2);
        end
    end
end


% PRINT SOME INFO
fprintf('final_Acc_mean = %.4f; final_Acc_std = %.3f; final_time = %.3f; best_learnRate = %.4f; best_numHiddenUnits = %d;\n', final_Acc_mean, final_Acc_std, final_time, best_learnRate, best_numHiddenUnits);

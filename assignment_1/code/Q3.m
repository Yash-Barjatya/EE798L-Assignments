clc;clear all;close all;
%% QUESTION
% Using K-fold cross-validation, and the value of λ that gives the best
% predictive performance on the Olympic men’s 100m data for
% (a) a first-order polynomial (i.e. the standard linear model).
% (b) a fourth-order polynomial
%% Generating data
% xn = [1896; 1900; 1904; 1906; 1908; 1912; 1920; 1924; 1928; 1932; 1936; 1948; 1952; 1956; 1960; 1964; 1968; 1972; 1976; 1980; 1984; 1988; 1992; 1996; 2000; 2004; 2008];
% tn = [12.00; 11.00; 11.00; 11.20; 10.80; 10.80; 10.80; 10.60; 10.80; 10.30; 10.30; 10.30; 10.40; 10.50; 10.20; 10.00; 9.95; 10.14; 10.06; 10.25; 9.99; 9.92; 9.96; 9.84; 9.87; 9.85; 9.69];
% xntn = xn.*tn;
% xnxn = xn.*xn;
% T = table(xn, tn, xntn, xnxn);

%% Storing data in olympic_data.mat file
% Save the variable xn,tn, into one *.mat file
% save olympic_data.mat T
% Clear them out of the workspace
% clear xn tn xntn xnxn 
% Load them again
load ('olympic_data.mat');

%% K-fold-cross-validation
%Extracting data from Table T in olympic_data.mat file
xn = T.xn;
tn = T.tn;
% Defining the range of values for lambda from [10^-6 ,10^6]
lambda_range =  10.^[-10:1:10];
% Create a k-fold cross-validation object with 10 folds
K=10;

%%  randomly shuffling the order of the rows in the matrix x and spliting the rows into K roughly equally sized groups.

%no of rows
N = size(xn,1);
% create a random permutaion of number from 1 to N and store it in store
store = randperm(N);
% split the rows into K roughly equally sized groups
group_sizes= repmat(floor(N/K),1,K);
% including the left out rows in the last group
group_sizes(end)=group_sizes(end)+N-sum(group_sizes);
% creating a vector that specifies the starting and ending indices of each group within the 'store' vector by calculating the cummulative sum of the group size vector.
group_sizes=[0 cumsum(group_sizes)];

% centering the vector xn around 0
xn= xn -xn(1);
% scaling by factor of 4
xn=xn./4;
Xn = [repmat(1,size(xn)) xn];
% for linear
Xn_linear=Xn;
% for fourth order
Xn_4 =[Xn xn.^2 xn.^3 xn.^4];
%% Loop over lambda values and polynomial for first order
for i = 1:length(lambda_range)
    lambda = lambda_range(i);
    for k = 1:K 
         % Extract the train and test data
         X_train = Xn_linear(store,:);
         t_train = tn(store);
         X_test = Xn_linear(store(group_sizes(k)+1:group_sizes(k+1)),:);
         t_test = tn(store(group_sizes(k)+1:group_sizes(k+1)));
         X_train(group_sizes(k)+1:group_sizes(k+1),:) = [];
         t_train(group_sizes(k)+1:group_sizes(k+1)) = [];
        
         % Fit the model
         w = (X_train'*X_train + lambda*eye(size(Xn,2)))\X_train'*t_train;
        
         % Compute loss on test data
        predictions = X_test*w;
        loss(i,k) = sum((predictions - t_test).^2);
    end
end

% Find the optimal lambda value that gives the smallest average loss
mean_loss = mean(loss, 2);
[min_loss, min_loss_idx] = min(mean_loss);
opt_lambda = lambda_range(min_loss_idx);

% Print the optimal lambda value
fprintf("\nFor first-order polynomial, the optimal lambda value is %g and the min loss is %g\n", opt_lambda,min_loss);

%% Loop over lambda values and polynomial for fourth order
for i = 1:length(lambda_range)
    lambda = lambda_range(i);
    for k = 1:K 
         % Extract the train and test data
         X_train = Xn_4(store,:);
         t_train = tn(store);
         X_test = Xn_4(store(group_sizes(k)+1:group_sizes(k+1)),:);
         t_test = tn(store(group_sizes(k)+1:group_sizes(k+1)));
         X_train(group_sizes(k)+1:group_sizes(k+1),:) = [];
         t_train(group_sizes(k)+1:group_sizes(k+1)) = [];
        
         % Fit the model
         w = (X_train'*X_train + lambda*eye(size(Xn_4,2)))\X_train'*t_train;
        
         % Compute loss on test data
        predictions = X_test*w;
        loss(i,k) = sum((predictions - t_test).^2);
    end
end
% Find the optimal lambda value that gives the smallest average loss
mean_loss = mean(loss, 2);
[min_loss, min_loss_idx] = min(mean_loss);
opt_lambda = lambda_range(min_loss_idx);
% Print the optimal lambda value
fprintf("\n For fourth -order polynomial, the optimal lambda value is %g and the min loss is %g\n", opt_lambda,min_loss);

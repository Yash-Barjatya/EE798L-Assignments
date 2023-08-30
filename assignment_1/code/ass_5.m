clear all;close all;
load ('olympic_data.mat');
% Extract men's 100m data

x = T.xn;
t = T.tn;

% Choose number of folds
K = 10;

 % Randomise the data order
 N = size(x,1);
 order = randperm(N);
 sizes = repmat(floor(N/K),1,K);
 sizes(end) = sizes(end) + N-sum(sizes);
sizes = [0 cumsum(sizes)];

 % Rescale x
 x = x - x(1);
 x = x./4;

 X = [repmat(1,size(x)) x];
 X = [X x.^2 x.^3 x.^4];
 % Scan a wide range of values of the regularisation perameter
 regvals = 10.^[-10:1:10];
 for r = 1:length(regvals)
 for k = 1:K
 % Extract the train and test data
 traindata = X(order,:);
 traint = t(order);
 testdata = X(order(sizes(k)+1:sizes(k+1)),:);
 testt = t(order(sizes(k)+1:sizes(k+1)));
 traindata(sizes(k)+1:sizes(k+1),:) = [];
 traint(sizes(k)+1:sizes(k+1)) = [];

 % Fit the model
 w = inv(traindata'*traindata + regvals(r)*eye(size(X,2)))*...
 traindata'*traint;

% Compute loss on test data
 predictions = testdata*w;
 loss(r,k) = sum((predictions - testt).^2);
 
 mean_loss = mean(loss, 2);
[min_loss, min_loss_idx] = min(mean_loss);
opt_lambda = regvals(min_loss_idx);

% Print the optimal lambda value
fprintf("\n the optimal lambda value is %g and the min loss is %g\n", opt_lambda,min_loss);
 end
end
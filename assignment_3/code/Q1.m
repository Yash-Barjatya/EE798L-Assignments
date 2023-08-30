%% QUESTION : Generation of synthetic data set:
clear all; close all;
%% Define the gaussian mixture components/parameters
mixture_means = [3 3; 1 -3];
mixture_covs(:,:,1) = [1 0;0 2];
mixture_covs(:,:,2) = [2 0;0 1];
priors = [0.8 0.2];

%% Generate and plot the data points from the mixture model
figure(1); 
hold off
plotpoints = [1:1:10,12:2:30,50:50:500];
X = [];
for n = 1:500
comp = randsample(1:length(priors),1,true,priors);
    X(n,:) = mvnrnd(mixture_means(comp,:)',mixture_covs(:,:,comp),1);
    if any(plotpoints==n)
        figure(1);
         hold off
         plot(X(end,1),X(end,2),'ko','markersize',20,'markerfacecolor',[0.6 0.6 0.6]);
         hold on        
         plot(X(:,1),X(:,2),'ko');
         
    end
end


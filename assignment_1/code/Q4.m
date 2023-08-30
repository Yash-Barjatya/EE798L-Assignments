clc;clear all;close all;
%% QUESTION
% Plot pdf of two-dimensional Gaussian random vector and Generate the contour plots also.
%% SOLUTION
%% parameters
mu = [2 1];
SigmaA = eye(2);% part a
SigmaB = [1 0.8 ; 0.8 1];% part b
rng('default'); % For reproducibility
%% generating random sample
X1 = mvnrnd(mu,SigmaA,100);
X2 = mvnrnd(mu,SigmaB,100);
%% Plotting the pdf over grid
% Create meshgrid
[X, Y] = meshgrid(linspace(-1,5,50),linspace(-1,5,50));
% Compute probability density values at each point in the meshgrid
Z1 = mvnpdf([X(:) Y(:)], mu, SigmaA);
Z1 = reshape(Z1, size(X));
Z2 = mvnpdf([X(:) Y(:)], mu, SigmaB);
Z2 = reshape(Z2, size(X));
%% Plotting the pdf
figure(1);
scatter3(X1(:,1), X1(:,2), mvnpdf(X1, mu, SigmaA), 'filled');
title('PDF of Gaussian distribution A');
xlabel('X1');
ylabel('X2');
zlabel('Probability Density');
figure(2);
scatter3(X2(:,1), X2(:,2), mvnpdf(X2, mu, SigmaB), 'filled');
title('PDF of Gaussian distribution B');
xlabel('X1');
ylabel('X2');
zlabel('Probability Density');
%% Plotting the contour
figure(3);
contour(X, Y, Z1, 10);
title('Contour plot of Gaussian distribution A');
xlabel('X1');
ylabel('X2');
figure(4);
contour(X, Y, Z2, 10);
title('Contour plot of Gaussian distribution B');
xlabel('X1');
ylabel('X2');

clc;clear all;close all;
%% QUESTION
%  Regularized least squares: Consider the following model
% t = 2x − 3 + n.
% Random noise n is distributed as N (0, 3).
% (a) Data set: Generate six values of x which are uniformly spread between 0 and 1.
% Calculate t corresponding to all the x values.
% (b) Fit a fifth-order polynomial to this data set using regularized least squares approach
% for λ = 0, 10−6,
% , 0.01 and 0.1.
% (c) Verify that the behavior matches with the one discussed in the class.
%% DATA SET
rng(1);
N=6;
x=rand(N,1);
%% Linear model
w0=-3;w1=2;
t = w0 +w1*x ;
% adding noise to this model
noisevar=3;
noise= sqrt(noisevar)*randn(size(x));
t =t + noise;
%% Plotting the model
figure(1);
hold off;
plot(x,t,'k.',MarkerSize=10);
xlabel('x (attribute)');
ylabel('t (target)');
title('Targets');
%% fifth order polynomial data
testx=[0:0.01:1]';
X=[];
testX=[];
for k = 0:5
    X=[X x.^k];
    testX=[testX testx.^k];
end
%% Regularising fifth order data
lambda_values=[0 1e-6 1e-2 1e-1];
for i =1:length(lambda_values)
    lambda = lambda_values(i);
    w_reg=(X'*X+N*lambda*eye(size(X,2)))\X'*t;
    testY=testX*w_reg;
    figure;
    hold off;
    plot(x,t,'k.',MarkerSize=20);
    hold on;
    plot(testx,testY,'b','LineWidth',2);
    xlim([-0.1 1.1])
    xlabel('$x$','interpreter','latex','fontSize',20);
    ylabel('$f(x)$','interpreter','latex','fontSize',20);
    ti=sprintf('$\\lambda= %g$',lambda);
    title(ti,'interpreter','latex','fontSize',20)
end
%% Conclusion
fprintf("For a N+1 data point , Nth order polynomial will completely fit it .In our case,N=6 therfore fifth order polynomial completely fits it.To reduce the overfitting and model complexity we regualrise it with a parameter lamda .AS lamda increases the model deviates from perfect fitting ");
%% QUESTION
% Predictive variance: Consider the following model
% t = 5x^3 − x^2 + x + n.
% Random noise n is distributed as N (0, 300).
% (a) Data set: Generate N = 100 values of x which are uniformly distributed between −5 and 5. Calculate t for all the x values.
% (b) Fit linear, cubic and sixth-order models to this data set using maximum-likelihood approach.
% (c) Plot the predictive error bars for all these models.
%% Solution
clc;clear all;close all;
%% DATA SET
rng(1);
N=100;
x=10*sort(rand(N,1)-0.5);
noisevar=300;
noise= sqrt(noisevar)*randn(size(x));
%% model
t = 5*x.^3 - x.^2 + x + noise;
%% Fiting the model
X=[];
for k= 0:6
    X= [X x.^k];
    if(k==1)
        w_linear=(X'*X)\X'*t;
        var_linear=(1/N)*(t'*t-t'*X*w_linear);
        noise_linear= sqrt(var_linear)*randn(size(x));
        t_linear=X*w_linear;
    end
    if(k==3)
        w_cubic=(X'*X)\X'*t;
        var_cubic=(1/N)*(t'*t-t'*X*w_cubic);
        noise_cubic= sqrt(var_cubic)*randn(size(x));
        t_cubic=X*w_cubic;
    end
    if(k==6)
        w_6=(X'*X)\X'*t;
        var_6=(1/N)*(t'*t-t'*X*w_6);
        noise_6= sqrt(var_6)*randn(size(x));
        t_6=X*w_6;
    end
end
%% plotting models with predictive error bars 
figure(1);
sgtitle('Maximum likelihood of linear,cubic,sixth order model');
subplot(2,2,1);
plot(x,t,'k.',MarkerSize=10);
xlabel('x (attribute)');
ylabel('t (target)');
title('Target');

subplot(2,2,2);
plot(x,t,'k.',MarkerSize=10);
hold on;
plot(x,t_linear,'b',LineWidth=2);
errorbar(x,t_linear,noise_linear);
legend('Data','Linear');
xlabel('x (attribute)');
ylabel('t (target)');
title('Linear model');
hold off;

subplot(2,2,3);
plot(x,t,'k.',MarkerSize=10);
hold on;
plot(x,t_cubic,'g',LineWidth=2);
errorbar(x,t_cubic,noise_cubic,'b.');
legend('Data','Cubic');
xlabel('x (attribute)');
ylabel('t (target)');
title('Cubic model');
hold off;

subplot(2,2,4);
plot(x,t,'k.',MarkerSize=10);
hold on;
plot(x,t_6,'y',LineWidth=2);
errorbar(x,t_6,noise_6,'g');
legend('Data','Sixth-order');
xlabel('x (attribute)');
ylabel('t (target)');
title('6th-order model');
hold off;
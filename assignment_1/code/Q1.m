clc;clear all;close all;
%% QUESTION
%Non-linear response from a linear model: For the following model:
% t = w0 + w1x + w2x^2+ n,
% the values of parameters are w0 = 1, w1 = −2 and w2 = 0.5. Random noise n is
% distributed as N (0, 1).
% (a) Data set: Generate N = 200 values of x which are uniformly distributed between
% −5 and 5. Calculate t for all the x values.
% (b) Fit a linear and quadratic model to this data set using least squares approach.
%% DATA SET
rng(1);
N=200;
x=10*sort(rand(N,1)-0.5);
%% Linear model
w0=1;w1=-2;w2=0.5;
t = w0 +w1*x +w2*(x.^2);
% adding noise to this model
noise= 0.5*randn(N,1);
t =t + noise;
%% Plotting the model
figure(1);
hold off;
plot(x,t,'k.',MarkerSize=10);
xlabel('x (attribute)');
ylabel('t (target)');
%% Fiting the model
X=[];
for k= 0:2
    X= [X x.^k];
    if(k==1)
        w_linear=(X'*X)\X'*t;
    end
    if(k==2)
        w_quad=(X'*X)\X'*t;
    end
end
fprintf("\nThe linear function is t = %g +%gx\n",w_linear(1),w_linear(2));
fprintf("\nThe quadratic function is t = %g +%gx+%gx^2\n",w_quad(1),w_quad(2),w_quad(3));
%% Plotting and comparing the linear and quad models

x_new=x;
X_new=[];
for k=0:2
    X_new=[X_new x_new.^k];
end
t_linear=X_new(:,1:2)*w_linear;
t_quad=X_new*w_quad;
figure(1);
hold off;
plot(x,t,'k.',MarkerSize=10);
xlabel('x (attribute)');
ylabel('t (target)');
hold on;
plot(x_new,t_linear,'b',LineWidth=2);
plot(x_new,t_quad,'g',LineWidth=2);
legend('Data','Linear','Quadratic');
%% Conclusion
% since the data is a second order polynomial therfore quadratic model fit better than linear
fprintf("Since the data is a second order polynomial therfore quadratic model fit better than linear");
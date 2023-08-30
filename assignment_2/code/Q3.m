clc;clear all; close all;
%% QUESTION 
% Apply SBL for regression from [R1] to get the maximum aposterior estimate of the weight vector w, which is given by (13).
%% Loading data
N=20;
M=40;
phi = randn(N,M);
D=7;
w= zeros(M,1);
% Selecting D indices at random from 1 to M
nonzero_idx = randperm(M, D);
% Generating the non-zero components of w with a standardized Gaussian distribution
w(nonzero_idx) = randn(D, 1);
noise_variances = [10^(-20/10),10^(-15/10), 10^(-10/10), 10^(-5/10),10^(0/10)];
%% run the SBL algorithm
for i = 1:length(noise_variances)
    % initialising the parameters
    noise_var = noise_variances(i);
    fprintf("\n\n --------FOR NOISE VARIANCE : %g dB --------\n",-10*log10(noise_var));
    t=phi*w+noise_var*randn(N,1);
    alpha = 100;
    lambda = 1e-8;
    a=0;
    b=1e-4;
    alpha_old = ones(M,1) * alpha;
    A =diag(alpha_old);
    covar=inv(phi'*phi/noise_var + A);
    w_old=covar * phi' *t/noise_var;
    tolerance =1e-3;
    while(true)
        % update the weight precision
        gamma = 1-alpha_old.*diag(covar);
        % update the hyperparameter alpha (see Equation 44,of reading paper)
        alpha_old=(gamma+2*a)./(w_old.^2+2*b);
        A =diag(alpha_old);
         %update the covariance matrix (see Equation 12,of reading paper)
        covar=inv(phi'*phi/noise_var + A);
         % update the weight mean (see Equation 13,of reading paper)
        w_new = covar * phi' *t/noise_var;
        if(norm(w_new- w_old)<=tolerance)
            break;
        end
        w_old = w_new;
    end
% extract the maximum a posteriori estimate of the weight vector w
w_map = [0;w_new(2:end)];
fprintf("\nthe maximum a posteriori estimate of the weight vector is :%g", w_map);
end

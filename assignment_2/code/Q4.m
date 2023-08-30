%% Q4 Plot the normalized mean squared error (NMSE)
clc;clear all; close all;
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
%% %% Generate data for different noise variances and calculate NMSE
noise_variances = [10^(-20/10),10^(-15/10), 10^(-10/10), 10^(-5/10),10^(0/10)];
NMSE =zeros(size(noise_variances));
alpha = 100;

for i = 1:length(noise_variances)
    % initialising the parameters
    noise_var = noise_variances(i);
    t=phi*w+noise_var*randn(N,1);
    alpha = 100;
    lambda = 1e-2;
    a=0;
    b=1e-8;
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
w_map = [0;w_new(2:end)]; % add a 0 to the beginning of w_new so that they are of same size after ignoring bias
% Calculate the NMSE
NMSE(i) = norm(w_map - w)^2 / norm(w)^2;
fprintf("Noise variance: %g( == %g dB), NMSE: %g\n", noise_var,-10*log10(noise_var),NMSE(i));
end
%% Plot the NMSE vs. noise variance(semi-log)
figure();
semilogx(noise_variances, NMSE, '-o', 'LineWidth', 1.5, 'MarkerSize', 5);
xlabel("Noise Variance");
ylabel("NMSE");
title("Normalized mean squared error(NMSE) vs. noise variance");
grid on;

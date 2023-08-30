clc;clear all; close all;
%% QUESTION
% Generate t for N = 20, M = 40, D0 = 7, noise variances -20, -15, -10, -5 and 0 dB.
N=20;
M=40;
phi = randn(N,M);
% ii)  Generate the M×1 sparse weight vector w such that it has D0 randomly selected nonzero entries (with standardized Gaussian distributed nonzero components).
D=7;
w= zeros(M,1);
% Selecting D indices at random from 1 to M
nonzero_idx = randperm(M, D);

% Generating the non-zero components of w with a standardized Gaussian distribution
w(nonzero_idx) = randn(D, 1);

% iii) Generate the noise entries epsilon_n ∼ N (0, σ2) for all n = 1, . . . , N. Generate theobservations t = Φw + epsilon.
t1 =phi*w+calc_noise(-20,N);
t2= phi*w+calc_noise(-15,N);
t3 =phi*w+calc_noise(-10,N);
t4= phi*w+calc_noise(-5,N);
t5 =phi*w+calc_noise(0,N);

function noise = calc_noise(variance,N)
    noise_var= 10^(variance/20);
    noise = noise_var*randn(N,1);
end

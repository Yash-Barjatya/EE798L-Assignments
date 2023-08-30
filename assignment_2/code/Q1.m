clc;clear all; close all;
%% QUESTION
% i) Generate a N × M design/dictionary matrix Φ, whose entries are each drawn from a standardized Gaussian distribution, i.e., N (0, 1).
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
noise_var=0.1;
noise = noise_var*randn(N,1);
t= phi*w+noise
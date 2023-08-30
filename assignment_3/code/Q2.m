%% QUESTION :Assuming K = 2, iteratively perform the following EM algorithm steps.
clear all; close all;
%% Define the gaussian mixture components/parameter
mixture_means = [3 3; 1 -3];
mixture_covs(:,:,1) = [1 0;0 2];
mixture_covs(:,:,2) = [2 0;0 1];
mixture_priors = [0.8 0.2];

%% Generate data points from the mixture model
X = [];
for n = 1:500
    comp = randsample(1:length(mixture_priors),1,true,mixture_priors);
    X(n,:) = mvnrnd(mixture_means(comp,:)',mixture_covs(:,:,comp),1);
end
%% Initilaise the mixture with random initial values
% Number of clusters.
K = 2;
means = randn(K,2);
for k = 1:K
    covs(:,:,k) = rand*eye(2);
end
% Assign equal prior probabilities to each cluster.
priors = ones(1, K) * (1 / K);
%% Run the EM algorithm
MaxIts = 1000;
N = size(X,1);
% Matrix to hold the probability that each data point belongs to each cluster.
q = zeros(N,K);
D = size(X,2);
cols = {'r','g','b'};
plotpoints = [1:1:10,12:2:30,50:50:500];
Bound(1) = -inf;
converged = 0;
it = 0;
tol = 1e-6;
% Loop until convergence
while 1
    it = it + 1;
    %% STEP 1: Expectation
    for k = 1:K
        const = -(D/2)*log(2*pi) - 0.5*log(det(covs(:,:,k)));
        % Subtract the cluster mean from all data points.
        Xm = X - repmat(means(k,:),N,1);
        logLikelihood(:,k) = const - 0.5 * diag(Xm*inv(covs(:,:,k))*Xm');
    end
    
    % Calculate the Bound on the likelihood
    if it>1
        Bound(it) = sum(sum(q.*log(repmat(priors,N,1)))) + sum(sum(q.*logLikelihood)) - sum(sum(q.*log(q)));
        if abs(Bound(it)-Bound(it-1))<tol
            converged = 1;    
        end
    end
    % Check for convergence.
    if converged == 1 || it>MaxIts
        break
    end
    
    logLikelihood = logLikelihood + repmat(priors,N,1);
    
    q = exp(logLikelihood - repmat(max(logLikelihood,[],2),1,K));
    q = q./repmat(sum(q,2),1,K);
    %% STEP 2: Maximization
    % Update prior probabilities for each cluster.
    priors = mean(q,1);
    % Update mean for cluster 'k' by taking the weighted average of all data points.
    for k = 1:K
        means(k,:) = sum(X.*repmat(q(:,k),1,D),1)./sum(q(:,k));
    end
    % update covariances for cluster 'k' by taking the  weighted average of the covariance for each training example.
    for k = 1:K
        % Subtract the cluster mean from all data points.
        Xm = X - repmat(means(k,:),N,1);
        % Calculate the contribution of each training example to the covariance matrix.
        covs(:,:,k) = (Xm.*repmat(q(:,k),1,D))'*Xm;
        % Divide by the sum of weights.
        covs(:,:,k) = covs(:,:,k)./sum(q(:,k));
    end
  % End of Expectation Maximization        
    %% Plot the current clusters
    if any(it==plotpoints)
        figure(1);hold off
        % Note the following plots points using q as their RGB colour value
        for n = 1:N
        % Assign the color based on the cluster assignment
            if q(n,1) > q(n,2)
                color = [1 0 0]; % red for cluster 1
            else
                color = [0 1 0]; % green for cluster 2
            end
            plot(X(n,1),X(n,2),'ko','markerfacecolor',color);
            hold on
        end
        for k = 1:K
            mu = means(k,:);
            sigma = covs(:,:,k);
            [X1,X2] = meshgrid(-2:0.1:6,-6:0.1:6);
            F = mvnpdf([X1(:) X2(:)],mu,sigma);
            F = reshape(F,length(-6:0.1:6),length(-2:0.1:6));
            contour(X1,X2,F,3,'LineWidth',2,'Color',cols{k});
            hold on
        end
        ti = sprintf('After %g iterations',it);
        title(ti)
    end 
end
%% Plot the bound v/s iteration graph
figure(1);hold off
plot(2:length(Bound),Bound(2:end),'k');
xlabel('Iterations');
ylabel('Bound');
%% Final value of sigma and mean after the convergence

% Compare the covariance matrices and match them based on their proximity.
if norm(covs(:,:,1)-mixture_covs(:,:,1)) > norm(covs(:,:,1)-mixture_covs(:,:,2))
    covs(:,:,1) = covs(:,:,1) + covs(:,:,2);
    covs(:,:,2) = covs(:,:,1) - 2*covs(:,:,2);
    covs(:,:,1) = (covs(:,:,1) - covs(:,:,2))/2;
end
% Assign the means to mixture_means based on which mean is closest in Euclidean distance
for k = 1:K
    dists = sum((means(k,:) - mixture_means).^2, 2);
    [~, min_idx] = min(dists);
    mixture_means(k,:) = mixture_means(min_idx,:);
end
fprintf("\n-----Original covariance-----");mixture_covs
fprintf("\n-----Covariance after covergence-----");covs
fprintf("\n-----Original mean-----");mixture_means
fprintf("\n-----Mean after covergence-----");means
fprintf("\n-----Original prior-----");mixture_priors
fprintf("\n-----Prior after covergence-----");priors
% Initialize parameters
K = 2; % Number of Gaussians
N = 500; % Number of data points
means = [3 3; 1 -3]; % Initial means
covs(:,:,1) = [1 0;0 2]; % Initial covariance matrix for Gaussian 1
covs(:,:,2) = [2 0;0 1]; % Initial covariance matrix for Gaussian 2
priors = [0.8 0.2]; % Initial mixing coefficients

% Generate data from the mixture model
data = zeros(N,2);
for i=1:N
    if rand() < priors(1)
        data(i,:) = mvnrnd(means(1,:),covs(:,:,1));
    else
        data(i,:) = mvnrnd(means(2,:),covs(:,:,2));
    end
end

% Initialize variables for the EM algorithm
tolerance = 1e-6;
maxIterations = 1000;
logLikelihoods = zeros(1,maxIterations);

for iteration = 1:maxIterations
    % E-step: compute responsibilities
    responsibilities = zeros(N,K);
    for i=1:N
        for j=1:K
            responsibilities(i,j) = priors(j)*mvnpdf(data(i,:),means(j,:),covs(:,:,j));
        end
        responsibilities(i,:) = responsibilities(i,:)/sum(responsibilities(i,:));
    end
    
    % M-step: update parameters
    for j=1:K
        sumR = sum(responsibilities(:,j));
        means(j,:) = (responsibilities(:,j)'*data)/sumR;
        covs(:,:,j) = zeros(2,2);
        for i=1:N
            covs(:,:,j) = covs(:,:,j) + responsibilities(i,j)*(data(i,:)-means(j,:))'*(data(i,:)-means(j,:))/sumR;
        end
        priors(j) = sumR/N;
    end
    
    % Compute log-likelihood
    logLikelihood = 0;
    for i=1:N
        logLikelihood = logLikelihood + log(sum(priors.*mvnpdf(data(i,:),means,covs)));
    end
    logLikelihoods(iteration) = logLikelihood;
    
    % Check convergence
    if iteration > 1 && abs(logLikelihoods(iteration)-logLikelihoods(iteration-1)) < tolerance
        break;
    end
end

% Plot the data and the learned Gaussians
figure;
scatter(data(:,1),data(:,2),10,'filled');
hold on;
x = linspace(-10,10,100);
y = linspace(-10,10,100);
[X,Y] = meshgrid(x,y);
Z = zeros(length(x),length(y));
for j=1:K
    for i=1:length(x)
        Z(i,:) = Z(i,:) + priors(j)*mvnpdf([X(i,:)' Y(i,:)'],means(j,:),covs(:,:,j));
    end
end
contour(X,Y,Z,[0.001 0.01 0.05 0.1 0.2 0.5],'LineWidth',2);
title('Gaussian mixture model');
xlabel('x');
ylabel('y');
legend('Data','Gaussian 1','Gaussian 2');

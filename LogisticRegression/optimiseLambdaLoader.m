% Load dataset and pass it to lambda optimiser
% Graham Jones, 1.8.12
load('BioData.mat');
X=[ones(size(X,1),1) X];
[Xtrain,ytrain,Xcv,ycv,Xtest,ytest]=segmentDataset(X,y);
[lambda theta]=optimiseLambda(Xtrain,ytrain,Xcv,ycv,true);
pr('Optimal value for lambda is %.2f\n',lambda);


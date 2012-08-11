
% Read data ....................;
trainset = csvread('train.csv');
testset = csvread('test.csv');

X = trainset(2:end, 2:end);
y = trainset(2:end, 1);
y(y==0) = -1;
%X_test = testset(2:end, :);
[m ,n] = size(X);

features_max = max(X);
features_min = min(X);

% Perform feature scaling [0, 1]
% Note: - store scaling factors in order to apply to the test data.;

X_scaled = zeros(m, n);
for i = 1:n
	X_scaled(:,i) = (X(:,i) - features_min(i))/(features_max(i) - features_min(i));
end
		
% Feature selection to be added later.

[Xtrain, ytrain, Xcv, ycv, Xtest, ytest] = segmentDataset(X_scaled,y);
Xtrain = [Xtrain; Xcv];
ytrain = [ytrain; ycv];

features_sparse = sparse(Xtrain);
libsvmwrite('Xtrain_sp', ytrain, features_sparse);

[ytrain_vec, instance_matrix] = libsvmread('Xtrain_sp'); 

C = [0.01; 0.1; 1; 5; 10; 20; 30; 50; 100; 1000];
gamma = [2^-5; 2^-3; 2^-1; 1; 2; 4; 8; 16; 32];
lenC = length(C);
lenG = length(gamma);

cross_val = zeros(lenC*lenG, 3);
[p, q] = meshgrid(C, gamma);
cross_val(:,1:2) = [p(:) q(:)];
k = 1;

t0 = clock();

for i = 1:lenC
	for j = 1:lenG
	kernel = 2; cost = C(i); gam = gamma(j); shrink = 0; cv = 10; 
	options = ['-c ' , num2str(cost), ' -t ', num2str(kernel), ' -g ', num2str(gam), ' -h ', num2str(shrink), ' -v ', num2str(cv), ' -q'];
	cross_val(k,3) = svmtrain(ytrain_vec, instance_matrix, options);
	k = k + 1;
	end
end
time_elapsed = etime(clock(), t0);
printf("\n elapsed time (in secs): %d", time_elapsed);
fflush(stdout);

index = find(cross_val(:,3) == max(cross_val(:,3)));

options2 = ['-c ' , num2str(cross_val(index,1)), ' -t ', num2str(kernel), ' -g ', num2str(cross_val(index,2)), ' -h ', num2str(shrink), ' -q'];
model = svmtrain(ytrain_vec, instance_matrix, options2);
[predict_label, accuracy, dec_values] = svmpredict(ytrain_vec, instance_matrix, model);
err = sum(ytrain_vec + predict_label==0)

sparse_Xtest = sparse(Xtest);
libsvmwrite('Xtest_sp', ytest, sparse_Xtest);

[ytest_vec, instance_matrix_test] = libsvmread('Xtest_sp');
[predict_test, accuracy_test, dec_values_test] = svmpredict(ytest_vec, instance_matrix_test, model);
err_test = sum(ytest_vec + predict_test==0)

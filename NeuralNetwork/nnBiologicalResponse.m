%% Biological Response - Kaggle
%
%

printf('\nBiological Response Learning\n') ;

%% =========== Part 1: Training Data Loading

printf("\nLoading TRAINING data ... ") ;
load("BioData.mat");

% Split the training data into train, CV and test
[X_train, y_train, X_cv, y_cv, X_test, y_test] = segmentDataset(X, y) ;

[m, cols] = size(X_train) ;
printf("done. Dimension for X_train is : %d X %d\n", m, cols);
printf("Dimension for y_train is : %d X %d\n", size(y_train));

%% =========== Part 2: Neural Network ============
input_layer_size  = 1776;  % Represent molecular descriptors (d1 through d1776)
hidden_layer_size = 25;   % 25 hidden units
output_layer_size = 1;   % biological response

printf("Learning using Neural Network. ");

printf('\nInitializing Neural Network Parameters ...\n');

initial_Theta1 = nnRandInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = nnRandInitializeWeights(hidden_layer_size, output_layer_size);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

printf('\nTraining Neural Network... \n');


options = optimset('MaxIter', 50);

lambda = 1;

% Loop for increasing values of m. Needed to obtain pairs
% (J_train, m) and (J_cv, m) in order to be able to plot
% the corresponding error curves

% Determine the step to use to increase m. Assuming that we need 20
% datapoints. We can change that number if needed.
DATAPOINTS_NEEDED = 5;

step_for_m = ceil(m / DATAPOINTS_NEEDED) ;
m_count = step_for_m ;

m_values = zeros(DATAPOINTS_NEEDED, 1) ;
J_train_values = zeros(DATAPOINTS_NEEDED, 1) ;
J_cv_values = zeros(DATAPOINTS_NEEDED, 1) ;

for i = 1:DATAPOINTS_NEEDED
	X_used = X_train(1:m_count, :) ;
	y_used = y_train(1:m_count, :) ;

	% Optimize
	costFunction = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, output_layer_size, X_used, y_used, lambda);
									   
	[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

	printf("\nLearning completed. Lambda used: %.1f -- # of samples used(m): %d\n",lambda, m_count);
	
	% store the values to be plotted. NEED TO ADD J_cv
	J_train_values(i) = cost(50);
	J_cv_values(i) = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, output_layer_size, X_cv, y_cv, lambda);
	m_values(i) = m_count ;
	   
	% increase the number of samples used, making sure we never exceed m
	m_count = min(m, m_count + step_for_m) ;
endfor

plotErrors(m_values, J_train_values, J_cv_values) ;

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 output_layer_size, (hidden_layer_size + 1));
		
%% =========== Part 3: Testing and evaluation ============
% Compute accuracy on our testing set.

[m_test, cols] = size(X_test) ;
printf("Test data dimension is : %d X %d\n", m_test, cols);

p = nnPredict(Theta1, Theta2, X_test);

printf("Train Accuracy: %f\n", mean(double(p == y_test)) * 100);
%% Biological Response - Kaggle
%
%
page_output_immediately(1) ;

printf('\nBiological Response Learning\n') ;

%% =========== Part 1: Training Data Loading

printf("\nLoading TRAINING data ... ") ;
load("BioData.mat");

% Split the training data into train, CV and test
[X_train, y_train, X_cv, y_cv, X_test, y_test] = segmentDataset(X, y) ;

[m, cols] = size(X_train) ;
printf("done. Dimension for X_train is : %d X %d\n", m, cols);
printf("Dimension for y_train is : %d X %d\n", size(y_train));

% Determine whether the data is skewed
%[ isskewed, skewnesspc ] = skewness( y_train, 0.1 ) ;
isskewed = false ;

aux = '' ;
if (! isskewed )
   aux = ' NOT' ;
endif
printf("Data is%s skewed\n", aux) ;
printf("\nProgram paused (3 secs).\n");
pause (3) ;

%% =========== Part 2: Regularized Logistic Regression ============

printf("Learning using Logistic Regression. ") ;

% Initialize fitting parameters
initial_theta = zeros(size(X_train, 2), 1);

% Set regularization parameter lambda to 1
lambda = 1;


% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 100);

% Loop for increasing values of m. Needed to obtain pairs
% (J_train, m) and (J_cv, m) in order to be able to plot
% the corresponding error curves

% Determine the step to use to increase m. Assuming that we need 20
% datapoints. We can change that number if needed.
DATAPOINTS_NEEDED = 20 ;

step_for_m = ceil(m / DATAPOINTS_NEEDED) ;
m_count = step_for_m ;

m_values = zeros(DATAPOINTS_NEEDED, 1) ;
J_train_values = zeros(DATAPOINTS_NEEDED, 1) ;
J_cv_values = zeros(DATAPOINTS_NEEDED, 1) ;

for i = 1:DATAPOINTS_NEEDED
   X_used = X_train(1:m_count, :) ;
   y_used = y_train(1:m_count, :) ;

   printf("\nSize of X_used is: %d X %d", size(X_used)) ;
   printf("\nSize of y_used is: %d X %d", size(y_used)) ;
   % Optimize
   [theta, J, exit_flag] = ...
	fminunc(@(t)(lrCostFunction(t, X_used, y_used, lambda)), initial_theta, options);

   printf("\nLearning completed. Lambda used: %f # of samples used(m): %d\n",lambda, m_count) ;
   
   % store the values to be plotted. NEED TO ADD J_cv
   J_train_values(i) = J ;
   J_cv_values(i) = lrCostFunction(theta, X_cv, y_cv, lambda) ;
   m_values(i) = m_count ;
   
   % increase the number of samples used, making sure we never exceed m
   m_count = min(m, m_count + step_for_m) ;
endfor


%% =========== Part 3: Testing and evaluation ============
% Compute accuracy on our testing set.

[m_test, cols] = size(X_test) ;
printf("Test data dimension is : %d X %d\n", m_test, cols);

p = predict(theta, X_test);

printf("Train Accuracy: %f\n", mean(double(p == y_test)) * 100);



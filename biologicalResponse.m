%% Biological Response - Kaggle
%
%
page_output_immediately(1) ;

printf('\nBiological Response Learning\n') ;

%% =========== Part 1: Training Data Loading

% Load Data from a subdirectory called 'data'. 
% Assumptions for the training data file:
%    1. it is called 'train.mat'
%    2. the data is normalized
%    3. the y column vector is the ... column
printf("\nLoading TRAINING data ... ") ;

load("BioData.mat");

[m, cols] = size(X) ;
printf("done. Dimension for X is : %d X %d\n", m, cols);
printf("Dimension for y is : %d X %d\n", size(y));

% Determine whether the data is skewed
%[ isskewed, skewnesspc ] = skewness( y, 0.1 ) ;
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
initial_theta = zeros(size(X, 2), 1);

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
m_values = zeros(DATAPOINTS_NEEDED, 1) ;

m_count = step_for_m ;
for i = 1:DATAPOINTS_NEEDED
   X_train = X(1:m_count, :) ;
   y_train = y(1:m_count, :) ;

   printf("\nSize of X_train is: %d X %d", size(X_train)) ;
   printf("\nSize of y_train is: %d X %d", size(y_train)) ;
   printf("\nSize of initial_theta is: %d X %d", size(initial_theta)) ;
   % Optimize
   [theta, J, exit_flag] = ...
	fminunc(@(t)(lrCostFunction(t, X_train, y_train, lambda)), initial_theta, options);

   printf("\nLearning completed. Lambda used: %f # of samples used(m): %d\n",lambda, m_count) ;
   
   % store the values to be plotted. NEED TO ADD J_cv
   J_train_values(i) = J ;
   m_values(i) = m_count ;
   
   % increase the number of samples used, making sure we never exceed m
   m_count = min(m, m_count + step_for_m) ;
endfor


%% =========== Part 3: Testing and evaluation ============
% Compute accuracy on our testing set.

[m_test, cols] = size(X_test) ;
printf("Test data dimension is : %d X %d\n", m_test, cols);

p = predict(theta, X_test);

%printf('Train Accuracy: %f\n', mean(double(p == y_test)) * 100);



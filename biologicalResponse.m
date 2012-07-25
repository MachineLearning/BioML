%% Biological Response - Kaggle
%
%

fprintf('\nBiological Response Learning\n') ;

%% =========== Part 1: Training Data Loading

% Load Data from a subdirectory called 'data'. 
% Assumptions for the training data file:
%    1. it is called 'train.mat'
%    2. the data is normalized
%    3. the y column vector is the ... column

fprintf("\nLoading TRAINING data ... ") ;

load("data/BioData.mat");

[m, cols] = size(X) ;
fprintf("done. Dimension is : %d X %d\n", m, cols);

% Determine whether the data is skewed
[ isskewed, skewness ] = skewness( y ) ;
aux = '' ;
if (! isskewed )
   aux = ' NOT' ;
endif
fprintf("Data is%s skewed\n", aux) ;

fprintf("\nProgram paused (3 secs).\n");
pause (3) ;

%% =========== Part 2: Regularized Logistic Regression ============

fprintf("Learning using Logistic Regression. ") ;

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

index = 1 ;
J_train_values = zeros(DATAPOINTS_NEEDED, 1) ;
m_values = zeros(DATAPOINTS_NEEDED, 1) ;

m_count = step_for_m ;
while (m_count <= m)
   X_train = X(m_count, :) ;

   % Optimize
   [theta, J, exit_flag] = ...
	fminunc(@(t)(lrCostFunction(t, X_train, y, lambda)), initial_theta, options);

   fprintf("Learning completed. Lambda used: %f # of samples %used(m): %%d\n",lambda, m_count) ;
   
   % store the values to be plotted. NEED TO ADD J_test
   J_train_values(index) = J ;
   m_values(index) = m_count ;
   
   % increase the number of samples used, making sure we never exceed m
   m_count = max(m, m_count + step_for_m) ;
   index = index + 1 ;
endwhile


%% =========== Part 3: Testing and evaluation ============
% Compute accuracy on our testing set

[m_test, cols] = size(X_test) ;
fprintf("Test data dimension is : %d X %d\n", m_test, cols);

p = predict(theta, X_test);

%fprintf('Train Accuracy: %f\n', mean(double(p == y_test)) * 100);



%% Biological Response - Kaggle
%
%
%page_output_immediately(1) ;

addpath("LogisticRegression:NeuralNetwork:SupportVectorMachine", "-end") ;
printf('\nBiological Response Learning\n') ;

%% =========== Part 1: Training Data Loading

printf("\nLoading TRAINING data ... ") ;
load("BioData.mat");

% Add ones to the data matrix
%X = [ones(size(X,1), 1) X];

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

% printf("\nGraphing data. Please wait ...\n");
% visualiseData(X_train) ;

%% =========== Part 2: LEARNING ============


% Initialize fitting parameters
initial_theta = zeros(size(X_train, 2), 1);


% Determine the step to use to increase m. Assuming that we need 20
% datapoints. We can change that number if needed.
DATAPOINTS_NEEDED = 20 ;

step_for_m = ceil(m / DATAPOINTS_NEEDED) ;
m_count = step_for_m ;

m_values = zeros(DATAPOINTS_NEEDED, 1) ;
J_train_values = zeros(DATAPOINTS_NEEDED, 1) ;
J_cv_values = zeros(DATAPOINTS_NEEDED, 1) ;

t0 = clock() ;

%% ------- LOGISTIC REGRESSION PARAMETERS ------------
% Set regularization parameter lambda to 1.
% Set Options
lambda = 1;
options = optimset('GradObj', 'on', 'MaxIter', 400);
%% ---------------------------------------------------

%% ----------------- SVM Parameters ------------------
% got 72.0 with C=10, sigma=1
% got 76.13 with C=30, sigma=3    TIME:4909 secs 
% got 72.40 with C=50, sigma=5    TIME:???? secs
% got 77.20 with C=30, sigma=5    TIME:5702 secs
% got 75.33 with C=30, sigma=7    TIME:5474 secs
% got 78.00 with C=30, sigma=4    TIME:4250 secs
% got 76.40 with C=30, sigma=2    TIME:2804 secs
% got 76.26 with C=30, sigma=3.5  TIME:4255 secs
% got 78.40 with C=30, sigma=4.5  TIME:4981 secs
% got 79.06 with C=30, sigma=4.75 TIME:4868 secs
% got 75.46 with C=30, sigma=4.85 TIME:3769 secs
% got 78.66 with C=16, sigma=4.75 TIME:5468 secs
% got 77.46 with C=24, sigma=4.75 TIME:5158 secs

C = 20; 
sigma = 4.75;
%% ---------------------------------------------------

learning_algorithm = "SVM" ;

if (strcmp(learning_algorithm, "LR"))
  printf("Learning using Logistic Regression. ") ;
elseif (strcmp(learning_algorithm, "SVM"))
  printf("Learning using Support Vector Machine (Gaussian Kernel). ") ;
endif

fflush(stdout) ;

% Loop for increasing values of m. Needed to obtain pairs
% (J_train, m) and (J_cv, m) in order to be able to plot
% the corresponding error curves
for i = 1:DATAPOINTS_NEEDED
   X_used = X_train(1:m_count, :) ;
   y_used = y_train(1:m_count, :) ;

   % Optimize

   if (strcmp(learning_algorithm, "LR"))
     [theta, J, exit_flag] = ...
	fminunc(@(t)(lrCostFunction(t, X_used, y_used, lambda)), \
		initial_theta, options);

     message = sprintf("Lambda used: %.1f", lambda) ;

   elseif (strcmp(learning_algorithm, "SVM"))
     model= svmTrain(X_used, y_used, C, @(x1, x2) gaussianKernel(x1, \
								 x2, \
								 sigma)); \
	 
     message = sprintf("C used: %.2f, sigma used: %.2f", C, sigma) ;

   endif

   printf("\nLearning completed. %s -- # of samples used(m): %d\n",message, m_count) ;
   
   % store the values to be plotted. NEED TO ADD J_cv
   if (strcmp(learning_algorithm, "LR"))
     J_train_values(i) = J ;
     J_cv_values(i) = lrCostFunction(theta, X_cv, y_cv, lambda) ;
     m_values(i) = m_count ;
   endif
   
   % increase the number of samples used, making sure we never exceed m
   m_count = min(m, m_count + step_for_m) ;
endfor

elapsed_time = etime(clock(), t0) ;
printf("\nElapsed time (in secs): %d",elapsed_time) ;

if (strcmp(learning_algorithm, "LR"))
  plotErrors(m_values, J_train_values, J_cv_values) ;
endif

%% =========== Part 3: Testing and evaluation ============
% Compute accuracy on our testing set.

[m_test, cols] = size(X_test) ;
printf("\nTest data dimension is : %d X %d\n", m_test, cols);


if (strcmp(learning_algorithm, "LR"))
  p = predict(theta, X_test);
elseif (strcmp(learning_algorithm, "SVM"))
  p = svmPredict(model, X_test) ;
endif

printf("Train Accuracy: %.2f\n", mean(double(p == y_test)) * 100);



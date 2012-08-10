%% Biological Response - Kaggle
%
%

addpath("LogisticRegression","NeuralNetwork","SupportVectorMachine","Utils", "-end") ;
pr('\nBiological Response Learning\n') ;

%% =========== Part 1: Training Data Loading

pr("\nLoading TRAINING data ... ") ;
load("BioData.mat");

% Add ones to the data matrix
%X = [ones(size(X,1), 1) X];

% Split the training data into train, CV and test
[X_train, y_train, X_cv, y_cv, X_test, y_test] = segmentDataset(X, y) ;

[m, cols] = size(X_train) ;
pr("done. \nDimension for X_train is : %d X %d\n", m, cols);
pr("Dimension for y_train is : %d X %d\n", size(y_train));
pr("Dimension for X_cv is : %d X %d\n", size(X_cv));
pr("Dimension for y_cv is : %d X %d\n", size(y_cv));
pr("Dimension for X_test is : %d X %d\n", size(X_test));
pr("Dimension for y_test is : %d X %d\n", size(y_test));

% Determine whether the data is skewed
[ isskewed, skewnesspc ] = skewness( y_train, 0.1 ) ;
isskewed = false ;

aux = '' ;
if (! isskewed )
   aux = ' NOT' ;
endif
pr("Data is%s skewed\n", aux) ;
pr("\nProgram paused (3 secs).\n");
pause (3) ;

pr("\nGraphing data. Please wait ...\n");
%visualiseData(X_train) ;

%% =========== Part 2: LEARNING ============

%% ------- LOGISTIC REGRESSION PARAMETERS ------------
% Set regularization parameter lambda to 1.
% Set Options
lambda = 1;
options = optimset('GradObj', 'on', 'MaxIter', 400);
%% ---------------------------------------------------

%% ----------------- SVM Parameters ------------------
% ---------------------------
% Result     C    sigma  
% ---------------------------
%  72.00    10     1.0
%  76.13    30     3.0
%  72.40    50     5.0    
%  77.20    30     5.0    
%  75.33    30     7.0   
%  78.00    30     4.0    
%  76.40    30     2.0    
%  76.26    30     3.5  
%  78.40    30     4.5  
%  79.06    30     4.75 
%  75.46    30     4.85 
%  78.66    16     4.75 
%  76.13    20     4.75 
%  77.46    24     4.75 
%  77.33    27     4.75 
%  77.60    29     4.75 
%  77.33    32     4.75 
%  79.87     2     4.75 
%  79.07     2     4.75 
%  78.93   0.5     4.75 
%  78.53     4     4.75 
%  78.13     1     4.75 
%  77.73     4     4.75 
%  79.87     8     4.75 
%  80.00     9     4.75 
%  78.40     9     4.85 
%  76.53     9     4.80 
%  77.87     9     4.70 

C = 4; 
sigma = 4.75;

KERNELS = { "gaussian" ; "linear" ; "poly" } ;
kernel = KERNELS{3} ; %  

% For polynomial kernel  (g * (x1' * x2) - r) ^ d
% Result      C     g      r      d
%  58.00    9.00  1.00  16.00   3.00 
%  56.00    9.00  1.00   1.00   3.00 
%  52.80    9.00  1.00   1.00   4.00 
%  62.00    9.00  1.00   1.00   2.00 
%  50.00   32.00  1.00   1.00   2.00 
%  55.07    4.00  1.00   1.00   2.00 
%  60.00    4.00  1.00   2.00   2.00 
%  66.13    4.00  1.00   4.00   2.00
%  61.33    4.00  1.00   8.00   2.00
%  49.07    4.00  1.00   6.00   2.00
%  52.67    4.00  0.03   8.00   2.00
%  54.67    4.00  0.06   4.00   2.00
%  61.87    4.00  0.12   4.00   2.00
%  54.13    4.00  1.50   4.00   2.00

g = 1.5 ;
r = 4 ;
d = 2 ;

%% ---------------------------------------------------

learning_algorithm = "SVM" ;

if (strcmp(learning_algorithm, "LR"))
  pr("Learning using Logistic Regression. ") ;
elseif (strcmp(learning_algorithm, "SVM"))
  pr("Learning using Support Vector Machine (%s Kernel). ",kernel) ;

  % combine the CV data with the training data
  X_train = [ X_train ; X_cv ] ;
  y_train = [ y_train ; y_cv ] ; 
endif


% Determine the step to use to increase m. Assuming that we need 20
% datapoints. We can change that number if needed. For SVM we use only
% one datapoint, i.e. we go through the whole training set.

DATAPOINTS_NEEDED = 1 ;

step_for_m = ceil(size(X_train,1) / DATAPOINTS_NEEDED) ;
m_count = step_for_m ;

m_values = zeros(DATAPOINTS_NEEDED, 1) ;
J_train_values = zeros(DATAPOINTS_NEEDED, 1) ;
J_cv_values = zeros(DATAPOINTS_NEEDED, 1) ;

t0 = clock() ;

% Initialize fitting parameters
initial_theta = zeros(size(X_train, 2), 1);

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

     message = sprintf("Lambda: %.1f", lambda) ;

   elseif (strcmp(learning_algorithm, "SVM"))

     if (strcmp(kernel, "gaussian"))
         model= svmTrain(X_used, y_used, C, @(x1, x2) gaussianKernel(x1, \
							 x2, sigma));

         message = sprintf("C: %.2f, sigma: %.2f", C, sigma) ;

     elseif (strcmp(kernel, "linear"))
         model = svmTrain(X_used, y_used, C, @linearKernel, 1e-3, 20);

         message = sprintf("C used: %.2f", C) ;
  
     else
         model = svmTrain(X_used, y_used, C, @(x1, x2)
			  polynomialKernel(x1, x2, g, r, d));
 
	 message = sprintf("C: %.2f, g: %.2f, r: %.2f, d: %.2f", C, g, r, d) ;
     endif

   endif

   pr("\nLearning completed. %s -- # of samples used(m): %d\n",message, m_count) ;
   
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
pr("\nElapsed time (in secs): %d",elapsed_time) ;

if (strcmp(learning_algorithm, "LR"))
  plotErrors(m_values, J_train_values, J_cv_values) ;
endif

%% =========== Part 3: Testing and evaluation ============
% Compute accuracy on our testing set.

[m_test, cols] = size(X_test) ;
pr("\nTest data dimension is : %d X %d\n", m_test, cols);


if (strcmp(learning_algorithm, "LR"))
  p = predict(theta, X_test);
elseif (strcmp(learning_algorithm, "SVM"))
  p = svmPredict(model, X_test) ;
endif

pr("Train Accuracy: %.2f\n", mean(double(p == y_test)) * 100);



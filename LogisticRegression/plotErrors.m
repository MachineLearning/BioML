function plotErrors(m_values, J_train_values, J_cv_values)
% PLOTERRORS plots the training and cross validation errors for multiple
% sizes of samples.

plot (m_values, J_train_values, "-;J_{train};", "linewidth", 3, \
      m_values, J_cv_values, "-4;J_{cv};", "linewidth", 3) ;
xlabel('m (training set size)') ;
ylabel('error') ;

end
Using Libsvm for learning - 

  a) In order to run algorithms using libsvm the data and codes need to be located within the appropriate folder. 
  For e.g., in our case everything goes under /libsvm-3.12/matlab/ as we're using Octave.
  
  b) The functions I used are - libsvmwrite, libsvmread, svmtrain, and svmpredict.
  
  c) libsvmwrite/libsvmread - all .csv and .txt files need to be converted to libsvm format. This involves converting
  X into a sparse vector as required by the tool, writing it out in the correct format, and reading it back. Ref - lines    
  31 - 34 in the code.
     
  d) svmtrain - used twice in the code, initially to choose the best C and gamma value, and then finally to train the  
  model. For cross-validation the option to specify is '-v N', where N is the number of buckets, and the output of 
  svmtrain is the accuracy which is a scalar. Lines 58 - 75 contain this part.
  
  e) Without -v, the output of svmtrain is a non-scalar model that can be directly plugged into svmpredict. Some basic 
  parameters are kernel type, cost, gamma, and degree of cross validation.
  
  f) Additional details at http://www.csie.ntu.edu.tw/~cjlin/libsvm/.
  
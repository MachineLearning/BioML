function saveCsvAsMat
	% For Predicting a Biological Response
	% First row of train and test is feature labels, so remove
	% y is column 1 of train. No y in test
	% Graham Jones, 26.7.12
	trainset=csvread('train.csv');
	testset=csvread('test.csv');
	X=trainset(2:end,2:end);
	y=trainset(2:end,1);
	X_test=testset(2:end,:);
	save -mat7-binary BioData.mat X y X_test;
end


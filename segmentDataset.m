function [Xtrain,ytrain,Xcv,ycv,Xtest,ytest]=segmentDataset(X,y)
	% TEST TO SEE GITHUB WORKING
	% Segment feature matrix X and label vector y into training, cross validation
	% and test sets:
	% -- Randomly shuffle X,y in parallel before segmenting
	% -- Assume 60/20/20 split between train/cv/test
	% -- Return:
	%		Xtrain,ytrain	- to fit theta to
	%		Xcv,ycv			- to do model selection, diagnostics, etc.
	%		Xtest,ytest		- to estimate future performance on unseen data
	% Graham Jones, 23.7.12

	m=size(X,1);
	mtrain=round(m*0.6);
	mcv=round(m*0.2);
	mtest=m-(mtrain+mcv);
	assert(mtrain&mcv&mtest,"Dataset too small to segment");	% Ensure non-empty sets
	shuffled=randperm(m);
	Xtrain=X(shuffled(1:mtrain),:);
	ytrain=y(shuffled(1:mtrain));
	Xcv=X(shuffled(mtrain+1:mtrain+mcv),:);
	ycv=y(shuffled(mtrain+1:mtrain+mcv));
	Xtest=X(shuffled(mtrain+mcv+1:end),:);
	ytest=y(shuffled(mtrain+mcv+1:end));
end


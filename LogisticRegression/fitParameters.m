function [theta]=fitParameters(X,y,lambda,iters)
	% Use logistic regression to fit parameter vector theta
	% to labelled dataset X/y, with regularisation term lambda.
	% Include iters in argument list for when we want quick & dirty training
	% Graham Jones, 1.8.12

	initialtheta=zeros(size(X,2),1);
	options=optimset('GradObj','on','MaxIter',iters);
	[theta,J,exit_flag]=fminunc(@(t)(costFunctionRegLR(t,X,y,lambda)),initialtheta,options);

end


function [J,grad]=costFunctionRegLR(theta,X,y,lambda)
	% Compute cost of using theta for regularised logistic regression
	m=size(X,1);
	grad=zeros(size(theta));
	h=sigmoid(X*theta);
	J=-(1/m)*sum(y.*log(h)+(1-y).*log(1-h))+(lambda/(2*m))*sum(theta(2:end,:).^2);
	grad(1)=(1/m)*sum((h-y).*X(:,1));
	for j=2:length(theta)
		grad(j)=(1/m)*sum((h-y).*X(:,j))+(lambda/m)*theta(j);
	end
end


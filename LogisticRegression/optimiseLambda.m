function [lambda,theta]=optimiseLambda(Xtrain,ytrain,Xcv,ycv,verbose)
	% Find near-optimal value for regularisation term lambda:
	% -- Train model with range of lambda values on training set
	% -- Store parameter vector theta for each model
	% -- Calculate cost of using each theta on training and CV sets
	% -- Return lambda and associated theta that minimises Jcv
	% verbose argument makes function display output or be silent
	% Graham Jones, 26.7.12

	mtrain=size(Xtrain,1);
	n=size(Xtrain,2);
	mcv=size(Xcv,1);

	values=12;
	lambda=Jtrain=Jcv=zeros(values,1);
	theta=zeros(values,n);

	for i=1:values
		lambda(i)=(2^(i-2)*(i>1))/100;			% 0.00, 0.01, 0.02, 0.04...
		theta(i,:)=fitParameters(Xtrain,ytrain,lambda(i),10)';
		h=sigmoid(Xtrain*theta(i,:)');
		Jtrain(i)=-(1/mtrain)*sum(ytrain.*log(h)+(1-ytrain).*log(1-h))+(lambda(i)/(2*mtrain))*sum(theta(i,2:end)'.^2);
		h=sigmoid(Xcv*theta(i,:)');
		Jcv(i)=-(1/mcv)*sum(ycv.*log(h)+(1-ycv).*log(1-h))+(lambda(i)/(2*mcv))*sum(theta(i,2:end)'.^2);
		if (verbose)
			pr('%d of %d: lambda=%.2f, Jtrain=%.5f, Jcv=%.5f.\n',i,values,lambda(i),Jtrain(i),Jcv(i));
		end
	end
	[Jcvmin Jcvminidx]=min(Jcv);

	if (verbose)
		pr('Jcv min = %.5f, Jcv min index = %d, so lambda = %.2f\n',Jcvmin,Jcvminidx,lambda(Jcvminidx));
		close all;
		plot(Jtrain,'-','linewidth',2,'color','b');
		hold on;
		plot(Jcv,'-','linewidth',2,'color','m');
	end

	lambda=lambda(Jcvminidx);
	theta=theta(Jcvminidx,:)';

end


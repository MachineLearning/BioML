function visualiseData(X)
	% Reduce unlabelled dataset from n-dim to 2-dim using PCA and plot it
	% Graham Jones, 26.7.12

	k=2;						% No. of dimensions to reduce to (to plot: k=2)
	m=size(X,1);			% No. of examples

	% Remove x0 if it exists
	if (isequal(X(:,1),ones(m,1)))
		X=X(:,2:end);
	end

	n=size(X,2);			% No. of original dimensions

	% Do mean normalisation
	mu=mean(X);
	for j=1:n
		X(:,j)-=mu(j);
	end

	sigma=(1/m)*X'*X;		% Compute covariance matrix
	[U,S,V]=svd(sigma);	% Compute eigenvectors of covariance matrix
	Ureduce=U(:,1:k);		% Discard u(i) eigenvectors for i>k
	z=Ureduce'*X';			% Map points in n-dim X onto k-dim subspace Ureduce

	figure;
	plot(z(1,:),z(2,:),'+',"markersize",2);

end


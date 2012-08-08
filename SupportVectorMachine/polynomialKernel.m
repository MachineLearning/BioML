function sim = polynomialKernel(x1, x2, g, r, d)
%POLYNOMIALKERNEL returns a polynomial kernel between x1 and x2
%   sim = polynomialKernel(x1, x2, g, r, d) returns a polynomial
%   kernel between x1 and x2 using the formula (g * x1' * x2 + r) ^ d 
%   and returns the value in sim

% Ensure that x1 and x2 are column vectors
x1 = x1(:); x2 = x2(:);

% Compute the kernel
sim = (g * (x1' * x2) - r) ^ d ;  % polynomial kernel

end
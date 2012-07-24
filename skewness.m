function [ isskewed, skewness ] = skewness( y, threshold )
%SKEWNESS Skewness of a vector of zeros and ones
%   [ISSKEWED,SKEWNESS] = SKEWNESS(Y, THRESHOLD) returns a boolean
%   value, ISSKEWED = 1 if the data in Y is skewed and ISSKEWED = 0 if
%   the data in Y isn't skewed, which is determined with respect to the
%   specified THRESHOLD. THRESHOLD is a value between 0 and 1. If no
%   threshold is specified, the function uses the threshold of 0.1 (10%).
%   
%   The function also returns a value SKEWNESS which is the skewness
%   of the data and is a value between 0 and 1.

if nargin < 2
    threshold = 0.1;
end

skewness = sum(y == 1)/length(y);
skewness = min(skewness, 1 - skewness);

if skewness <= threshold
    isskewed = true;
else
    isskewed = false;
end

end


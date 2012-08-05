function pr(varargin)
	% Use instead of printf() when you want printing to be flushed
	% to console immediately but don't want to use
	% page_output_immediately(). Use as you would printf(), e.g.
	% pr('ABC, %d%d%d, 22/7 = %.5f\n',1,2,3,22/7);
	% Graham Jones, 5.8.12
	printf(varargin{:});
	fflush(stdout);
end


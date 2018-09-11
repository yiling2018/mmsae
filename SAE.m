function [ W ] = SAE(X, S, alph)
% min_W ||WX - S||^2 + alph||X - W^T S||^2
% Inputs:
%    X: (d x n) data matrix.
%    S: (k x n) semantic matrix.
%    alph: regularisation parameter.
% Return:
%    W: (k x d) projection matrix.
% adapt from https://github.com/Elyorcv/SAE

A = alph*S*S';
B = X*X';
C = (1 + alph) * S*X';
W = sylvester(A,B,C);
end


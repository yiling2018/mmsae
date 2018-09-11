function [ U, V ] = lsdr_mcplst( Xa, Xb, Y, params )
% Inputs:
%    Xa: (n x da) feature matrix of modality a
%    Xb: (n x db) feature matrix of modality b
%    Y: (n x c) label matrix
%    params.h: hidden dimemtion
% Return:
%    U: (n x h) code vector
%    V: (h x c) project matrix
% adapt from https://github.com/hsuantien/mlc_lsdr

shift = mean(Y);

[N, K] = size(Y);
Yshift = Y - repmat(shift, N, 1);

[~, ~, V] = svd(Yshift' * ridgereg_hat(Xa, 0.001) * Yshift + Yshift' * ridgereg_hat(Xb, 0.001) * Yshift, 0);
Vm = V(:, 1:params.h);
U = Yshift * Vm;
end


function H = ridgereg_hat(X, lambda)
% ``pseudo-inverse'' subject to regulariztion in ridge regression
%   needs lambda > 0

H = X/(X' * X + lambda * eye(size(X, 2))) * X';
end

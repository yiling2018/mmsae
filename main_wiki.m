addpath('util');

%% load data
load('data/wiki/data_norzm.mat');
load('data/wiki/icptv4_log_norzm.mat');
img_tr = double(img_tr);
img_te = double(img_te);
label_tr = scale2vec(label_tr);
label_te = scale2vec(label_te);

pca = 1; % perform PCA
if pca == 1
    options_=[];
    options_.PCARatio = 0.99;
    [eigvector,eigvalue] = myPCA(img_tr,options_);
    img_tr = img_tr*eigvector;
    img_te = img_te*eigvector;
    [eigvector,eigvalue] = myPCA(txt_tr,options_);
    txt_tr = txt_tr*eigvector;
    txt_te = txt_te*eigvector;
end

%% set parameters
params.h = 10; % hidden dimension

% parameter for stage 1
params.wy = 1;
params.wa = 1;
params.wb = 1;

% parameter for stage 2
params.max_iter = 5; % max number of iterations
params.alph = 0.01;
params.beta = 1;

%% training
% in here, we use data matrix where each colume is a sample.
V = img_tr';
T = txt_tr';
L = label_tr';

% stage 1
[C, ~] = lsdr_mcplst(V', T', L', params);

% stage 2
alph = params.alph;
beta = params.beta;
S = C';
U = S;
for i = 1 : params.max_iter
    Pi = SAE(V, U, alph);
    Pt = SAE(T, U, alph);
    U = (alph*(Pi*Pi') + alph*(Pt*Pt') + (2+beta)*eye(params.h))\((1+alph)*Pi*V + (1+alph)*Pt*T + beta*S);
    fprintf('%d\n', i);
end

%% testing 
img_te_proj = img_te*Pi';
txt_te_proj = txt_te*Pt';
%
fout = fopen('record_wiki.txt', 'a');
fprintf(fout, '[%s] --------------------------------------\n', datestr(now,31));
% test img2txt
fprintf('img search txt:\n');
fprintf(fout, 'img search txt:\n');
smatrix = img_te_proj * txt_te_proj';
test_s_map(smatrix, label_te, label_te, fout);
% test txt2img
fprintf('txt search img:\n');
fprintf(fout, 'txt search img:\n');
test_s_map(smatrix', label_te, label_te, fout);
%
fprintf(fout, '[%s] --------------------------------------\n', datestr(now,31));
fclose(fout);
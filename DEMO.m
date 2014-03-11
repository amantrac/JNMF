%Copyright (c) 2014 Yahoo! Inc.
%Copyrights licensed under the MIT License. See the accompanying LICENSE file for terms.
%Authors: Martin Saveski, Amin Mantrach
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% DEMO
% 
% Example of how to apply the Joint NMF and JNMF with Graph regularization
% methods on the NIPS dataset to predict the most likely authors of new 
% publications.
%
% The data is publicly available at:
% http://www.cs.nyu.edu/~roweis/data.html
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%
% Loading the data
%
load('data/nips12raw_str602.mat', 'counts', 'apapers');
load('data/T.mat');

%
% Transforming the data in a suitable matrix form
%
Nd = size(counts, 2);
Nw = size(counts, 1);
Na = size(apapers, 1);

Xu = sparse(Nd, Na);
for i=1:Na,
	Xu(apapers{i}, i) = 1.0;
end
clear apapers;

train_time = find(sum(T(:,1:12),2));
test_time  = find(sum(T(:,13),2));

vocab = 1:Nw;
Xu_train = Xu(train_time, :);
Xs_train = counts(:, train_time)';
Xs_train = Xs_train(:, vocab);

Xu_test = Xu(test_time, :);
Xs_test = counts(:, test_time)';
Xs_test = Xs_test(:, vocab);

%
% Preprocessing the data
%
train_authors = (sum(Xu_train) > 0);
Xu_train = Xu_train(:, train_authors);
Xu_test = Xu_test(:, train_authors);

[Xs_train, Xs_test] = tfidf(Xs_train, Xs_test);

%
% Running the Joint NMF model
%
k = 500;
alpha = 0.5;
lambda = 0.5;
epsilon = 0.001;
maxiter = 500;
verbose = true;

fprintf('This step may take some time ... \n');
tic
[~, Hs, Hu, ObjHistory] = JNMF(Xs_train,  L2_norm_row(Xu_train), k, alpha, lambda, epsilon, maxiter, verbose);
toc
plot(ObjHistory)

% Inference
W_test = Xs_test / Hs; 
W_test(W_test < 0) = 0;
JNMF_ranking = W_test*Hu;

jnmf_res = NDCG(JNMF_ranking, Xu_test);

%
% Running the Joint NMF model with Graph Regularization
%
k = 500;
alpha = 0.5;
lambda = 0.5;
epsilon = 0.001;
maxiter = 500;
verbose = true;
beta = 0.05;

% constructing the adjacency matrix
A = construct_A(Xs_train, 1, true);

fprintf('This step may take some time ... \n');
tic
[~, Hs, Hu, ObjHistory] = JNMF_GR(Xs_train,  L2_norm_row(Xu_train), A, k, alpha, beta, lambda, epsilon, maxiter, verbose);
toc
plot(ObjHistory)

% Inference
W_test = Xs_test / Hs; 
W_test(W_test < 0) = 0;
JNMF_GR_ranking = W_test*Hu;

jnmf_gr_res = NDCG(JNMF_GR_ranking, Xu_test);

%
% baseline: using the user profiles only
%
Ap = L2_norm_row(L2_norm_row(Xu_train)') * Xs_train;
BL_ranking = L2_norm_row(Xs_test * L2_norm_row(Ap)');
bl_res = NDCG(BL_ranking, Xu_test);

%
% Measuring NDCG
%
fprintf('JNMF: %f, JNMF-GR: %f, BL: %f \n', jnmf_res, jnmf_gr_res, bl_res);

% END

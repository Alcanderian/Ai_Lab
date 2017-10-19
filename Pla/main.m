%%
% PLA train session.
%
M_T = load('train.mat', '-ascii');
N = length(M_T(1, :))-1;

Tag_T = M_T(:, N+1);
M_T = M_T(:, 1:N);

W = pla_train(M_T, Tag_T, 100, 'ones');
save('w.mat', 'W', '-ascii');
%%
% PLA validate session.
%
M_V = load('val.mat', '-ascii');
W = load('w.mat', '-ascii');
N = length(M_V(1, :))-1;

Tag_V = M_V(:, N+1);
M_V = M_V(:, 1:N);

[Eval_V, Tag_R] = pla_val(M_V, Tag_V, W);
disp(Eval_V);
%%
% PLA Test session.
%
M_S = load('test.mat', '-ascii');
W = load('w.mat', '-ascii');
N = length(M_S(1, :))-1;

M_S = M_S(:, 1:N);

Tag_S = pla_test(M_S, W);
save('result.mat', 'Tag_S', '-ascii');

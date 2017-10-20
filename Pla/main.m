%%
% PLA train session.
%
M_T = load('train.mat', '-ascii');
N = size(M_T, 2)-1;

Tag_T = M_T(:, N+1);
M_T = M_T(:, 1:N);

P.mode = 'pocket';
P.init = 'ones';
P.progress = 1;
P.eval = 'f1';
P.iteration = 550;

W = pla_train(M_T, Tag_T, P);
save('w.mat', 'W', '-ascii');

disp('train done.');
disp(' ');

%%
% PLA validate session.
%
M_V = load('val.mat', '-ascii');
if ~exist('W', 'var')
    W = load('w.mat', '-ascii');
end
N = length(M_V(1, :))-1;

Tag_V = M_V(:, N+1);
M_V = M_V(:, 1:N);

[Eval_V, ~] = pla_eval(Tag_V, pla_test(M_V, W));

disp('evalution of validation:');
disp(Eval_V);
disp('validation done.');
disp(' ');

%%
% PLA Test session.
%
M_S = load('test.mat', '-ascii');
if ~exist('W', 'var')
    W = load('w.mat', '-ascii');
end
N = length(M_S(1, :))-1;

M_S = M_S(:, 1:N);
Tag_S = pla_test(M_S, W);

dlmwrite('result.csv', Tag_S);

disp('test done.');
disp(' ');

function [w] = pla_train(train, tag, K, init_method)
M = length(tag);
N = length(train(1, :))+1;
train = [ones(M, 1) train];

if strcmp(init_method, 'ones')
    w = ones(1, N);
elseif strcmp(init_method, 'rand')
    w = rand(1, N);
else
    w = zeros(1, N);
end

best_w = w;
best_e = M;

for k = 1:K
    err = 0;
    for i = 1:M
        if tag(i) ~= sign(w*train(i, :)')
            err = err+1;
            w = w+tag(i)*train(i, :);
        end
    end
    if err < best_e
        best_e = err;
        best_w = w;
    end
    if err == 0
        break;
    end
end

w = best_w;
end

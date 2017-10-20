function [tag_s] = pla_test(test, w)
M = size(test, 1);
test = [ones(M, 1) test];

tag_s = sign((w*test')');
end

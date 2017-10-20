function [predict] = pla_test(test, w)
M = size(test, 1);
predict = sign([ones(M, 1) test]*w');
end

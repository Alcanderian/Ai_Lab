function [evals, tag_r] = pla_val(val, tag, w)
tag_r = pla_test(val, w);

% ==========================================
% tag actual predict state(actual+2*predict)
% ------------------------------------------
% tp   1      1       3
% fn   1     -1      -1
% tn  -1     -1      -3
% fp  -1      1       1
% ==========================================
tfpn = tag+2*tag_r;

tp = sum(tfpn == 3);
fn = sum(tfpn == -1);
tn = sum(tfpn == -3);
fp = sum(tfpn == 1);

evals.accuracy = (tp+tn)/(tp+fp+tn+fn);
evals.recall = tp/(tp+fn);
evals.precision = tp/(tp+fp);
evals.f1 = 2*evals.precision*evals.recall/(evals.precision+evals.recall);
end

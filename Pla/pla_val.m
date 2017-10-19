function [evals, tag_r] = pla_val(val, tag, w)
M = length(tag);
val = [ones(M, 1) val];

tag_r = sign((w*val')');
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

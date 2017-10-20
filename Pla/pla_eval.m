function [evals, err] = pla_eval(actual, predict)
% ==========================================
% output args:
% ------------------------------------------
% evals:   accuracy, recall, precision, f1.
% err:     err vector of prediction.
% ==========================================
%
%
% ==========================================
%        state table of statistics
% ==========================================
% tag actual predict state(actual+2*predict)
% ------------------------------------------
% tp   1      1       3
% fn   1     -1      -1
% tn  -1     -1      -3
% fp  -1      1       1
% ==========================================
tfpn = actual+2*predict;

fn = tfpn == -1;
fp = tfpn == 1;

err = fn | fp;
tp = sum(tfpn == 3);
tn = sum(tfpn == -3);
fn = sum(fn);
fp = sum(fp);

evals.accuracy = (tp+tn)/(tp+fp+tn+fn);
evals.recall = tp/(tp+fn);
evals.precision = tp/(tp+fp);
evals.f1 = 2*evals.precision*evals.recall/(evals.precision+evals.recall);

if isnan(evals.f1)
    evals.f1 = 0;
end
end

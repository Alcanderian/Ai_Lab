function [w] = pla_train(train, tag, param)
% =============================================================================
% Input args: 
% -----------------------------------------------------------------------------
% train:              train matrix, vector stored by row.
% tag:                tag of each vector.
% param:              struct include all parameter for pla.
%   init:             method of initializing w.
%                       'ones': use ones.
%                       'zeros': use zeros.
%                       'rand': use random values.
%   progress:         if =1, evalution of each new best w is displayed.
%   eval:             how to renew best w:
%                       'f1': use f1 score to compare current w and the
%                         known best w.
%                       'accuracy': use f1 score to compare current w 
%                         and the known best w.
%   K:                number of iterations to perform. 
% =============================================================================
M = length(tag);
N = size(train, 2)+1;
train = [ones(M, 1) train];

% prepareing params.
if ~isfield(param, 'init')
    param.init = 'ones';
end
if ~isfield(param, 'eval')
    param.eval = 'accuracy';
end
if ~isfield(param, 'progress')
    param.progress = 1;
end

if strcmp(param.init, 'ones')
    w = ones(1, N);
elseif strcmp(param.init, 'rand')
    w = rand(1, N);
else
    w = zeros(1, N);
end

% init best w.
b_e = pla_val(train(:, 2:N), tag, w);
b_w = w;

for k = 1:param.K
    for i = 1:M
        % predict one.
        if tag(i) ~= sign(w*train(i, :)')
            % renew w.
            w = w+tag(i)*train(i, :);
            
            % eval current w.
            [evals, ~] = pla_val(train(:, 2:N), tag, w);
            
            if param.eval == 'f1'
                better = evals.f1 > b_e.f1;
            else
                better = evals.accuracy > b_e.accuracy;
            end
            
            % renew bset w.
            if better
                b_e = evals;
                b_w = w;
                
                if param.progress
                    disp(['iteration ', num2str(k),...
                        ', found better ', param.eval, ':']);
                    disp(b_e);
                end
            end
        end
    end
    
    if evals.accuracy == 1
        break;
    end
end

w = b_w;
end

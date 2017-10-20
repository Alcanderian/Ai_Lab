function [w] = pla_train(train, tag, param)
% =========================================================================
% input args:
% -------------------------------------------------------------------------
% train:              train matrix, vector stored by row.
% tag:                tag of each vector.
% param:              struct include all parameter for pla.
%   iteration:        number of iterations to perform.
%   mode:             mode of traing, all method wil terminate when the
%                       accuracy mactch 100%.
%                       pocket: evalute every w and find the best w in all
%                         result.
%                       normal: perfrom K times of iteration without
%                         evaluting every w.
%   init:             method of initializing w.
%                       ones: use ones.
%                       zeros: use zeros.
%                       rand: use random values.
%   progress:         if =1, evalution of each new best w is displayed.
%   eval:             how to renew best w:
%                       f1: use f1 score to compare current w and the
%                         known best w.
%                       accuracy: use f1 score to compare current w
%                         and the known best w.
% =========================================================================
M = length(tag);
N = size(train, 2)+1;
train = [ones(M, 1) train];

% prepareing params.
if ~isfield(param, 'iteration')
    param.iteration = 1;
end
if ~isfield(param, 'mode')
    param.mode = 'normal';
end
if ~isfield(param, 'init')
    param.init = 'ones';
end

if strcmp(param.mode, 'pocket')
    if ~isfield(param, 'eval')
        param.eval = 'accuracy';
    end
    if ~isfield(param, 'progress')
        param.progress = 1;
    end
end

if strcmp(param.init, 'ones')
    w = ones(1, N);
elseif strcmp(param.init, 'rand')
    w = rand(1, N);
else
    w = zeros(1, N);
end

% init best w, err.
[b_e, err] = pla_eval(tag, sign(train*w'));
for k = 1:param.iteration
    for i = 1:M
        if err(i)
            % renew current w.
            w = w+tag(i)*train(i, :);
            % eval current w.
            [e, err] = pla_eval(tag, sign(train*w'));
            if strcmp(param.mode, 'pocket')
                if strcmp(param.eval, 'f1')
                    better = e.f1 > b_e.f1;
                else
                    better = e.accuracy > b_e.accuracy;
                end
                
                % renew best w.
                if better
                    b_e = e;
                    b_w = w;
                    
                    if param.progress
                        disp(['iteration ', num2str(k),...
                            ', found better self ', param.eval, ':']);
                        disp(b_e);
                    end
                end
            end
        end
    end
    
    if e.accuracy == 1
        break;
    end
end

if strcmp(param.mode, 'pocket')
    w = b_w;
end
end

function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)
%lambda = 0;
%beta = 0;
% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
 
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 
 
% 学习率 自己定义的
alpha = 0.01;
 
% 隐藏神经元的个数是   25   = hiddenSize
 
% 计算隐藏层神经元的激活度
p = zeros(hiddenSize,1);
 
% 25x64
W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
% 64 X 25
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
% 25 X1
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
% 64 x 1
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);
 
% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
 
% costFunction 的第一项
%{
J_sparse = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));
%}
%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
% 
 
 
% 批量梯度下降法的一次迭代  data 64x10000
numPatches = size(data,2);
KLdist = 0;
 
% 25x10000
%a2 = zeros(size(W1,1),numPatches);
% 64x10000
%a3 = zeros(size(W2,1),numPatches);
 
%% 向前传输
% 25x10000  25x64 64x10000 
a2 = sigmoid(W1*data+repmat(b1,[1,numPatches]));
p = sum(a2,2);
a3 = sigmoid(W2 * a2 + repmat(b2,[1,numPatches]));
J_sparse = 0.5 * sum(sum((a3-data).^2));
 
%{
for curPatch = 1:numPatches
 
    % 计算激活值   
    % 25 X1 第二层的激活值   25x64  64x1
    a2(:,curPatch) = sigmoid(W1 * data(:,curPatch) + b1);
    % 计算隐藏层神经元的总激活值
    p = p + a2(:,curPatch); 
    % 64 x1 第三层的激活值
    a3(:,curPatch) = sigmoid(W2 * a2(:,curPatch) +b2);    
    %  计算costFunction的第一项
    J_sparse = J_sparse + 0.5 * (a3(:,curPatch)-data(:,curPatch))' * (a3(:,curPatch)-data(:,curPatch)) ;
end
%}
%% 计算 隐藏层的平均激活度
p = p /  numPatches ;
%% 向后传输 
 %64x10000
    residual3 = -(data-a3).*a3.*(1-a3);
    %25x10000
    tmp = beta * ( - sparsityParam ./ p + (1-sparsityParam) ./ (1-p));
    %  25x10000   25x64 64x10000  
    residual2 = (W2' * residual3 + repmat(tmp,[1,numPatches])) .* a2.*(1-a2);
    W2grad = residual3 * a2' / numPatches + lambda * W2 ;
    W1grad = residual2 * data'  / numPatches + lambda * W1 ;
    b2grad = sum(residual3,2) / numPatches; 
    b1grad = sum(residual2,2) / numPatches; 
 
    %{
for curPatch = 1:numPatches
 
    %  计算残差  64x1    
   % residual3 = -( data(:,curPatch) - a3(:,curPatch)) .* (a3 - a3.^2);
    residual3 = -(data(:,curPatch) - a3(:,curPatch)).* (a3(:,curPatch) - (a3(:,curPatch).^2));
    %  25x1         25x 64  *  64X1   ==>  25X1  .*   25X1
    residual2 = (W2' * residual3 + beta * (- sparsityParam ./ p + (1-sparsityParam) ./ (1-p))) .* (a2(:,curPatch) - (a2(:,curPatch)).^2);
  %  residual2 = (W2' * residual3 ) .* (a2(:,curPatch) - (a2(:,curPatch)).^2);
    % 计算偏导数值
    %   64 x25   =  64x1    1x25
    W2grad = W2grad + residual3 * a2(:,curPatch)';
    % 64 x1 = 64x1
    b2grad = b2grad + residual3;
    % 25x64  =  25x1  * 1x64
    W1grad = W1grad + residual2 *   data(:,curPatch)';
    % 25x1 = 25x1
    b1grad = b1grad + residual2;
    %J_sparse = J_sparse + (a3 - data(:,curPatch))' * (a3 - data(:,curPatch));
 
end
 
W2grad = W2grad / numPatches + lambda * W2;
W1grad = W1grad / numPatches + lambda * W1;
b2grad = b2grad / numPatches;
b1grad = b1grad / numPatches;
 
 %}
 
%% 更新权重参数   加上 lambda  权重衰减
W2 = W2 - alpha * ( W2grad  );
W1 = W1 - alpha * ( W1grad );
 
b2 = b2 - alpha * (b2grad );
b1 = b1 - alpha * (b1grad );
 
%% 计算KL相对熵
for j = 1:hiddenSize
    KLdist = KLdist + sparsityParam *log( sparsityParam / p(j) )   +   (1 - sparsityParam) * log((1-sparsityParam) / (1 - p(j)));
end
 
%% costFunction 加上 lambda 权重衰减
cost = J_sparse / numPatches + (sum(sum(W1.^2)) + sum(sum(W2.^2))) * lambda / 2  + beta * KLdist;
 
%cost = J_sparse / numPatches + (sum(sum(W1.^2)) + sum(sum(W2.^2))) * lambda / 2;
 
 
%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.
 
grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];
 
end
 
%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 
 
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));

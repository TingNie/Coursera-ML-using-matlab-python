function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
%% 对y进行处理 Y(find(y==3))= [0 0 1 0 0 0 0 0 0 0]; 用于 Feedforward cost function 1和2   
Y=[];
E = eye(num_labels);    % 要满足K可以是任意，则不能写eye(10)！！
for i=1:num_labels
    Y0 = find(y==i);    % 找到等于y=i的序列号,替换向量
    Y(Y0,:) = repmat(E(i,:),size(Y0,1),1);
end

%% unregularized Feedforward cost function lambda=0
% % 计算前向传输 Add ones to the X data matrix  -jin
% X = [ones(m, 1) X];
% a2 = sigmoid(X * Theta1');   % 第二层激活函数输出
% a2 = [ones(m, 1) a2];        % 第二层加入b
% a3 = sigmoid(a2 * Theta2');  
% 
% cost = Y .* log(a3) + (1 - Y ) .* log( (1 - a3));  % cost是m*K(5000*10)的结果矩阵  sum(cost(:))全部求和
% J= -1 / m * sum(cost(:));   


%% regularized Feedforward cost function lambda=1
% 计算前向传输 Add ones to the X data matrix  -jin
X = [ones(m, 1) X];
a2 = sigmoid(X * Theta1');   % 第二层激活函数输出
a2 = [ones(m, 1) a2];        % 第二层加入b
a3 = sigmoid(a2 * Theta2');  

temp1 = [zeros(size(Theta1,1),1) Theta1(:,2:end)];   % 先把theta(1)拿掉，不参与正则化
temp2 = [zeros(size(Theta2,1),1) Theta2(:,2:end)];
temp1 = sum(temp1 .^2);     % 计算每个参数的平方，再就求和
temp2 = sum(temp2 .^2);

cost = Y .* log(a3) + (1 - Y ) .* log( (1 - a3));  % cost是m*K(5000*10)的结果矩阵  sum(cost(:))全部求和
J= -1 / m * sum(cost(:)) + lambda/(2*m) * ( sum(temp1(:))+ sum(temp2(:)) );  


%% 计算 Gradient 
delta_1 = zeros(size(Theta1));
delta_2 = zeros(size(Theta2));

for t = 1:m
   % step 1
   a_1 = X(t,:)';          
%        a_1 = [1 ; a_1];
   z_2 = Theta1 * a_1;   
   a_2 = sigmoid(z_2);  
      a_2 = [1 ; a_2];
   z_3 = Theta2 * a_2;
   a_3 = sigmoid(z_3);
   % step 2
   err_3 = zeros(num_labels,1);
   for k = 1:num_labels     
      err_3(k) = a_3(k) - (y(t) == k);
   end
   % step 3
   err_2 = Theta2' * err_3;                % err_2有26行！！！
   err_2 = err_2(2:end) .* sigmoidGradient(z_2);   % 去掉第一个误差值，减少为25. sigmoidGradient(z_2)只有25行！！！
   % step 4
   delta_2 = delta_2 + err_3 * a_2';
   delta_1 = delta_1 + err_2 * a_1';
end

% step 5
Theta1_temp = [zeros(size(Theta1,1),1) Theta1(:,2:end)];
Theta2_temp = [zeros(size(Theta2,1),1) Theta2(:,2:end)];
Theta1_grad = 1 / m * delta_1 + lambda/m * Theta1_temp;
Theta2_grad = 1 / m * delta_2 + lambda/m * Theta2_temp ;
      



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

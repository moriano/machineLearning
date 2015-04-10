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

% COMPUTE H
debug = 0;

X = [ones(size(X)(1), 1) X];

a1 = X;


z2 = a1 * Theta1';
a2 = sigmoid(z2);
a2 = [ones(size(a2)(1), 1) a2];




z3 = a2 * Theta2';
a3 = sigmoid(z3);
%No need to add the ones to the last element...
%a3 = [ones(size(a3)(1), 1) a3];
h = a3;
%Compute y as vectors of K elements
Q = eye(num_labels);
yV = Q(y, :);

if debug == 1,
    printf('Theta1 IS %d x %d\n', size(Theta1)(1), size(Theta1)(2))
    printf('Theta2 IS %d x %d\n', size(Theta2)(1), size(Theta2)(2))
    printf('X IS %d x %d\n', size(X)(1), size(X)(2))
    printf('a1 IS %d x %d\n', size(a1)(1), size(a1)(2))
    printf('z2 IS %d x %d\n', size(z2)(1), size(z2)(2))
    printf('a2 IS %d x %d\n', size(a2)(1), size(a2)(2))
    printf('a3 IS %d x %d\n', size(a3)(1), size(a3)(2))
    printf('h IS %d x %d\n', size(h)(1), size(h)(2))
    printf('yV IS %d x %d\n', size(yV)(1), size(yV)(2))
    printf('y IS %d x %d\n', size(y)(1), size(y)(2))
end;

% END OF COMPUTE H

% COMPUTE COST FUNCTION J
costSum = 0;
j1 = -yV .* log(h);
j2 = (1-yV) .* log(1-h);

sumTotal = j1 - j2;
sumTotal = sum(sum(sumTotal));
J = (1/m) * sumTotal;
% END OF COMPUTE COST FUNCTION J

% REGULARIZATION

costReg1 = Theta1.^2;
costReg1(:, 1) = 0;
costReg1 = sum(sum(costReg1));


costReg2 = Theta2.^2;
costReg2(:, 1) = 0;
costReg2 = sum(sum(costReg2));


costReg = (lambda/(2*m))*(costReg1 + costReg2);



J = J + costReg;



% END OF REGULARIZATION


% BACK PROPAGATION
Delta = 0;


delta3 = a3 - yV;
delta2 = (delta3 * Theta2(:, 2:end)) .* sigmoidGradient(z2);

Delta1 = delta2' * (a1);
Delta2 = delta3' * a2;

if debug == 1,
    printf('delta3 IS %d x %d\n', size(delta3)(1), size(delta3)(2))
    printf('delta2 IS %d x %d\n', size(delta2)(1), size(delta2)(2))
    printf('Delta1 IS %d x %d\n', size(Delta1)(1), size(Delta1)(2))
    printf('Delta2 IS %d x %d\n', size(Delta2)(1), size(Delta2)(2))
    printf('Theta1_grad IS %d x %d\n', size(Theta1_grad)(1), size(Theta1_grad)(2))
    printf('Theta2_grad IS %d x %d\n', size(Theta2_grad)(1), size(Theta2_grad)(2))
end;

Theta1_grad = (1/m) .* Delta1;
Theta2_grad = (1/m) .* Delta2;


UnregTheta1 = Theta1_grad(:, 1);
UnregTheta2 = Theta2_grad(:, 1);


Theta1_grad +=  ((lambda/m) .* Theta1);
Theta1_grad(:, 1) = UnregTheta1;


Theta2_grad +=  ((lambda/m) .* Theta2);
Theta2_grad(:, 1) = UnregTheta2;

% END OF BACK PROPAGATION








% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

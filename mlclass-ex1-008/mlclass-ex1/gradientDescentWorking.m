function [theta, J_history] = gradientDescentWorking(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %







    % ============================================================
    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    J_history(iter)
        
    numRows = size(X)(1, 1);
    numColumns = size(X)(1, 2);

    thetaZero = theta(1, 1);
    thetaOne = theta(2, 1);

    h = X*theta;
    diff = h-y;

    sumForZero = 0;
    sumForOne = 0;
    for row = 1:numRows
        sumForZero = sumForZero + ((h(row)-y(row)) * X(row, 1));
        sumForOne = sumForOne  + ((h(row)-y(row)) * X(row, 2));
    end;

    thetaZeroAux = thetaZero;
    

    thetaZero = thetaZero - ((alpha/m) * sumForZero);
    thetaOne = thetaOne - ((alpha/m) * sumForOne);

    theta(1, 1) = thetaZero;
    theta(2, 1) = thetaOne;

    
    
    
    


end

end

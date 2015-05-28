function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%size(theta) % 2x1
%size(X) % 12x2
%size(y) % 12x1

temp = theta;
temp(1) = 0.0;
h = X * theta - y; % 12 x 1
J = J + 0.5 * (h' * h + lambda * temp' * temp) / m;

grad = (X' * h + lambda * temp) / m;









% =========================================================================

grad = grad(:);

end

function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

parameter = X*theta;
hypothesis_SET = sigmoid(parameter);

log_of_hypothesis_SET=log(hypothesis_SET); %log(h(x))
parameter_2=1-hypothesis_SET;
modified_log_of_hypothesis_SET=log(parameter_2); %log(1-h(x))

part_1 = -(y.*log_of_hypothesis_SET);
part_2=  -(1-y).*modified_log_of_hypothesis_SET;
part = part_1+part_2;

% calculate lambda part
square_SET=theta.^2;
square_SET(1)=0;


J = sum(part)/m + sum(square_SET)*lambda/(2*m);


% finding gradient




difference = hypothesis_SET.-y;


% gradient_1 for j=0
grad_part=difference.*X(:,1);
grad(1)=sum(grad_part)/m;


%========= for j>0 ============

for i=2:size(theta,1),
	grad_part=difference.*X(:,i);
	grad(i)=sum(grad_part)/m + lambda*theta(i)/m;


% =============================================================

end

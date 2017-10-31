function p = predict(Theta1,Theta2,X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);



a1 = [ones(m,1) X];

Z2=a1*Theta1';
a2=  [ones(size(Z2),1) sigmoid(Z2)];

Z3 =a2*Theta2';
a3 =  sigmoid(Z3);

[predict_max,index_max] = max(a3, [], 2);

p = index_max;


% =========================================================================


end

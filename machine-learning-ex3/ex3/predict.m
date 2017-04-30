function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

X = [ones(m, 1) X];

size(Theta1)

z2 = Theta1*X';

a2 = sigmoid(z2)';

temp = size(a2,1);
a2 = [ones(temp,1) a2];

size(a2)
size(Theta2)

z3 = Theta2*a2';

a3 = sigmoid(z3);

for i=1:num_labels
    if(a3(i)>=0.5) 
        a3(i) = 1;
    else 
        a3(i) = 0;
    end
end

[~,p] = max(a3);

p = p';

% =========================================================================


end

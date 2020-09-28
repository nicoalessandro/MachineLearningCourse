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
k=num_labels;

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


%Foward Propagation:
A1=X;%R(5000*400)
A1=[ones(m,1),A1]';%R(400*5000)
Z2=Theta1*A1;%R(25*401)*R(401*5000)=R(25*5000)
A2=sigmoid(Z2);%R(25*5000)
A2=[ones(1,m);A2];
Z3=Theta2*A2;%R(10*26)*R(26*5000)=R(10*5000)
A3=sigmoid(Z3);%R(10*5000)

% y={1,2,3,4,5,6,7,8,9,10} >>> y2={(1,0,...,0),(0,1,...,0),...,(0,0,...,1)}
y2=zeros(size(A3'));
for i=1:k
  y2(:,i)= (y==i);
endfor;

% Cost Function:
J= sum(log(A3(find(y2'==1)))(:))+sum(log(1-A3(find(y2'==0)))(:));
J=-J/m;
% Regularization:
reg=sum(sum( Theta1(:,2:size(Theta1,2)).^2 ))+sum(sum( Theta2(:,2:size(Theta2,2)).^2 ));
reg=lambda*reg/(2*m);
J=J+reg;

% Backpropagation:
%Theta1-->R(25*401)
%Theta2-->R(10*26)
deltha3=A3-y2';% R(10*5000)
deltha2=(Theta2')*deltha3.*A2.*(1-A2);% R(26*10)*R(10*5000).*R(26*5000)
Theta1_grad=deltha2(2:size(deltha2,1),:)*A1';%R(25*5000)*R(5000*401)= R(25*401)
Theta2_grad=deltha3*A2';%R(10*5000)*R(5000*26)*= R(10*26)

Theta1_grad=Theta1_grad/m;
Theta2_grad=Theta2_grad/m;
% +Regularization:
[f c]=size(Theta1);
Theta1Reg=lambda*[zeros(f,1),Theta1(:,2:c)]/m;
[f c]=size(Theta2);
Theta2Reg=lambda*[zeros(f,1),Theta2(:,2:c)]/m;

Theta1_grad=Theta1_grad+Theta1Reg;
Theta2_grad=Theta2_grad+Theta2Reg;






% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

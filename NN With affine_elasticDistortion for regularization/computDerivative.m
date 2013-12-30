function numgrad=computDerivative(theta,data,netconfig)
%% Numerical Gradient
mm=size(data,2);
numgrad = zeros(size(data));

EPSILON=10^-4;
for j=1:size(data,1)
    theta_temp1=data;
    theta_temp2=data;
    theta_temp1(j,:)=theta_temp1(j,:)+repmat(EPSILON,1,mm);
    theta_temp2(j,:)=theta_temp2(j,:)-repmat(EPSILON,1,mm);
    numgrad(j,:)=(linearPred(theta,theta_temp1,netconfig)-linearPred(theta,theta_temp2,netconfig))./(2*EPSILON);        
end


end


function sigm = sigmoid(x)
sigm = 1 ./ (1 + exp(-x));
end
function g = dsigmoid(z)
g=sigmoid(z).*(1-sigmoid(z));
end
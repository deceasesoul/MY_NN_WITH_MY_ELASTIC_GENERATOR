function numgrad = computeNumericalGradient(J, theta)

numgrad = zeros(size(theta));


EPSILON=10^-4;
for i=1:size(theta)
   theta_temp1=theta;
   theta_temp2=theta;
   theta_temp1(i)=theta_temp1(i)+EPSILON;
   theta_temp2(i)=theta_temp2(i)-EPSILON;
   numgrad(i)=(J(theta_temp1)-J(theta_temp2))./(2*EPSILON);
   fprintf('%d\n',i);
end


%% ---------------------------------------------------------------
end

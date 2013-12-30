function pred=DCnnPre(data,netconfig,theta)
%pred=DCnnPre(data,netconfig,theta)
dataSize=size(data,2);
kk=size(netconfig,2);
wcell=cell(kk-2,1);
bcell=cell(kk-2,1);
acell=cell(kk-1,1);

softmaxTheta=reshape(theta(1:netconfig(end-1)*netconfig(end)),[netconfig(end), netconfig(end-1)]);
pointer=1+netconfig(end-1)*netconfig(end);
for i=2:kk-1
    wcell{i-1}=reshape(theta(pointer:pointer+netconfig(i-1)*netconfig(i)-1), netconfig(i), netconfig(i-1));
    pointer=pointer+netconfig(i-1)*netconfig(i);    
    bcell{i-1}=theta(pointer:pointer+netconfig(i)-1);
    pointer=pointer+netconfig(i);
end


acell{1}=data;
for i=1:kk-2
    z=(wcell{i}*acell{i})+repmat(bcell{i},1,dataSize);
    acell{i+1}=sigmoid(z);
end
t = softmaxTheta*acell{end};     % (numClasses,N)*(N,M)
[~,pred]=max(t,[],1);
pred=pred';
pred=pred-1;

% M = bsxfun(@minus, t, max(t, [], 1));
% h = exp(M);
% h =  bsxfun(@rdivide, h, sum(h));

end
function sigm = sigmoid(x)
sigm = 1 ./ (1 + exp(-x));
end
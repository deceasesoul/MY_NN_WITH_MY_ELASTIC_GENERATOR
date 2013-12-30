function [cost, grad]=DCnnCostSoftMax_SGD_FixedMask(theta,netconfig,data,label,lambda,mcell,batchSize)
kk=size(netconfig,2);
wcell=cell(kk-2,1);

bcell=cell(kk-2,1);
acell=cell(kk-1,1);
dcell=cell(kk-2,1);
wgrandcell=cell(kk-2,1);
bgrandcell=cell(kk-2,1);
% numClasses=netconfig(end);

softmaxTheta=reshape(theta(1:netconfig(end-1)*netconfig(end)),[netconfig(end), netconfig(end-1)]);
softmaxTheta=mcell{end}.* softmaxTheta;

pointer=1+netconfig(end-1)*netconfig(end);
for i=2:kk-1
    wcell{i-1}=reshape(theta(pointer:pointer+netconfig(i-1)*netconfig(i)-1), netconfig(i), netconfig(i-1));
    pointer=pointer+netconfig(i-1)*netconfig(i);
    wcell{i-1}=mcell{i-1}.*wcell{i-1};
    bcell{i-1}=theta(pointer:pointer+netconfig(i)-1);
    pointer=pointer+netconfig(i);
end

% display_network(data);
%% ff

acell{1}=data;
for i=1:kk-2
    z=(wcell{i}*acell{i})+repmat(bcell{i},1,batchSize);
    acell{i+1}=sigmoid(z);
end

%% Softmax
M = softmaxTheta*acell{end};     % (numClasses,N)*(N,M)
M = bsxfun(@minus, M, max(M, [], 1));
h = exp(M);
h =  bsxfun(@rdivide, h, sum(h));
%% Cost
cost1=0;
for i=1:length(wcell)
    cost1=cost1+sum(sum(wcell{i}.^2));
end
cost1=lambda/2*cost1;

cost = -1/batchSize*sum(sum(label.*log(h)))+lambda/2*sum(sum(softmaxTheta.^2))+cost1;
%% Grad
thetagrad = mcell{end}.*(-1/batchSize*((label-h)*acell{end}')+lambda*softmaxTheta);%log(h)

dcell{end}=-(softmaxTheta' * (label - h)) .* acell{end} .* (1-acell{end});
for i=kk-3:-1:1
    dcell{i}= (wcell{i+1}' * dcell{i+1}) .* acell{i+1} .* (1-acell{i+1});
end

for i=kk-2:-1:1
    wgrandcell{i} = mcell{i}.*((1/batchSize) * dcell{i} * acell{i}')+lambda*wcell{i};
    bgrandcell{i} = (1/batchSize) * sum(dcell{i}, 2);
end

gv=getparaVector(wgrandcell,bgrandcell,pointer-1-size(thetagrad(:),1));

grad = [thetagrad(:) ;gv  ];
end
function sigm = sigmoid(x)
sigm = 1 ./ (1 + exp(-x));
end
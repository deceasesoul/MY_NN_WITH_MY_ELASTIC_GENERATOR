function pred=linearPred(theta,data,netconfig)
kk=size(netconfig,2);
wcell=cell(kk-1,1);
bcell=cell(kk-1,1);
acell=cell(kk,1);
pointer=1;
for i=2:kk
    wcell{i-1}=reshape(theta(pointer:pointer+netconfig(i-1)*netconfig(i)-1), netconfig(i), netconfig(i-1));
    pointer=pointer+netconfig(i-1)*netconfig(i);
    bcell{i-1}=theta(pointer:pointer+netconfig(i)-1);
    pointer=pointer+netconfig(i);
end
batchSize=size(data,2);
acell{1}=data;
for i=1:kk-1
    z=(wcell{i}*acell{i})+repmat(bcell{i},1,batchSize);
    acell{i+1}=sigmoid(z);
end
pred=acell{end};
end
function sigm = sigmoid(x)

sigm = 1 ./ (1 + exp(-x));
end
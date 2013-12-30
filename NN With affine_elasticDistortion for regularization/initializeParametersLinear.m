function theta = initializeParametersLinear(netconfig,para)
k=size(netconfig,2);
count=0;
for i=2:k
    count=count+netconfig(i-1)*netconfig(i);
end
for i=2:k
    count=count+netconfig(i);
end
theta=zeros(count,1);
r  = para / sqrt(sum(netconfig));   % we'll choose weights uniformly from the interval [-r, r]
p=1;
for i=2:k
    W = rand(netconfig(i-1)*netconfig(i),1) * 2 * r - r ;
    length=netconfig(i-1)*netconfig(i);
    theta(p:p+length-1)=W(:);
%     W = rand(netconfig(i),1) * 2 * r - r;
%     theta(p+length:p+length+netconfig(i)-1)=W(:);
    p=p+length+netconfig(i);
end

end


function theta = initializeParameters(para)
k=size(para,2);
count=0;
for i=2:k
    count=count+para(i-1)*para(i);
end
for i=2:k-1
    count=count+para(i);
end
theta=zeros(count,1);
r  = sqrt(6) / sqrt(sum(para));   % we'll choose weights uniformly from the interval [-r, r]
softMaxW=rand(para(end)*para(end-1),1)*2*r-r;
length=para(end)*para(end-1);
theta(1:length)=softMaxW;
p=1+length;
for i=2:k-1
    W = rand(para(i-1)*para(i),1) * 2 * r - r ;
    length=para(i-1)*para(i);
    theta(p:p+length-1)=W(:);
    p=p+length+para(i);
end
end


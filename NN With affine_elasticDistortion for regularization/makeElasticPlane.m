function model=makeElasticPlane(imageSize,sample)
netconfig=[2 200 200 1];
imagesize=imageSize;
theta = initializeParametersLinear(netconfig,30);
%% Prepare the sample
sample1=sample/imagesize;

 pred=linearPred(theta,sample1,netconfig);
grad=computDerivative(theta,sample1,netconfig);
gap=max(pred)-min(pred);
grad=grad/gap;


%% Translation
gradmean=mean(grad,2);
% gradmean=gradmean+(rand(2,1)-0.5)*5;
gradmean=repmat(gradmean,1,1024);
grad2=grad-gradmean;
% figure(3),hold on
% for i=1:imagesize
%     for j=1:imagesize
%         index=imagesize*(i-1)+j;
%         line([i,i+scale*grad2(1,index)],[j,j+scale*grad2(2,index)]);
%     end
%     fprintf('%d\n',i);
% end

% figure(3),hold off;
%% Rotation
zz=zeros(1,1024);
tgrad2=[grad2;zz];
sample1=sample-repmat([33/2;33/2],1,1024);
tsample2=[sample1;zz];
a=cross(tgrad2,tsample2,1);
s=mean(a(3,:));
O=6*s/(32*32)+(rand-0.5)* 0.2;
O=repmat(O,1,1024);
zz=zeros(2,1024);
O=[zz;O];
v=cross(O,tsample2,1);
v=v(1:2,:);
grad3=grad2-v;
% figure(4),hold on
% for i=1:imagesize
%     for j=1:imagesize
%         index=imagesize*(i-1)+j;
%         line([i,i+scale*grad3(1,index)],[j,j+scale*grad3(2,index)]);
%     end
%     fprintf('%d\n',i);
% end
% figure(4),hold off;
%% Scale

Imagescale=(rand-0.5)*0.1;
tsample3=sample1*Imagescale;
grad4=grad3+tsample3;
% scale=1.5;
% figure(5),hold on
% for i=1:imagesize
%     for j=1:imagesize
%         index=imagesize*(i-1)+j;
%         line([i,i+scale*grad4(1,index)],[j,j+scale*grad4(2,index)]);
%     end
%     fprintf('%d\n',i);
% end
% figure(5),hold off;
model=grad4;


%% 


end

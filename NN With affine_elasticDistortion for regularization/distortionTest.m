% pic = mxGetPr(prhs[0]);
% grad=mxGetPr(prhs[1]);
% imagesize=mxGetPr(prhs[2]);
% scale=mxGetPr(prhs[3]);
% cols=mxGetPr(prhs[4]);
labelt=loadMNISTLabels('train-labels.idx1-ubyte');
imagesize=32;
sample=zeros(2,imagesize*imagesize);
xx=1:imagesize;
yy=1:imagesize;
for i=1:size(xx,2)
    for j=1:size(yy,2)
        sample(:,(i-1)*size(xx,2)+j)=[xx(i);yy(j)];
    end
end
 DisTortionModel=makeElasticPlane(32,sample);
ran=randi(60000,1,50);
out=affine_elasticDistortion(dataout,DisTortionModel,32,1.5,ran);

display_network(out);
label=labelt(ran)

% figure,display_network(dataout(:,ran));
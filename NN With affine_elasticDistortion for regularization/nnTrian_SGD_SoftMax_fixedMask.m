netconfig=[28*28 200 200 10];
lamda=0;        
maskProb=0;  %DROP CONNECTION [0,1)
batchSize=100;
alpha=1;    %INITIAL LEARNING RATE
theta = initializeParameters(netconfig); 
momenton=0.5;    
v=zeros(size(theta));
%% Prepare the sample
trainSize=50000;
data=loadMNISTImages('train-images.idx3-ubyte');
labelt=loadMNISTLabels('train-labels.idx1-ubyte');


data32=zeros(32*32,size(data,2));

for i=1:size(data,2);
    temptemppp=zeros(32,32);
    temptemppp(3:30,3:30)=reshape(data(:,i),[28 28]);
    data32(:,i)=reshape(temptemppp,[32*32,1]);
end



% load('trainlabel.mat');
% labelt=trainlabel;
% validationdata32=data32(:,trainSize+1:end);
validationdata=data(:,trainSize+1:end);
validatoinlabel=labelt(trainSize+1:end);
trainlabel=zeros(10,trainSize);
traindata=data32(:,1:trainSize);
labeltt=labelt(1:trainSize);
for i=1:trainSize
    trainlabel(labeltt(i)+1,i)=1;
end


%% Check

% [cost, grad]=DCnnCostSoftMax_SGD_FixedMask(theta, netconfig,traindata,trainlabel,lamda,maskset,batchSize);
% numgrad = computeNumericalGradient( @(x) DCnnCostSoftMax_SGD_FixedMask(x,netconfig,traindata,trainlabel,lamda,maskset,batchSize), theta);
%
% disp([numgrad grad]);
% diff = norm(numgrad-grad)/norm(numgrad+grad);
% disp(diff);
%% Distort Sample
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


% % for i=1:size(validationdata,2)
% %     temptemppp=zeros(32,32);
% %     temptemppp(3:30,3:30)=reshape(validationdata(:,i),[28 28]);
% %     validationdata32(:,i)=reshape(temptemppp,[32*32,1]);
% % end
% ran=1:size(validationdata32,2);
% validationdata32out=affine_elasticDistortion(validationdata32,DisTortionModel,32,1.5,ran);
% for i=1:size(validationdata32out,2)    
%     temptemppp=reshape(validationdata32out(:,i),[32 32]);
%     temptemppp=temptemppp(3:30,3:30);
%     validationdata(:,i)=reshape(temptemppp,[28*28,1]);
% end
% display_network(validationdata);
%% FOR DROP CONNECTION
maskset=makeMask(netconfig,maskProb);
%%
batchTrainData=zeros(28*28,batchSize);
batchTrainLabel=zeros(10,batchSize);
hold on;
iter=0;
pre_iter=iter;
pre_acc=0;
plot(iter,1,'.');
step=1;
modeliter=1;
while(true)
    %     maskset=makeMask(netconfig,maskProb);
    if modeliter>2
        DisTortionModel=makeElasticPlane(32,sample);
        modeliter=1;
    else
        modeliter=modeliter+1;
    end
    
    %     display_network(out);
    for i=1:150
        
        ran=randi(trainSize,1,batchSize);
        out=affine_elasticDistortion(traindata,DisTortionModel,32,1.8,ran);
%         display_network(out);
        for tt=1:batchSize
            tempPic=reshape(out(:,tt),[32 32]);
            batchTrainPic=tempPic(3:30,3:30);
            batchTrainData(:,tt)=reshape(batchTrainPic,[28*28 1]);
        end
%         imshow(reshape(batchTrainData(:,3),[28 28]))
%         display_network(batchTrainData);
        batchTrainLabel=trainlabel(:,ran);
        alpha=rand(1,1)*step;
%         maskProb=rand(1,1);
        %% Make mask
        
        %% COST FUNCTION
        [cost, grad]=DCnnCostSoftMax_SGD_FixedMask(theta, netconfig,batchTrainData,batchTrainLabel,lamda,maskset,batchSize);
        v=momenton*v+alpha*grad;
        theta=theta-v;
        %         [cost, grad]=DCnnCostSoftMax_SGD_FixedMask(theta, netconfig,traindata,trainlabel,lamda,maskset2,batchSize);
        %         theta=theta-alpha*step*grad;
    end
    step=step*0.9996;
    set(gca,'ygrid','on');
    pre=DCnnPre(validationdata,netconfig,theta);
    acc = mean(validatoinlabel(:) == pre(:));
    
    fprintf('Step: %d\nacc: %0.3f%%\n',step, acc * 100);
    
    iter=iter+1;
    %     plot(iter,acc,'x');
    plot([pre_iter,iter],[pre_acc,acc],'-k');
    drawnow;
    pre_iter=iter;
    pre_acc=acc;
end
%% TEST
testdata=loadMNISTImages('train-images.idx3-ubyte');
testlabel=loadMNISTLabels('train-labels.idx1-ubyte');

% test32=zeros(32*32,size(testdata,2));
%  for i=1:size(data,2);
%     temptemppp=zeros(32,32);
%     temptemppp(3:30,3:30)=reshape(data(:,i),[28 28]);
%     test32(:,i)=reshape(temptemppp,[32*32,1]);
% end
%  DisTortionModel=makeElasticPlane(32,sample);
%  testindex=1:size(test32,2);
% test32=affine_elasticDistortion(test32,DisTortionModel,32,1.5,ran);
% for i=1:size(test32,2)    
%     temptemppp=reshape(test32(:,i),[32 32]);
%     temptemppp=temptemppp(3:30,3:30);
%     testdata(:,i)=reshape(temptemppp,[28*28,1]);
% end
pre=DCnnPre(testdata,netconfig,theta);
 acc = mean(testlabel(:) == pre(:));

 
fprintf('Test acc: %0.3f%%\n', acc * 100);

function maskset=makeMask(netconfig,p)
    kk=size(netconfig,2)-1;
    maskset=cell(kk,1);
    for i=1:kk        
        maskset{i}=rand(netconfig(i+1), netconfig(i));
        maskset{i}(maskset{i}<p)=0;
        maskset{i}(maskset{i}~=0)=1;
    end
end
function paraVector=getparaVector(wgrandcell,bgrandcell,count)
    paraVector=zeros(count,1);
    p1=1;
    for d=1:numel(wgrandcell)
        p2=size(wgrandcell{d}(:),1);
        paraVector(p1:p1+p2-1) = wgrandcell{d}(:);
        p1=p1+p2;
        p2=size(bgrandcell{d}(:),1);
        paraVector(p1:p1+p2-1) = bgrandcell{d}(:);
        p1=p1+p2;
    end
    
end
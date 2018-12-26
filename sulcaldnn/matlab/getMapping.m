function result = getMapping(curve,sphere,label,distmap,ref_sphere)

    v=read_vtk(ref_sphere);
    [phi, theta] = cart2sph(v(:,1),v(:,2),v(:,3));
    pc = [phi theta [1:size(v,1)]'];
    pc=round(pc,4);
    pcr=sortrows(pc,[1 2]);

    v1=read_vtk(sphere);   
    sp=textread(curve,'%d')+1;   
    
    [k,d]=dsearchn(v,v1(sp,:));  
    %[k,d]=dsearchn(v,v1(a2+1,:));  
    
    map=squeeze(label);
%     figure(1); imagesc(map); colorbar;
    % TODO optimize the threshold levels by empirical testing
    map(distmap>0.1)=0;
    s1=find(map>0.7); % fixing to 0.6 for label 2
    s2=pcr(s1+2,3);
%     figure(1); imagesc(map); colorbar;
%     figure(2); imagesc(distmap); colorbar;
    result1=intersect(k,s2);
    
    id2=[];
    for i = 1: length(result1)
        id = find(k == result1(i));
        id2(i) = sp(id(1))-1;
    end
    
    result=id2;


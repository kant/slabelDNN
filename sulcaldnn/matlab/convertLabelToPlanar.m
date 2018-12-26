function convertToPlanar(pc,label_file,outdir,label_out)
    
    % label file
    if strcmp(label_file,'')~=1
        original = load(label_file);
        new_orig=original(pc(:,3));
        img=reshape(new_orig(3:end),512,[]);

        % save label mat file
        save(sprintf('%s/%s',outdir, label_out),'img');
    end


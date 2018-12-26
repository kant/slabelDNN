function convertToPlanar(pc,distmap_file,spectra_files,outdir,feature_out)
    
    % distmap - do not normalize
    original = load(distmap_file);
    new_orig=original(pc(:,3));
    img(:,:,1) =reshape(new_orig(3:end),512,[]);
    
    % spectra 
    for j=1:length(spectra_files)
        original = load(spectra_files{j});
        new_orig=img_normalize(original(pc(:,3))); %normalize spectra
        img(:,:,j+1) =reshape(new_orig(3:end),512,[]);
    end

    % save feature mat file
    save(sprintf('%s/%s',outdir, feature_out),'img');


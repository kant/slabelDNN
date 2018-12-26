function generateSpectra(tgtSurface,srcSurface,U1_tgt,outdir,outfile_prefix)
    g1r=gifti(tgtSurface);
    g1=gifti(srcSurface);
    [U1,U2]= SpectralSurfaceSpectra_v1(g1r,g1);
    % Check sign
    dotp  = diag( sign(U1)' * sign(U1_tgt) ); % dot product between corresponding E2 and E1
    signf = 1 - 2*(dotp<0);  % column vector telling if there is a signflip (-1) or not (+1)    
    % Flip sign
    U2 = bsxfun(@times,U2,signf');  % Align E2 in the same direction as E1
    getPropertyOnSurface(g1.vertices,g1.faces,sprintf('%s/%s.vtk',outdir,outfile_prefix),{'E1','E2','E3','E4','E5'},[U2]);
    % Create seperate txt file for each spectra
    for j=1:size(U2,2)
            metric_file = sprintf('%s/%s_%d.txt',outdir,outfile_prefix,j);
            fp = fopen(metric_file,'w+');
            fprintf(fp,'%f\n',U2(:,j));
            fclose(fp);
    end
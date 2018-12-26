function generateCurvesFun(ref_sphere, subj_sphere, curve,subj,hemi,datadir,labelfile)

% initialize
p_thr=0.2; f_thr=0.8;

if strcmp(hemi,'lh')==1
	labelnames={'1','2','3','4','5','6','8','9'};
	labelvalues={'CS_left','STS_left','SFS_left','IFS_left','OTS_left','CingS_left','CalcS_left','OLF_left'};
else 
	labelnames={'15','16','17','18','19','20','22','23'};
	labelvalues={'CS_right','STS_right','SFS_right','IFS_right','OTS_right','CingS_right','CalcS_right','OLF_right'};
end

% Generate curves
   
    fp3 = fopen(labelfile,'w');
    for j=1:length(labelnames)
	outfile=sprintf('%s/output_%s_%s_%s_spectra10.mat',datadir,subj,hemi,labelnames{j});
        if ~exist(outfile,'file')
            fprintf('%s doesnt exist\n',outfile);
        else
            labelmap=load(outfile);
            d1=squeeze(labelmap.X2_test(1,:,:,1));
            p1=labelmap.y2_pred(1,:,:); 
            p1_result=getMapping(curve,subj_sphere,p1,d1,ref_sphere);
            p1_result = postProcessCurveIndicesV3(p1_result',curve(1:end-7),curve(1:end-7),p_thr,f_thr)
    %         % Write slabel files
            curveName=labelvalues{j}
            fprintf(fp3,'%s\n', curveName);
            for k=1:length(p1_result)
                fprintf(fp3,'%d ', p1_result(k));
            end
            fprintf(fp3,'\n');
        end
    end
    fclose(fp3);
end

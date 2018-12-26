addpath('~/masimatlab/trunk/users/parvatp/utils');
addpath(genpath('/home/local/VANDERBILT/parvatp/masimatlab/trunk/xnatspiders/matlab/surf_quant_pipeline_v1_0_0/'));
libpath='/home/local/VANDERBILT/lyui/Projects/Spider/SulcalCurve/curve_labeling_pipeline_v1_0_0';
ref_sphere='/share5/parvatp/SulcalLabelData/reference_sphere_0316.vtk';
labelnames={'label1','label2','label3','label4','label5','label6','label7','label8','label9'};
labelvalues={'CS_left','STS_left','SFS_left','IFS_left','OTS_left','CingS_left','OPS_left','CalcS_left','OLF_left'};
p_thr=0.2; f_thr=0.8;

datafile=readtable('/home-nfs/masi-shared-home/home/local/VANDERBILT/parvatp/masimatlab/trunk/users/parvatp/SurfaceSoftware/SulcalCurves-DL/Sep172018/data/blsa_groundtruthAll_filter.csv','delimiter',',');
datadir='/share5/parvatp/SulcalLabelData/Oct082018/Experiments2/8Features/outputs/';
outdir=sprintf('%s/Results',datadir);
mkdir(outdir)
% 
for i=1:size(datafile,1)
     subjdir=datafile.fPath{i};
     subj=datafile.Session{i}

    subj_sphere = sprintf('/share5/parvatp/SulcalLabelData/Apr062018/preprocessed/SphereRotation/%s_lh_sphere_reg.vtk',subj);
   
    curve=sprintf('%s/Curve/lh.target_image_GMimg_centralSurf.scurve',subjdir);
    fp3 = fopen(sprintf('%s/%s_PredLabelN3.slabel',outdir,subj),'w');
    fp1 = fopen(sprintf('%s/%s_CurveLabelN3.slabel',outdir,subj),'w');
    for j=1:length(labelnames)
         close all
        if ~exist(sprintf('%s/output_0001_%s_spectra10.mat',datadir,labelnames{j}),'file')
            fprintf('%s/output_0001_%s_spectra10.mat doesnt exist\n',datadir,labelnames{j});
        else
            labelmap=load(sprintf('%s/output_0001_%s_spectra10.mat',datadir,labelnames{j}));
            d1=squeeze(labelmap.X2_test(i,:,:,1));
            p1=labelmap.y2_pred(i,:,:); 
            c1=labelmap.y2_test(i,:,:); 
            p1_result=getMapping(curve,subj_sphere,p1,d1,ref_sphere);
            p1_result = postProcessCurveIndicesV3(p1_result',subjdir,libpath,p_thr,f_thr)
            cl_result=getMapping(curve,subj_sphere,c1,d1,ref_sphere);

    %         % Write slabel files
            curveName=labelvalues{j}
            fprintf(fp3,'%s\n', curveName);
            for k=1:length(p1_result)
                fprintf(fp3,'%d ', p1_result(k));
            end
            fprintf(fp3,'\n');
            
    %         % Write slabel files
            curveName=labelvalues{j}
            fprintf(fp1,'%s\n', curveName);
            for k=1:length(cl_result)
                fprintf(fp1,'%d ', cl_result(k));
            end
            fprintf(fp1,'\n');
            
        end
    end
    fclose(fp3);
    fclose(fp1);
end

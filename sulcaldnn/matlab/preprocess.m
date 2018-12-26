function preprocess(surf_file,scurve,sphere_file,parcel_file,hemi,subj,outdir)

if ~exist(outdir,'dir') mkdir(outdir); end
featuredir=sprintf('%s/Features/%s',outdir,subj);
if ~exist(featuredir,'dir') mkdir(featuredir); end


ref_sphere='/extra/sulcaldnn/data/reference_sphere_0316.vtk';
% generate distance map
system(sprintf('/extra/sulcaldnn/bin/SulcalMap -i %s -s %s -o %s/%s_%s ',surf_file, scurve,featuredir,subj,hemi));
% generate mean curvature
system(sprintf('/extra/sulcaldnn/bin/MeshProperty -i %s -m -o %s/%s_%s_MC --tensorsmoothing 5 --surfsmoothing 5',surf_file, featuredir,subj,hemi));
% generate parcel data
system(sprintf('cp %s %s/%s_%s.parcel.txt',parcel_file, featuredir,subj,hemi));
% generate spectra
% Load reference spectra and generate consistent spectra for all data
load('/extra/sulcaldnn/data/refSpectra.mat');
if strcmp(hemi,'lh')==1
    tgtSurface='/extra/sulcaldnn/data/lh_spectra_tgtSurface.vtk';
    U1=U1_tgtl;
    rotation='0.0913,0.8168,0.5696,0,-0.1071,-0.5606,0.8211,0,0.9900,-0.1360,0.0363,0';
else if strcmp(hemi,'rh')==1
        tgtSurface='/extra/sulcaldnn/data/rh_spectra_tgtSurface.vtk';
        U1=U1_tgtr;
        rotation='-0.118571,0.903104,0.412727,0,-0.566091,-0.402974,0.719133,0,0.815771,-0.148373,0.55902,0';
    else
        display('hemisphere should be specified as lh/rh');
    end
end

% perform smoothing on surface
% [filepath,name,ext] = fileparts(surf_file)
% smooth_surf=sprintf('%s/%s_%s.s10.vtk',featuredir,subj,hemi);
% system(sprintf('mris_smooth %s %s',surf_file, smooth_surf));

generateSpectra(tgtSurface,surf_file,U1,featuredir,[subj '_' hemi '.spectra'])

% Rotate sphere

% Make sphere
%system(sprintf('bin/make_sphere_freesurfer %s %s/%s_%s.sphere.vtk 1',surf_file, outdir,subj,hemi));
% uniform_sphere = 'data/sphere_327680.vtk';

rotsphdir=sprintf('%s/RotatedSphere',featuredir);
if ~exist(rotsphdir,'dir') mkdir(rotsphdir); end

%Rotate
system(sprintf('/extra/sulcaldnn/bin/MeshDeform -i %s -o %s/%s_%s_sphere_reg.vtk --matrix %s',sphere_file,rotsphdir,subj,hemi,rotation));

% Resample
resampledir=sprintf('%s/Resampled',featuredir);
if ~exist(resampledir,'dir') mkdir(resampledir); end

system(sprintf('/extra/sulcaldnn/bin/SurfRemesh2 -p %s/%s_%s.absDist.txt -t %s/%s_%s_sphere_reg.vtk -r %s --outputProperty %s/%s_%s_distmap_res --noheader',featuredir,subj,hemi,rotsphdir,subj,hemi,ref_sphere,resampledir,subj,hemi));
system(sprintf('/extra/sulcaldnn/bin/SurfRemesh2 -p %s/%s_%s_MC.H.txt -t %s/%s_%s_sphere_reg.vtk -r %s --outputProperty %s/%s_%s_mc_res --noheader',featuredir,subj,hemi,rotsphdir,subj,hemi,ref_sphere,resampledir,subj,hemi));
system(sprintf('/extra/sulcaldnn/bin/SurfRemesh2 -p %s/%s_%s.parcel.txt -t %s/%s_%s_sphere_reg.vtk -r %s --outputProperty %s/%s_%s_parcel_res --noheader',featuredir,subj,hemi,rotsphdir,subj,hemi,ref_sphere,resampledir,subj,hemi));
system(sprintf('/extra/sulcaldnn/bin/SurfRemesh2 -p %s/%s_%s.spectra_1.txt -t %s/%s_%s_sphere_reg.vtk -r %s --outputProperty %s/%s_%s_spectra_1_res --noheader',featuredir,subj,hemi,rotsphdir,subj,hemi,ref_sphere,resampledir,subj,hemi));
system(sprintf('/extra/sulcaldnn/bin/SurfRemesh2 -p %s/%s_%s.spectra_2.txt -t %s/%s_%s_sphere_reg.vtk -r %s --outputProperty %s/%s_%s_spectra_2_res --noheader',featuredir,subj,hemi,rotsphdir,subj,hemi,ref_sphere,resampledir,subj,hemi));
system(sprintf('/extra/sulcaldnn/bin/SurfRemesh2 -p %s/%s_%s.spectra_3.txt -t %s/%s_%s_sphere_reg.vtk -r %s --outputProperty %s/%s_%s_spectra_3_res --noheader',featuredir,subj,hemi,rotsphdir,subj,hemi,ref_sphere,resampledir,subj,hemi));
system(sprintf('/extra/sulcaldnn/bin/SurfRemesh2 -p %s/%s_%s.spectra_4.txt -t %s/%s_%s_sphere_reg.vtk -r %s --outputProperty %s/%s_%s_spectra_4_res --noheader',featuredir,subj,hemi,rotsphdir,subj,hemi,ref_sphere,resampledir,subj,hemi));
system(sprintf('/extra/sulcaldnn/bin/SurfRemesh2 -p %s/%s_%s.spectra_5.txt -t %s/%s_%s_sphere_reg.vtk -r %s --outputProperty %s/%s_%s_spectra_5_res --noheader',featuredir,subj,hemi,rotsphdir,subj,hemi,ref_sphere,resampledir,subj,hemi));

% Convert to Planar
%     % 2.3 Load reference sphere
v=read_vtk(ref_sphere);
[phi, theta] = cart2sph(v(:,1),v(:,2),v(:,3));
%
pc = [phi theta [1:size(v,1)]'];
pc=round(pc,2);
pc=sortrows(pc,[1 2]);
%         % Left hemisphere
distmap_file = sprintf('%s/%s_%s_distmap_res.absDist.txt',resampledir,subj,hemi);
spectra1_file = sprintf('%s/%s_%s_spectra_1_res.spectra_1.txt',resampledir,subj,hemi);
spectra2_file = sprintf('%s/%s_%s_spectra_2_res.spectra_2.txt',resampledir,subj,hemi);
spectra3_file = sprintf('%s/%s_%s_spectra_3_res.spectra_3.txt',resampledir,subj,hemi);
spectra4_file = sprintf('%s/%s_%s_spectra_4_res.spectra_4.txt',resampledir,subj,hemi);
spectra5_file = sprintf('%s/%s_%s_spectra_5_res.spectra_5.txt',resampledir,subj,hemi);
mc_file = sprintf('%s/%s_%s_mc_res.H.txt',resampledir,subj,hemi);
parcel_file = sprintf('%s/%s_%s_parcel_res.parcel.txt',resampledir,subj,hemi);
other_files={spectra1_file,spectra2_file,spectra3_file,spectra4_file,spectra5_file,mc_file,parcel_file};
%
feature_out = [subj '_' hemi '_feature.mat'];
planardir=sprintf('%s/Planar',outdir);
if ~exist(planardir,'dir') mkdir(planardir); end
convertToPlanar(pc,distmap_file,other_files,planardir,feature_out)

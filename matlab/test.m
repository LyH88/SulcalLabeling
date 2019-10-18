%% setup environment
addpath(genpath('~/Tools/matlab/SurfLibrary'));

setenv('OMP_NUM_THREADS', '1');

% % init once
% [~,conda_path] = system('which conda');
% conda_path = strtrim(conda_path);
% setpref('condalab','base_path',conda_path);
% conda.init();
conda.setenv('parc_release');

label_lh = [0, 101, 103, 105, 107, 109, 113, 115, 117, 119, 121, 123, 125, 129, 133, 135, 137, 139, 141, 143, 145, 147, 149, ...
            151, 153, 155, 157, 161, 163, 165, 167, 169, 171, 173, 175, 177, 179, 181, 183, 185, 187, 191, 193, 195, 197, ...
            199, 201, 203, 205, 207];
label_rh = [0, 100, 102, 104, 106, 108, 112, 114, 116, 118, 120, 122, 124, 128, 132, 134, 136, 138, 140, 142, 144, 146, 148, ...
            150, 152, 154, 156, 160, 162, 164, 166, 168, 170, 172, 174, 176, 178, 180, 182, 184, 186, 190, 192, 194, 196, ...
            198, 200, 202, 204, 206];

%% subject
ico_level = 6;
hemi = 'lh';
subj = '/home/lyui/mount/fs4/ZALD_TTS/Zald_209556/209556/ZALD_TTS-x-Zald_209556-x-209556-x-surf_postproc_v1-x-12f315e3-ba2b-46c2-a2e4-05cebb285336/Surface/';
root = '/home/lyui/mount/nfs/Parcellation/ZALD_TTS';

if ico_level == 5
    model = ['/home/lyui/mount/nfs/Parcellation/ZALD_TTS/model/8k/' hemi '_ico5/checkpoint_latest.pth.tar_SUNet_021.pth.tar'];
elseif ico_level == 6
    model = ['/home/lyui/mount/nfs/Parcellation/ZALD_TTS/model/8k/' hemi '/checkpoint_latest.pth.tar_SUNet_045.pth.tar'];
end

if strcmpi(hemi, 'lh')
    label = label_lh;
else
    label = label_rh;
end
ico_level = num2str(ico_level);
[~, fname] = fileparts(fileparts(fileparts(subj)));
fname = strsplit(fname, '-x-');
fname = [fname{2} '_' fname{5}];
subj_sphere = [subj filesep 'lh.sphere.vtk'];
ico_sphere = ['/home/lyui/mount/nfs/Parcellation/icosphere_' ico_level '.vtk'];

%%
tmpdir = tempname;
[~, tmpsubj] = fileparts(tempname);
mkdir(tmpdir);

subj_sphere_deg = [tmpdir '/' tmpsubj '.sphere.vtk'];

%% create features
extract_features([subj '/lh.white.vtk'], [tmpdir '/' tmpsubj]);

% rigid alignment
cmd = ['HSD -s ' root '/template/sphere_327680.vtk,' subj_sphere ...
       ' --outputcoeff NaN,' tmpdir '/' tmpsubj '.coeff.txt' ' -p ' root '/template/' ...
       hemi '.oasis45.iH.mean.txt,' tmpdir '/' tmpsubj '.iH.txt' ' -d 0 --fixedSubjects 0 --icosahedron 4'];
system(cmd);
cmd = ['HSD -s ' root '/template/sphere_327680.vtk,' subj_sphere ...
       ' -c NaN,' tmpdir '/' tmpsubj '.coeff.txt', ...
       ' --outputcoeff NaN,' tmpdir '/' tmpsubj '.coeff.txt' ' -p ' root '/template/' ...
       hemi '.oasis45.curv.mean.txt,' tmpdir '/' tmpsubj '.H.txt' ' -d 0 --fixedSubjects 0 --icosahedron ' ico_level ' --noguess'];
system(cmd);

%% resampling
cmd = ['SurfRemesh -r ' ico_sphere ' -t ' subj_sphere ...
       ' -c ' tmpdir '/' tmpsubj '.coeff.txt' ' -p ' ...
       tmpdir '/' tmpsubj '.iH.txt,' tmpdir '/' tmpsubj '.sulc.txt,' tmpdir '/' tmpsubj '.H.txt' ...
       ' --outputProperty ' tmpdir '/' tmpsubj '.res --deg 0 --noheader --deform ' subj_sphere_deg];
system(cmd);

%% likelihood
cmd = ['/home/lyui/Tools/python/parc/test.sh ' model ' ' ico_level ' ' tmpdir ' ' tmpdir '/' tmpsubj '.mat'];
system( cmd );

%% refinement
load([tmpdir '/' tmpsubj '.mat']);
prob = squeeze(prob);

prob_list = {};
for j = 1: size(prob,1)
    pfile = sprintf('%s/parc.prob%d.txt',tmpdir,j);
    prob_list = [prob_list pfile];
    fp = fopen(pfile,'w');
    fprintf(fp, '%f\n', prob(j,:));
    fclose(fp);
end
plist = sprintf('%s,',prob_list{:});
plist = plist(1:end-1);

system(sprintf('SurfRemesh -p %s -r %s -t %s --noheader --outputProperty %s/parc',plist,subj_sphere_deg,ico_sphere,tmpdir));

prob_new = [];
for j = 1: size(prob,1)
    pfile = sprintf('%s/parc.prob%d.txt',tmpdir,j);
    prob_new = [prob_new; load(pfile)'];
end
system(['rm -f ' tmpdir '/parc.prob*.txt']);

[v, f] = read_vtk([subj '/' hemi '.white.vtk']);
pred_refine = gcut(v, f,prob_new);
pred_refine = label(pred_refine)';

truth = load([subj '/' hemi '.parc.refine.edited.txt']);
% truth = load([subj '/' hemi '.parc.txt']);

[~, pred] = max(prob_new);
pred = label(pred)';
write_property([root '/parc/' fname '.' hemi '.parc.vtk'], v, f, struct('truth', truth, 'pred', pred, 'pred_refine', pred_refine));

%% dice
% pred = load('/fs4/masi/parvatp/SoftwarePlugins/ugscnn/ugscnn/experiments/exp3_2d3ds/logs/parcel/correct/0325/Zald_SHUFFLE/pleaseWork_reg10curv_big_big_data_8k_feat32/outdir/Zald_209556/Zald_209556_parc_res.parc.txt');
pred = pred_refine;
dice = zeros(length(label),1);
for i = 1:length(label)
    dice(i) = numel(pred(pred(:,1) == label(i) & truth == label(i))) / ( numel(pred(pred(:,1) == label(i)))+ numel(truth(truth==label(i))))*2;
end
dice = dice(2:end);
mean(dice)

%%
system(['rm -rf ' tmpdir]);
disp('done');

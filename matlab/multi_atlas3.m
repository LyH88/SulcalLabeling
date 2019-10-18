%% setup environment
addpath(genpath('/home/lyui/Tools/matlab/SurfLibrary'));
addpath(genpath('gcut'));

setenv('OMP_NUM_THREADS', '1');

reg_postfix = '';
%cv_postfix = 'ma_lh';
cv_postfix = 'ma_ce';

ico_level = 5;
root = '/home/lyui/mount/nfs/Parcellation/Sulcus';
label = [0, 1, 2, 3, 4, 5, 6];
    
% subject
slist = dir('/home/lyui/mount/nfs/Sulcus/NORA_PFCSulci/n*');
subj_list = [];
for i = 1: length(slist)
    if slist(i).isdir
        subj_list = [subj_list; {[slist(i).folder filesep slist(i).name]}];
    end
end
%subj_list = strcat(repelem(subj_list, 1, 1), repmat({'.lh'},length(subj_list),1));
%subj_list = strcat(repelem(subj_list, 1, 1), repmat({'.rh'},length(subj_list),1));
subj_list = strcat(repelem(subj_list, 2, 1), repmat({'.lh';'.rh'},length(subj_list),1));

%% 
% one time processing
%fold = 1;

for fold = 1:5
    
    subj_list2 = circshift(subj_list, (fold - 1) * 12);
    subj_list2 = subj_list2(49:end);
    
    % ico5_64
    %model = ['/home/lyui/mount/nfs/Parcellation/Sulcus/model/cv/' num2str(fold) '/ico5_64_dice_rh_run2/checkpoint_latest.pth.tar_SUNet_best.pth.tar'];
    %model = ['/home/lyui/mount/nfs/Parcellation/Sulcus/model/cv/' num2str(fold) '/ico5_64_dice_lh/checkpoint_latest.pth.tar_SUNet_best.pth.tar'];
    model = ['/home/lyui/mount/nfs/Parcellation/Sulcus/model/cv/' num2str(fold) '/ico5_64_cross/checkpoint_latest.pth.tar_SUNet_best.pth.tar'];

    
    ico_level = num2str(ico_level);
    ico_sphere = ['/home/lyui/mount/nfs/Parcellation/icosphere_' ico_level '.vtk'];
    
    flist = dir('/home/lyui/mount/nfs/Sulcus/NORA_PFCSulci/deform/input_reflect/*reg15*label.txt');
    id = split({flist.name},'.');
    id = squeeze(id(:,:,1));
    id = sort(id);
    
    % read data
    parc = zeros(163842, length(flist));
    for i = 1: length(flist)
        parc(:,i) = load([flist(i).folder filesep flist(i).name]);
    end
    parc(parc>7) = 0;
    
    %% likelihood
    parfor i = 1: length(subj_list2)
        subj = dir([subj_list2{i}(1:end-3) '/curves/']);
        subj = subj(1).folder;
        
        [~, fname] = fileparts(fileparts(subj));
        [~, ~, hemi] = fileparts(subj_list2{i}); % changed subj_list to subj_list2
        hemi = hemi(2:end);
        if strcmp(hemi,'rh')
            subj_sphere = [subj filesep hemi '.sphere.reflect.vtk'];
        else
            subj_sphere = [subj filesep hemi '.sphere.vtk'];
        end
        
        %
        tmpdir = [root filesep 'validation' filesep 'ico' ico_level reg_postfix filesep fname];
        tmpsubj = hemi;
        
        leave_one_out = i;
        prob = histc(parc(:, setdiff(1:length(flist), leave_one_out)),label,2);
        prob = prob / (size(parc,2)-1);
        prob = prob';
        
        subj_sphere_deg = [tmpdir '/' tmpsubj '.sphere.vtk'];
        
        prob_list = {};
        for j = 1: size(prob,1)
            pfile = sprintf('%s/%s.parc.prob%d.txt',tmpdir,hemi,j);
            prob_list = [prob_list pfile];
            fp = fopen(pfile,'w');
            fprintf(fp, '%f\n', prob(j,:));
            fclose(fp);
        end
        plist = sprintf('%s,',prob_list{:});
        plist = plist(1:end-1);
        
        system(sprintf('/home/lyui/bin/SurfRemesh -p %s -r %s -t %s --noheader --outputProperty %s/%s.parc',plist,subj_sphere_deg,ico_sphere,tmpdir,tmpsubj));
        
        prob_new = [];
        for j = 1: size(prob,1)
            pfile = sprintf('%s/%s.parc.prob%d.txt',tmpdir,tmpsubj,j);
            prob_new = [prob_new; load(pfile)'];
        end
        system(['rm -f ' tmpdir '/' tmpsubj '.parc.prob*.txt']);
        fp = fopen([tmpdir '/' tmpsubj '.prob' cv_postfix '.txt'], 'w');
        fmt = [repmat('%f ', 1, size(prob_new,1)-1), '%f\n'];
        fprintf(fp, fmt, prob_new);
        fclose(fp);
    end
end
disp('done.');


%% graphcut
ico_level = '5';
parfor i=1:length(subj_list)
    subj = dir([subj_list{i}(1:end-3) '/curves/']);
    subj = subj(1).folder;

    [~, fname] = fileparts(fileparts(subj));
    [~, ~, hemi] = fileparts(subj_list{i});
    hemi = hemi(2:end);
    if strcmp(hemi,'rh')
        subj_sphere = [subj filesep hemi '.sphere.reflect.vtk'];
    else
        subj_sphere = [subj filesep hemi '.sphere.vtk'];
    end

    %
    tmpdir = [root filesep 'validation' filesep 'ico' ico_level reg_postfix filesep fname];
    tmpsubj = hemi;

    prob_new = load([tmpdir '/' tmpsubj '.prob' cv_postfix '.txt'])';

    [v, f] = read_vtk([subj '/' tmpsubj '.pial.vtk']);
    pred_refine = gcut(v, f,prob_new);
    pred_final = label(pred_refine)';

%     [M, I] = max(prob_new);
%     pred_final = label(I);

    fp = fopen([tmpdir '/' tmpsubj '.parc' cv_postfix '.txt'], 'w');
    fprintf(fp, '%d\n', pred_final);
    fclose(fp);
end

disp('DONE.');

%% dice
dice = NaN(length(subj_list), length(label));
parfor i = 1: length(subj_list)
    subj = dir([subj_list{i}(1:end-3) '/curves/']);
    subj = subj(1).folder;

    [~, fname] = fileparts(fileparts(subj));
    [~, ~, hemi] = fileparts(subj_list{i});
    hemi = hemi(2:end);
    
    tmpdir = [root filesep 'validation' filesep 'ico' ico_level reg_postfix filesep fname];
    tmpsubj = hemi;
    
    if ~exist([tmpdir '/' tmpsubj '.parc' cv_postfix '.txt'],'file'); continue; end
    pred_refine = load([tmpdir '/' tmpsubj '.parc' cv_postfix '.txt']);
    truth = load([subj_list{i}(1:end-3) '/curves/' hemi '.label.txt']);

    % dice
    pred = pred_refine;
    d = zeros(length(label),1);
    for j = 1:length(label)
        d(j) = numel(pred(pred(:,1) == label(j) & truth == label(j))) / ( numel(pred(pred(:,1) == label(j)))+ numel(truth(truth==label(j))))*2;
%         dice(j) = numel(pred(pred(:,1) == label(j) & truth == label(j))) / numel(truth(truth==label(j)));
    end
    dice(i, :) = d';
%     dice(i, end) = sum(pred == truth) / numel(truth);
%     [v, f] = read_vtk([subj '/' hemi '.white.vtk']);
%     write_property('/tmp/test.vtk',v,f,struct('pred', pred, 'truth', truth));
end

summary = nanmean(dice,2);
nanmean(summary)
nanstd(summary)

[mean(summary(1:12)) mean(summary(13:24)) mean(summary(25:36)) mean(summary(37:48)) mean(summary(49:60))]


%% visualization 

i = 17;  % i = 11 for n022t2.lh and i = 39 for n115t1.lh
subj = dir([subj_list{i}(1:end-3) '/curves/']);
subj = subj(1).folder;

[~, fname] = fileparts(fileparts(subj));
[~, ~, hemi] = fileparts(subj_list{i});
hemi = hemi(2:end);

tmpdir = [root filesep 'validation' filesep 'ico' ico_level reg_postfix filesep fname];
tmpsubj = hemi;

pred_refine = load([tmpdir '/' tmpsubj '.parc' cv_postfix '.txt']);
truth = load([subj_list{i}(1:end-3) '/curves/' hemi '.label.txt']);
truth(~ismember(truth, label)) = 0;

curv = load([subj_list{i}(1:end-3) '/curves/' hemi '.curv.txt']);
curv = -curv;
curv(curv > 1) =1;
curv(curv < -1) = -1;
curv = (curv + 1)/4;
pred_refine = double(pred_refine);
truth=double(truth);
pred_refine(pred_refine == 0) = curv(pred_refine == 0);
truth(truth == 0) = curv(truth == 0);

%[v, f] = read_vtk([subj '/' hemi '.white.vtk']);
[v, f] = read_vtk([subj '/' hemi '.inflated.vtk']);
write_property('/tmp/ma.vtk',v,f,struct('pred', pred_refine, 'truth', truth));
disp('done.');


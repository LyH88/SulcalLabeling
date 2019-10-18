addpath(genpath('~/Tools/matlab/SurfLibrary'));
surf_root = '/fs4/masi/lyui/ZALD_TTS/*/*/*/Surface';
root = '/home/lyui/mount/nfs/Parcellation/ZALD_TTS';
% surf_root = '/home/lyui/mount/fs4/Mindboggle/FreeSurfer/*/Registration';
% root = '/home/lyui/mount/nfs/Parcellation/Mindboggle';

%% labels
refine_label = false;
if ~exist([root filesep 'input'], 'dir')
    mkdir([root filesep 'input']);
end
if ~exist([root filesep 'input/labels'], 'dir')
    mkdir([root filesep 'input/labels']);
end
% flist = dir([surf_root filesep '*h.parc.txt']);
flist = dir([surf_root filesep '*h.label.DKT31_refined.txt']);

if refine_label
    parfor i = 1: length(flist)
        parc = load([flist(i).folder filesep flist(i).name]);
        fname = strsplit(fileparts(flist(i).folder),'-x-'); %xnat
        fname = [fname{2} '_' fname{5}];
        if contains(flist(i).name,'lh')
            parc(parc < 100 | mod(parc, 2) == 0) = 0;
            surf = [flist(i).folder filesep 'lh.white.vtk'];
            fname = [fname '.lh.parc.txt'];
        else
            parc(parc < 100 | mod(parc, 2) == 1) = 0;
            surf = [flist(i).folder filesep 'rh.white.vtk'];
            fname = [fname '.rh.parc.txt'];
        end
        fname = [root filesep 'input/labels' filesep fname];
        if ~exist(fname, 'file')
            parc = roi_correction(surf, parc);
            fp = fopen(fname, 'w');
            fprintf(fp, '%d\n', parc);
            fclose(fp);
        end
    end
else
    for i = 1: length(flist)
        [~, fname] = fileparts(fileparts(flist(i).folder)); %mindboggle
        if contains(flist(i).name,'lh')
            hemi = 'lh';
        else
            hemi = 'rh';
        end
        copyfile([flist(i).folder filesep flist(i).name], [root filesep 'input/labels/' fname '.' hemi '.parc.txt']);
    end
end

%% features
if ~exist([root filesep 'input'], 'dir')
    mkdir([root filesep 'input']);
end
if ~exist([root filesep 'input/features'], 'dir')
    mkdir([root filesep 'input/features']);
end
slist = dir([surf_root filesep '*h.white.vtk']);

parfor i = 1: length(slist)
    [~, fname] = fileparts(fileparts(slist(i).folder));
%     fname = strsplit(fileparts(slist(i).folder),'-x-'); %xnat
%     fname = [fname{2} '_' fname{5}];
    if contains(slist(i).name,'lh')
        hemi = 'lh';
    else
        hemi = 'rh';
    end
    extract_features([slist(i).folder filesep slist(i).name],[root filesep 'input/features/' fname '.' hemi]);
end

% [v,f]=read_vtk([slist(i).folder filesep slist(i).name]);
% write_property('/tmp/test.vtk',v,f,struct('iH',load('Mindboggle/input/features/Afterthought-1.lh.iH.txt'),'sulc',load('Mindboggle/input/features/Afterthought-1.lh.sulc.txt'),'H',load('Mindboggle/input/features/Afterthought-1.lh.H.txt')));
%% distance map
if ~exist([root filesep 'dist'], 'dir')
    mkdir([root filesep 'dist']);
end

slist = dir([surf_root filesep '*h.white.vtk']);
flist = dir([root filesep 'input/labels/*h.parc.txt']);
parfor i = 1: length(flist)
    fname = strrep(flist(i).name, '.parc', '.dist');
    fname = [root filesep 'dist' filesep fname];
    surf = strsplit(flist(i).name, '.');
%     surf = strsplit(surf{1}, '_');    %xnat
%     surf = {slist(contains({slist.folder}, surf{2})).folder}; %xnat
    surf = {slist(contains({slist.folder}, surf{1})).folder}; %mindboggle
    surf = surf{1};
    if contains(flist(i).name,'lh')
        surf = [surf filesep 'lh.white.vtk'];
    else
        surf = [surf filesep 'rh.white.vtk'];
    end
    if ~exist(fname, 'file')
        dist_map(surf, [flist(i).folder filesep flist(i).name], fname);
    end
end
%%
% distance map (template)
% dist_map([root '/template/lh.oasis45.white.vtk'], [root '/template/lh.parc.mode.txt'], [root '/template/lh.dist.txt']); %xnat
% dist_map([root '/template/rh.oasis45.white.vtk'], [root '/template/rh.parc.mode.txt'], [root '/template/rh.dist.txt']); %xnat
dist_map([root '/template/lh.mb101.white.vtk'], [root '/template/lh.DKT31_refined.mode.txt'], [root '/template/lh.dist.txt']);
dist_map([root '/template/rh.mb101.white.vtk'], [root '/template/rh.DKT31_refined.mode.txt'], [root '/template/rh.dist.txt']);

%% surface registration
% run sh script on cluster
if ~exist([root filesep 'coeff'], 'dir')
    mkdir([root filesep 'coeff']);
end

slist = dir([surf_root filesep '/*h.sphere.vtk']);
flist = dir([root filesep 'dist/*h.dist.txt']);
fp = fopen([root filesep 'run_reg.sh'], 'w');
fprintf(fp, 'export OMP_NUM_THREADS=1\n');
for i = 1: length(flist)
    fname = strrep(flist(i).name, '.dist', '.coeff');
    fname = [root filesep 'coeff' filesep fname];
    surf = strsplit(flist(i).name, '.');
%     surf = strsplit(surf{1}, '_');  %xnat
%     surf = {slist(contains({slist.folder}, surf{2})).folder};   %xnxat
    surf = {slist(contains({slist.folder}, surf{1})).folder}; %mindboggle
    surf = surf{1};
    hemi = 'lh';
    if contains(flist(i).name,'lh')
        surf = [surf filesep 'lh.sphere.vtk'];
        hemi = 'lh';
    else
        surf = [surf filesep 'rh.sphere.vtk'];
        hemi = 'rh';
    end
    fprintf(fp, ...
            ['HSD -s ' root '/template/sphere_327680.vtk,' surf ...
            ' --outputcoeff NaN,' fname ' -p ' root '/template/' ...
            hemi '.dist.txt,' flist(i).folder filesep flist(i).name ' -d 10 --fixedSubjects 0 --idprior 200 &\n']);
    if mod(i, 19) == 0
        fprintf(fp, 'wait\n');
    end
end
fclose(fp);
system(['chmod 755 ' root filesep 'run_reg.sh']);

%% data augmentation
% run sh script on cluster
prop = 'curv';
if ~exist([root filesep 'aug/' prop '/'], 'dir')
    mkdir([root filesep 'aug/' prop '/']);
end
if ~exist([root filesep 'aug/' prop '/features'], 'dir')
    mkdir([root filesep 'aug/' prop '/features']);
end
if ~exist([root filesep 'aug/' prop '/labels'], 'dir')
    mkdir([root filesep 'aug/' prop '/labels']);
end

slist = dir([surf_root filesep '*h.sphere.vtk']);
flist = dir([root filesep 'coeff/' prop '/*h.coeff.txt']);

fp0 = fopen([root filesep 'run_aug_label.sh'], 'w');
fprintf(fp0, 'export OMP_NUM_THREADS=1\n');
fp = fopen([root filesep 'run_aug.sh'], 'w');
fprintf(fp, 'export OMP_NUM_THREADS=1\n');
for deg = 0: 10
    for i = 1: length(flist)
        [~, fname] = fileparts(fileparts(slist(i).folder));
%         fname = strsplit(fileparts(slist(i).folder),'-x-'); %xnat
%         fname = [fname{2} '_' fname{5}];%xnat

        surf = strsplit(flist(i).name, '.');
%         surf = strsplit(surf{1}, '_');  %xnat
%         surf = {slist(contains({slist.folder}, surf{2})).folder};   %xnxat
        surf = {slist(contains({slist.folder}, surf{1})).folder}; %mindboggle
        surf = surf{1};
        if contains(flist(i).name,'lh')
            iH = [root filesep 'input/features/' fname '.lh.iH.txt'];
            sulc = [root filesep 'input/features/' fname '.lh.sulc.txt'];
            curv = [root filesep 'input/features/' fname '.lh.H.txt'];
            label = [root filesep 'input/labels/' fname '.lh.parc.txt'];
            surf = [surf filesep 'lh.sphere.vtk'];
            hemi = 'lh';
        else
            iH = [root filesep 'input/features/' fname '.rh.iH.txt'];
            sulc = [root filesep 'input/features/' fname '.rh.sulc.txt'];
            curv = [root filesep 'input/features/' fname '.rh.H.txt'];
            label = [root filesep 'input/labels/' fname '.rh.parc.txt'];
            surf = [surf filesep 'rh.sphere.vtk'];
            hemi = 'rh';
        end
        fprintf(fp0, ['SurfRemesh -r /home/lyui/mount/nfs/Parcellation/icosphere_7.vtk -t ' surf ...
                ' -c ' flist(i).folder filesep flist(i).name ' -p ' label ...
                ' --outputProperty ' root '/aug/' prop '/labels/' fname '.' hemi '.deg' num2str(deg) ' --deg ' num2str(deg) ' --noheader --nneighbor &\n']);
        fprintf(fp, ['SurfRemesh -r /home/lyui/mount/nfs/Parcellation/icosphere_7.vtk -t ' surf ...
                ' -c ' flist(i).folder filesep flist(i).name ' -p ' iH ',' sulc ',' curv ...
                ' --outputProperty ' root '/aug/' prop '/features/' fname '.' hemi '.deg' num2str(deg) ' --deg ' num2str(deg) ' --noheader &\n']);
        if mod(i, 16) == 0
            fprintf(fp0, 'wait\n');
            fprintf(fp, 'wait\n');
        end
    end
end
fclose(fp0);
fclose(fp);
system(['chmod 755 ' root filesep 'run_aug_label.sh']);
system(['chmod 755 ' root filesep 'run_aug.sh']);
% d1=load('aug/108768_da795406-a26e-432e-bc81-f80affd08296.lh.deg0.dist.txt');
% d2=load('template/lh.dist.txt');
% [v,f]=read_vtk('template/sphere_327680.vtk');
% write_property('/tmp/test.vtk',v,f,struct('d1',d1,'d2',d2));

%% make them into binary
prop = 'curv';
flist1 = dir([root filesep 'aug/' prop '/features/*.txt']);
% flist2 = dir([root filesep 'aug/' prop '/labels/*.txt']);

parfor i = 1: length(flist1)
    t = load([flist1(i).folder filesep flist1(i).name]);
    fp=fopen([strrep(flist1(i).folder, 'aug', 'aug_bin') filesep strrep(flist1(i).name, '.txt', '.dat')],'wb');
    fwrite(fp, t, 'double');
    fclose(fp);
end
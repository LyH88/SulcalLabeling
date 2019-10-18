function extract_features(surf_model, hemi_base)
    tmpfolder = tempname;
    mkdir(tmpfolder);
    [~, hemi] = fileparts(surf_model);
    hemi = strsplit(hemi, '.');
    hemi = hemi{1};
    cmd = ['mris_inflate ' surf_model ' ' tmpfolder filesep hemi '.inflated'];
    system(cmd);
    cmd = ['MeshProperty -i ' surf_model ' -o ' hemi_base ' -m --surfsmoothing 5 --tensorsmoothing 5'];
    system(cmd);
    cmd = ['mris_curvature -thresh .999 -n -a 5 -w -distances 10 10 ' tmpfolder filesep hemi '.inflated'];
    system(cmd);
    
    iH = read_curv([tmpfolder filesep hemi '.inflated.H']);
    sulc = read_curv([tmpfolder filesep hemi '.sulc']);
    H = load([hemi_base '.H.txt']);
    
    iH = zscore(iH);
    iH(iH < -3) = -3 - (1 - exp(3 + iH(iH < -3)));
    iH(iH > 3) = 3 + (1 - exp(3 - iH(iH > 3)));
    iH = iH / std(iH);
    iH(iH < -3) = -3 - (1 - exp(3 + iH(iH < -3)));
    iH(iH > 3) = 3 + (1 - exp(3 - iH(iH > 3)));
    
    sulc = zscore(sulc);
    sulc(sulc < -3) = -3 - (1 - exp(3 + sulc(sulc < -3)));
    sulc(sulc > 3) = 3 + (1 - exp(3 - sulc(sulc > 3)));
    sulc = sulc / std(sulc);
    sulc(sulc < -3) = -3 - (1 - exp(3 + sulc(sulc < -3)));
    sulc(sulc > 3) = 3 + (1 - exp(3 - sulc(sulc > 3)));
    
    H = -H;
    H = zscore(H);
    H(H < -3) = -3 - (1 - exp(3 + H(H < -3)));
    H(H > 3) = 3 + (1 - exp(3 - H(H > 3)));
    H = H / std(H);
    H(H < -3) = -3 - (1 - exp(3 + H(H < -3)));
    H(H > 3) = 3 + (1 - exp(3 - H(H > 3)));
    
    fp = fopen([hemi_base '.H.txt'], 'w');
    fprintf(fp, '%f\n', H);
    fclose(fp);
    fp = fopen([hemi_base '.iH.txt'], 'w');
    fprintf(fp, '%f\n', iH);
    fclose(fp);
    fp = fopen([hemi_base '.sulc.txt'], 'w');
    fprintf(fp, '%f\n', sulc);
    fclose(fp);
    system(['rm -rf ' tmpfolder]);
end
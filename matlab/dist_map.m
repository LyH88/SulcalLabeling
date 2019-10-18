function dist_map(surf,parc,outdist)
    g = load(parc);
    
	[v,f]=read_vtk(surf);
	nnv = connectivity(v,f + 1);

    for iter = 1: 1
        g1 = g;
        for i = 1: size(v,1)
            if numel(unique(g(nnv{i}))) > 1
                g1(nnv{i}) = 0;
            end
        end
        g = g1;
    end

    boundary = tempname;
	fp = fopen(sprintf('%s',boundary), 'w');
	fprintf(fp, '%d\n', find(g == 0)-1);
    fprintf(fp, '\n');
	fclose(fp);
	
    tmp = tempname;
    system(sprintf('SulcalMap -i %s -s %s -o %s -r 100',surf,boundary,tmp));
    
    % normalization
    g = load(parc);
    dist = load(sprintf('%s.absDist.txt',tmp));
    dist = [dist [1:size(v,1)]'];
    for i = unique(g)'
        d = dist(g == i,:);
        val = max(d(:,1));
        if val > 0
            dist(g==i, 1) = dist(g==i, 1) / val;
        end
    end
    
    dist(g==0,1)=0;
    dist(isnan(dist(g==0,1)),1)=0;
    fp = fopen(outdist, 'w');
    fprintf(fp, '%f\n', dist(:,1)');
    fclose(fp);
    
    delete(sprintf('%s',boundary))
    delete(sprintf('%s.absDist.txt',tmp))
end

function nnv = connectivity(v,f)
    nnv = cell(size(v, 1), 1);
    for i = 1: size(f, 1)
        nnv{f(i,1)} = [nnv{f(i,1)}, f(i, [2 3])];
        nnv{f(i,2)} = [nnv{f(i,2)}, f(i, [1 3])];
        nnv{f(i,3)} = [nnv{f(i,3)}, f(i, [1 2])];
    end
    for i = 1: size(v, 1)
        nnv{i} = unique(nnv{i});
    end
end

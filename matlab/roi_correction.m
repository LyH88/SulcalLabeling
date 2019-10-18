function parc = roi_correction(mesh, label)
    parc=label;

    [~,f]=read_vtk(mesh);

    F=f+1;
    n = max(F(:));

    % remove duplicated edges
    rows = [F(:,1); F(:,1); F(:,2); F(:,2); F(:,3); F(:,3)];
    cols = [F(:,2); F(:,3); F(:,1); F(:,3); F(:,1); F(:,2)];
    rc = unique([rows,cols], 'rows','first');

    % fill adjacency matrix
    A = sparse(rc(:,1),rc(:,2),1,n,n);
    G = graph(A);

    bsize_pre = 0;
    bsize = 17;
    iter = 1;

    while bsize_pre ~= bsize
%         fprintf('Iteration#%d\n', iter);
        parc_refine = parc;
        bsize_pre = bsize;

        e1 = G.Edges.EndNodes(:,1);
        e2 = G.Edges.EndNodes(:,2);
        cut = parc_refine(e1,:)~=parc_refine(e2,:);
        e1(cut) = [];
        e2(cut) = [];
        A2 = sparse([e1;e2],[e2;e1],1,n,n);
        G2= graph(A2);
        bins = conncomp(G2);
        bsize = length(unique(bins));
%         fprintf(' bin#: %d\n', bsize);

        for bin = 1: bsize
            nneighbor = find(bins==bin);
            nlabel = unique(parc(nneighbor));
            for label = 1: length(nneighbor)
                nlabel = [nlabel; parc(neighbors(G,nneighbor(label)))];
            end
            nlabel = unique(nlabel);
            if length(nlabel) == 2
                old_label = unique(parc(nneighbor));
                new_label = setdiff(nlabel,old_label);
                if new_label == 0 && length(nneighbor) > 50
                    continue;
                end
%                 fprintf('%d (n=%d): %d -> %d\n',bin,length(nneighbor),old_label,new_label);
                parc_refine(nneighbor) = new_label;
            elseif length(nneighbor) < 20
                old_label = unique(parc(nneighbor));
                new_label = mode(nlabel);
                if old_label == new_label
                    continue;
                end
%                 fprintf('%d (n=%d): %d -> %d\n',bin,length(nneighbor),old_label,new_label);
                parc_refine(nneighbor) = new_label;
            end
        end
        parc = parc_refine;
        iter = iter + 1;
    end
end
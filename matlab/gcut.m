function parc = gcut(v, f, prob)
    %% graph
    F=f+1;
    n = max(F(:));
    % remove duplicated edges
    if size(F,2) == 2
      rows = [F(:,1); F(:,2)];
      cols = [F(:,2); F(:,1)];
    else
      rows = [F(:,1); F(:,1); F(:,2); F(:,2); F(:,3); F(:,3)];
      cols = [F(:,2); F(:,3); F(:,1); F(:,3); F(:,1); F(:,2)];
    end
    rc = unique([rows,cols], 'rows','first');

    % fill adjacency matrix
    G = graph(sparse(rc(:,1),rc(:,2),1,n,n));
    parc = gcut_(v, G, prob);
end
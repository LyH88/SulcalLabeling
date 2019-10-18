function parc = gcut_(v, G, prob)
    n = size(G.Nodes,1);
    nlabel = size(prob,1);
    SmoothnessCost = ones(nlabel) - eye(nlabel);

    % data cost
    % softmax and -log
    % prob = (1 + exp(-prob)) .^ -1);
    % prob = log(abs(prob)) .* sign(prob);
    prob = exp(prob - repmat(max(prob), nlabel, 1));
    prob = prob ./ repmat(sum(prob), nlabel, 1);

    %prob = -log(prob+1e-5);
    prob = -log(prob);
    DataCost = prob;

    % len = (prior(G.Edges.EndNodes(:,1),:)+prior(G.Edges.EndNodes(:,2),:))/2;
    % try to maximize the cut - we want to cut through bigger triangles 
    len = v(G.Edges.EndNodes(:,1),:)-v(G.Edges.EndNodes(:,2),:);
    len = sqrt(sum(len .* len,2));
    len = exp(-len);
    %len = 1./(len+1e-1);

    % sparse cost
    SparseSmoothness = sparse([G.Edges.EndNodes(:,1);G.Edges.EndNodes(:,2)],[G.Edges.EndNodes(:,2);G.Edges.EndNodes(:,1)],[len;len],n,n);
    
    % weak
    % SparseSmoothness = SparseSmoothness^2;
    
    % medium
    % SparseSmoothness = spfun(@exp,SparseSmoothness);
    
    % strong
    % SparseSmoothness = SparseSmoothness^3;
    
    [~,output]= min(prob);
    
    % graphcut
    [gch] = GraphCut('open', DataCost, SmoothnessCost, 5*SparseSmoothness); % 50
    [gch] = GraphCut('set', gch, output-1);
    [gch, L] = GraphCut('expand',gch);
    % [gch, L] = GraphCut('swap',gch);
    GraphCut('close', gch);
    
    % label
    parc = L'+1;
end
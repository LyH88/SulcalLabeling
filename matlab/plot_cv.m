function plot_cv()
    hemi = 'lh';
    test_model1 = 'cv';
    test_model2 = 'naive_200';
    test_model3 = 'naive';
    epoch = 100;
    scale = 1;
    
    loss_test = zeros(epoch,5);
    loss_val = zeros(epoch,5);
    loss_train = zeros(epoch,5);
    naive = true;

    for fold = 1: 5
        
        if (fold == 2 || fold == 5 || fold == 1 || fold == 3) && (naive == false)
            %fn = ['/home/lyui/mount/nfs/Parcellation/Sulcus/model/' test_model1 '/' num2str(fold) filesep 'ico5_64_cross/train.log'];
        else
            % model 1
            %fn = ['/home/lyui/mount/nfs/Parcellation/Sulcus/model/' test_model1 '/' num2str(fold) filesep 'ico5_64/train.log'];
            %fn = ['/home/lyui/mount/nfs/Parcellation/Sulcus/model/' test_model1 '/' num2str(fold) filesep 'ico5_64_dice/train.log']
            %fn = ['/home/lyui/mount/nfs/Parcellation/Sulcus/model/' test_model1 '/' num2str(fold) filesep 'ico5_64_dice_run2/train.log'];
            %fn = ['/home/lyui/mount/nfs/Parcellation/Sulcus/model/' test_model1 '/' num2str(fold) filesep 'ico5_64_dice_lh/train.log'];
            %fn = ['/home/lyui/mount/nfs/Parcellation/Sulcus/model/' test_model1 '/' num2str(fold) filesep 'ico5_64_dice_lh_run2/train.log'];
            %fn = ['/home/lyui/mount/nfs/Parcellation/Sulcus/model/' test_model1 '/' num2str(fold) filesep 'ico5_64_dice_rh_run2/train.log'];
            
            % model 2
            %fn = ['/home/lyui/mount/nfs/Parcellation/Sulcus/model/' test_model2 '/' num2str(fold) filesep 'ico5_64/train.log'];
            %fn = ['/home/lyui/mount/nfs/Parcellation/Sulcus/model/' test_model2 '/' num2str(fold) filesep 'ico5_64_lh/train.log'];
            %fn = ['/home/lyui/mount/nfs/Parcellation/Sulcus/model/' test_model2 '/' num2str(fold) filesep 'ico5_64_dice_rh_run2/train.log'];
            
            % model 3
            fn = ['/home/lyui/mount/nfs/Parcellation/Sulcus/model/' test_model3 '/' num2str(fold) filesep 'ico5_64/train.log'];
            
            % model 1, feature 32
            %fn = ['/home/lyui/mount/nfs/Parcellation/Sulcus/model/' test_model3 '/' num2str(fold) filesep 'ico5_32_dice/train.log'];
        end
        
        fid = fopen(fn,'r');
        data = textscan(fid,'%s','delimiter','\n');
        fclose(fid);
        data = data{:};
        
        query_m = 'Avg loss: ';
        
        query = 'train stats';
        loss_train(:, fold) = getLoss(data, query, query_m);
        query = 'val stats';
        loss_val(:, fold) = getLoss(data, query, query_m);
        query = 'test stats';
        loss_test(:, fold) = getLoss(data, query, query_m);
    end
    
    loss_train_m = mean(loss_train,2);
    loss_train_s = std(loss_train')';
    loss_train_m1 = loss_train_m + loss_train_s*scale;
    loss_train_m2 = loss_train_m - loss_train_s*scale;
    
    loss_val_m = mean(loss_val,2);
    loss_val_s = std(loss_val')';
    loss_val_m1 = loss_val_m + loss_val_s*scale;
    loss_val_m2 = loss_val_m - loss_val_s*scale;
    
    loss_test_m = mean(loss_test,2);
    loss_test_s = std(loss_test')';
    loss_test_m1 = loss_test_m + loss_test_s*scale;
    loss_test_m2 = loss_test_m - loss_test_s*scale;
    
    [mean(loss_train_s) mean(loss_val_s) mean(loss_test_s)]

    figure;
    hold on;
    h = plot(1:epoch,[loss_train_m loss_val_m loss_test_m]','LineWidth',2);
    c = get(h,'Color');
    plotshaded(1:epoch,[loss_train_m1 loss_train_m2]',c{1});
    plotshaded(1:epoch,[loss_val_m1 loss_val_m2]',c{2});
    plotshaded(1:epoch,[loss_test_m1 loss_test_m2]',c{3});
    ylim([0 1]);
    yticks([0 0.2 0.4 0.6 0.8 1]);
    if naive == true
        xlim([0 80]);
        xticks(0:20:80);
    else
        xlim([0 25]);
        xticks(0:5:25);
    end
    LG = legend('Train','Validation','Test');
    LG.FontSize = 16;
    l = xlabel('Epoch'); 
    g = ylabel('Loss');
    set(l, 'FontSize', 16) 
    set(g, 'FontSize', 16) 
    xl = get(gca,'XAxis');
    set(xl, 'FontSize', 15);
    yl = get(gca,'YAxis');
    set(yl, 'FontSize', 15);
    hold off;
end

function loss = getLoss(data, query, query_m)
    data = data(contains(data, query));
    loss = zeros(length(data),1);
    id = strfind(data, query_m);
    id = [id{:}];

    for i = 1: length(data)
        loss(i) = str2double(data{i}(length(query_m) + id(i):length(query_m) + id(i)+5));
    end
end

function varargout = plotshaded(x,y,fstr)
    % http://jvoigts.scripts.mit.edu/blog/nice-shaded-plots/
    % x: x coordinates
    % y: either just one y vector, or 2xN or 3xN matrix of y-data
    % fstr: format ('r' or 'b--' etc)
    %
    % example
    % x=[-10:.1:10];plotshaded(x,[sin(x.*1.1)+1;sin(x*.9)-1],'r');

    if size(y,1)>size(y,2)
        y=y';
    end

    if size(y,1)==1 % just plot one line
        plot(x,y,fstr);
    end

    if size(y,1)==2 %plot shaded area
        px=[x,fliplr(x)]; % make closed patch
        py=[y(1,:), fliplr(y(2,:))];
        patch(px,py,1,'FaceColor',fstr,'EdgeColor','none');
    end

    if size(y,1)==3 % also draw mean
        px=[x,fliplr(x)];
        py=[y(1,:), fliplr(y(3,:))];
        patch(px,py,1,'FaceColor',fstr,'EdgeColor','none');
        plot(x,y(2,:),fstr);
    end

    alpha(.2); % make patch transparent
end

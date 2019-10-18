
addpath(' /home/lyui/mount/nfs/Sulcus/NORA_PFCSulci/gcut');
vis_option = 1; % 1: dice, 2: dist
nclass = 3; % # of baseline

root = '/home/lyui/mount/nfs/Sulcus/NORA_PFCSulci/parc/test_mat/dice';


full_measure1 = ma_exp(:,2:end)'; % MA dice
full_measure2 = baseline_g5(:,2:end)'; % Baseline dice
full_measure3 = ours_final(:,2:end)'; % CV dice

labels = readtable('~/../lyui/mount/nfs/Sulcus/NORA_PFCSulci/label_valid_list.csv');

% t-test
if vis_option == 1
    [~,p]=ttest(full_measure1',full_measure3');
    sr1 = fdr_bh(p);
    labels.name(sr1)

    [~,p]=ttest(full_measure2',full_measure3');
    sr2 = fdr_bh(p);
    labels.name(sr2)
    
    [mean(full_measure1(:)) mean(full_measure2(:)) mean(full_measure3(:))]
    [std(mean(full_measure1(:),2)) std(mean(full_measure2(:),2)) std(mean(full_measure3(:),2))]
end

cc = lines(nclass);
% https://stackoverflow.com/questions/15971478/most-efficient-way-of-drawing-grouped-boxplot-matlab
% boxplot( full_measure', 'Labels', labels.name', 'Notch','off', 'Colors', cc );
labels.name = strrep(labels.name,'_','\_');
figure('Position', [10 10 1200 300]);

%% bar
b= bar( [nanmean(full_measure1,2), nanmean(full_measure2,2), nanmean(full_measure3,2)],'FaceColor','flat' ); xticks(1:6); 
xticklabels(labels.name);
for k = 1:nclass
    b(k).CData = repmat(cc(k,:),6,1);
end

% if vis_option == 2
%     figure('Position', [10 10 1200 400]);
%     b= bar( [sum(isnan(full_measure1),2), sum(isnan(full_measure2),2), sum(isnan(full_measure3),2)],'FaceColor','flat' ); xticks(1:19); xticklabels(labels.name);
%     for k = 1:nclass
%         b(k).CData = repmat(cc(k,:),19,1);
%     end
%     ylabel('No labels');
%     ylim([0 100]);
%     set(gca,'YScale','log');
% end

avg = [nanmean(full_measure1,2) nanmean(full_measure2,2) nanmean(full_measure3,2)];
stdev = [nanstd(full_measure1')' nanstd(full_measure2')' nanstd(full_measure3')'];
nonan = [sum(~isnan(full_measure1),2) sum(~isnan(full_measure2),2) sum(~isnan(full_measure3),2)];
nonan(nonan == 0) = 1;
stdev = stdev ./ sqrt(nonan);
hold on;
for ib = 1:numel(b)
    %XData property is the tick labels/group centers; XOffset is the offset
    %of each distinct group
    xData = b(ib).XData+b(ib).XOffset;
    errorbar(xData,avg(:,ib),stdev(:,ib),'k.','HandleVisibility','off')
    if vis_option == 1 && ib == 3
        stars = repmat({''},size(full_measure3,1),1);
        stars(sr1 & sr2) = {'\color[rgb]{0 0 0}*'};
        stars(sr1 & ~sr2) = {sprintf('\\color[rgb]{%f %f %f}*',cc(1,:))};
        stars(~sr1 & sr2) = {sprintf('\\color[rgb]{%f %f %f}*',cc(2,:))};
        text(xData, avg(:,ib)+stdev(:,ib),stars,'Point X', ...
     'HorizontalAlignment', 'center', ...
     'VerticalAlignment', 'bottom', 'FontSize',15)
    end
end
hold off;

% %% box
% group = [repmat(1:3:57,1,size(full_measure1,2)) repmat(2:3:57,1,size(full_measure2,2)) repmat(3:3:57,1,size(full_measure3,2))];
% positions = repelem(1:19,1,nclass);
% positions(2:nclass:end) = positions(2:nclass:end) + .25;
% positions(3:nclass:end) = positions(3:nclass:end) + .5;
% 
% boxplot( [full_measure1(:)' full_measure2(:)' full_measure3(:)'], group,'Notch','off' ,'positions',positions);
% set(gca,'xtick', (positions(1:nclass:end)+positions(2:nclass:end)+positions(3:nclass:end))/nclass);
% set(gca,'xticklabel',labels.name)
% 
% h = findobj(gca,'tag','Median');
% for j=1:length(h)
% %     set(h(j),'linestyle','-.','Color',cc(3-j,:),{'linew'},{2});
%     set(h(j),'linestyle','-.','Color',[0 0 0],{'linew'},{2});
% end
% 
% set(findobj(gca,'tag','Outliers'),'MarkerEdgeColor',[0 0 0]);
% h = findobj(gca,'Tag','Box');
% for j=1:length(h)
% %     if mod(j, 2) == 1; c = [0.25,0.25,0.25]; end
% %     if mod(j, 2) == 0; c = [1,1,1]; end
%     c = cc(nclass-mod(j,nclass),:);
%     patch(get(h(j),'XData'),get(h(j),'YData'), c,'FaceAlpha',.7,'EdgeColor',c);
% end

%% label
%xtickangle(45)
l = xlabel('ROIs');
set(l, 'FontSize', 15) 
set(gca,'TickLength',[0 0])
xl = get(gca,'XAxis');
set(xl, 'FontSize', 15);
yl = get(gca,'YAxis');
set(yl, 'FontSize', 15);
if vis_option == 1
    g = ylabel('Dice');
    set(g, 'FontSize', 15)
    ylim([0.4 1]);
    yticks(0.4:0.2:1);
else
    ylabel('Surface Distance (mm)');
    ylim([0 12]);
end
ax = gca;
ax.YGrid = 'on';
ax.GridLineStyle = '-';
box on;
lg = legend('Multi-atlas','Standard Spherical U-Net','Ours','Location','northoutside','Orientation','horizontal');
lg.FontSize = 16;

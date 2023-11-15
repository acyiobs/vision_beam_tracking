load('test_acc.mat');

bar([test_acc.top1(1), test_acc.top5(1),test_acc.top3(1), test_acc.top5(3)], 'FaceColor', "#D95319",'BarWidth',0.5)
grid on;
row1 = {'Future beam 1' 'Future beam 1' 'Future beam 3' 'Future beam 3'};
row2 = {'Top1' 'Top5' 'Top1' 'Top5'};
labelArray = [row1; row2];
tickLabels = strtrim(sprintf('%s\\newline        %s\n', labelArray{:}));

set(gca, 'xticklabel', tickLabels);

l = legend('Vision Input');
l.Location = 'northwest';
ylabel('Top-k Accuracy');
ylim([0.5,1]);

set(gcf, 'Position', [100, 100, 800, 300]);
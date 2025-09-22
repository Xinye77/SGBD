function  performance = evaluateModel(results)
performance.accuracy = sum(results.predictedLabel ==results.label)/results.testnum;
[~,~,~,performance.auc] = perfcurve(results.label,results.distance,true);
[~, ~, ~, performance.auprc] = perfcurve(results.label,results.distance, true, 'xCrit', 'reca', 'yCrit', 'prec');
fprintf('AUC = %.4f\n',performance.auc);
fprintf('AUPRC = %.4f\n',performance.auprc);


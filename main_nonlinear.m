clear all
clc
addpath('...\data')
datasets = {
    '40_vowels.mat','47_yeast.mat','20_letter.mat','12_fault.mat', '17_InternetAds.mat',...
    '6_cardio.mat','7_Cardiotocography.mat','25_musk.mat', '41_Waveform.mat','36_speech.mat',...
    '38_thyroid.mat', '35_SpamBase.mat','44_Wilt.mat', '27_PageBlocks.mat','31_satimage-2.mat',...
    '30_satellite.mat','26_optdigits.mat','19_landsat.mat','28_pendigits.mat','2_annthyroid.mat',...
    '24_mnist.mat','23_mammography.mat','22_magic.gamma.mat','5_campaign.mat','32_shuttle.mat',...
    '1_ALOI.mat','34_smtp.mat','3_backdoor.mat','8_celeba.mat','33_skin.mat',...
    '13_fraud.mat','10_cover.mat','9_census.mat','16_http.mat', '11_donors.mat'
    };

datanum = length(datasets);
AUCmean_results = zeros(datanum,1);
AUPRCmean_results = zeros(datanum,1);
AUCstd_results = zeros(datanum,1);
AUPRCstd_results = zeros(datanum,1);
time_results = zeros(datanum,1);

for ii = 1:length(datasets)
    dataset_name = datasets{ii};
    load(dataset_name);
    X = double(X);
    y = double(y');
    
    normal_data = X(y==0,:);
    outlier_data = X(y==1,:);
    normal_label = y(y==0);
    outlier_label = y(y==1);
    
    rng
    
    [normal_num,normal_dim]=size(normal_data);
    cv = cvpartition(normal_label,'HoldOut',0.3);
    test_normalidx = cv.test;
    test_nordata = normal_data(test_normalidx,:);
    test_norlabel = normal_label(test_normalidx);
    train_idx = cv.training;
    train_data = normal_data(train_idx,:);
    test_data = [outlier_data;test_nordata];
    test_label = [outlier_label;test_norlabel];
    if strcmp(dataset_name,'11_donors.mat')
        train_num = size(train_data,1);
        train_data_norm = train_data(randsample(train_num,ceil(train_num/5)),:);
        test_num = size(test_data,1);
        test_data_norm = train_data(randsample(test_num,ceil(test_num/5)),:);
        test_label = test_label(randsample(test_num,ceil(test_num/5)),1);
    else
        mu = mean(train_data);
        sigma = std(train_data);
        sigma(sigma==0)=1;
        train_data_norm = normalize(train_data,'zscore');
        train_data_norm(isnan(train_data_norm))=0;
        test_data_norm = (test_data-mu)./sigma;
        test_data_norm(isnan(test_data_norm))=0;
    end
    if  (100000 > normal_num) && (normal_num > 10000)
        min_gb = ceil(size(train_data,1)/300);
    elseif normal_num > 100000
        min_gb = ceil(size(train_data,1)/3000);
    else
        min_gb = ceil(size(train_data,1)/30);
    end
    
    tic
    auc1 = [];
    auprc1 = [];
    f11 = [];
    gamma_ = [0.001,0.01,0.1,1,10];
    for run = 1:5
        
        auc11 = [];
        auprc11 = [];
        f111 = [];
        for gam = 1:length(gamma_)
            gamma = gamma_(gam);
            Ball = GGB_nonlinear(train_data_norm,min_gb,gamma);
          
            c = [10^-1,10^0,10^1];
            auc = [];
            auprc = [];
            
            for i = 1:length(c)
                c(i)
                %model = train_GBSVDDnonlinear(Ball,c(i));\
                model = train_SGBDnonlinear(Ball,c(i));
                results =test_SGBDnonlinear(Ball,test_data_norm,test_label,model,gamma);
                auc = [auc, results.performance.auc];
                auprc = [auprc,results.performance.auprc];
            end
            [~,idx] = max(auc);
            auc11 = [auc11,auc(idx)];
            auprc11 = [auprc11,auprc(idx)];
        end
        [~,idxx] = max(auc11);
        auc1 = [auc1,auc11(idxx)];
        auprc1 = [auprc1,auprc11(idxx)];
    end
    toc
    fprintf('AUCmean = %.4f\n',mean(auc1));
    fprintf('AUPRCmean = %.4f\n',mean(auprc1));
    fprintf('AUCstd = %.4f\n',std(auc1));
    fprintf('AUPRCstd = %.4f\n',std(auprc1));
    fprintf('time = %.4f\n',toc);
    
    AUCmean_results(ii,1) = mean(auc1);
    AUPRCmean_results(ii,1) = mean(auprc1);
    AUCstd_results(ii,1) = std(auc1);
    AUPRCstd_results(ii,1) = std(auprc1);
    time_results(ii,1) = toc;

    save('AUCmean.mat','AUCmean_results');
    save('AUCstd.mat','AUCstd_results');
    save('AUPRCmean.mat','AUPRCmean_results');
    save('AUPRCstd.mat','AUPRCstd_results');
    save('time.mat','time_results');
    
end
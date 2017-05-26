function des_accs = run_hmdb_split(split)
%run_split:
% Example:
%  run_split('split',1,'descriptor',{'mbhx','mbhy'}, 'encode', 'fv', 'gmmSize', 256, 'normalize', 'Power-Intra-L2', 'dataset', 'hmdb51')
%
%    descriptor: {'hog','hof','mbhx','mbhy'} or its subset.
%    encode: choose one method from {'fv','svc','svc-k','svc-all','vlad','vlad-k','vlad-all','llc','sa-k','vq'}
%    normalize: choose one method from {'Power-L2','Power-Intra-L2'}.


    gmmSize = 256;
    descriptorType = {'hog','hof','mbhx','mbhy'};
    encode_method = 'fv';
    normalize_method = 'Power-Intra-L2';

    [videoname, classlabel,fv_dir, vocab_dir, descriptor_path, video_dir, actions,tr_index] = getConfig(split);
    feat_path = fullfile(fv_dir, sprintf('feat_all_split_%d.mat', split));
    if ~exist(feat_path,'file')
        extractIDT(video_dir,videoname,descriptor_path);
        [gmm] = getIdtEncoder(split,videoname(tr_index==1),vocab_dir,descriptor_path, gmmSize);
        feat_all = encodeVideos(videoname,gmm,fv_dir,descriptor_path);
        save(feat_path,'feat_all','-v7.3');
    else
        fprintf('video file exists. loading %s..........\n',feat_path);
        %load(feat_path);
        %fprintf('loading sucesses.\n');
    end
    clear feat_all;
    clear gmm;
    tr_kern_sum = []; ts_kern_sum = [];
    des_accs = zeros(numel(descriptorType)+1,1);
    trn_indx  = find(tr_index==1);
    test_indx = find(tr_index==0);
    trainLabels = classlabel(trn_indx);
    testLabels = classlabel(test_indx);
    for i = 1 : numel(descriptorType)
        [~,ides] = ismember(descriptorType{i},{'trj','hog','hof','mbhx','mbhy'});
        if ~exist( fullfile('/home/hu/remote/Data/HMDB51/feats/train',sprintf('%s_%s_%d_Kern.mat',descriptorType{i},encode_method,split)),'file')
            pathOfEachFeature = fullfile(fv_dir, sprintf('fv_%s.mat', descriptorType{i}));
            load(pathOfEachFeature);
            feature = fvecs;
            clear fvecs;
            fprintf('%s.......... loaded\n',pathOfEachFeature);
            %feature = normalize(feature,'RaN', gmmSize);
            %feature = normalize(feature,'RootSift', gmmSize); % now feature in column-wise
            %feature = normalize(feature,normalize_method,gmmSize);
            %TrainData = feature(trn_indx,:);
            %save( fullfile('/home/hu/remote/Data/HMDB51/feats/train/o', sprintf('%d.mat',i)),'TrainData','TestData');
            TrainData = feature(trn_indx,:);
            TestData = feature(test_indx,:);
            clear feature;
            [TrainData,TestData] = ran_test(TrainData,TestData,3); 
            TrainData_Kern = TrainData * TrainData';
            TestData_Kern = TrainData * TestData';
            clear TrainData; clear TestData;
            %save(fullfile('/home/hu/remote/Data/HMDB51/feats/train',sprintf('%s_%s_%d_Kern.mat',descriptorType{i},encode_method,split)), 'TrainData_Kern', 'TestData_Kern','-v7.3');
        else
            load(fullfile('/home/hu/remote/Data/HMDB51/feats/train',sprintf('%s_%s_%d_Kern.mat',descriptorType{i},encode_method,split)));
        end
        if i==1
            tr_kern_sum = TrainData_Kern;
            ts_kern_sum = TestData_Kern;
        else
            tr_kern_sum = tr_kern_sum + TrainData_Kern;
            ts_kern_sum = ts_kern_sum + TestData_Kern;
        end
        score_test = svm_one_vs_all(TrainData_Kern, TestData_Kern, trainLabels', max(classlabel));
        fprintf('------------reranking------------');
        %score_test = reranking(score_test);
        [~, predict_labels] = max(score_test');
        [~,avg_acc,~] = get_cm(testLabels',predict_labels',1);
        des_accs(i) = avg_acc;
        fprintf('split---%d, %s--->accuracy:\n %f\n',split, descriptorType{i}, avg_acc);
        clear TrainData_Kern;
        clear TestData_Kern;
    end
    %save(fullfile('/home/hu/remote/Data/HMDB51/feats/train',sprintf('%d_%s_%s_SumKern.mat',split,encode_method,cell2mat(descriptorType))), 'tr_kern_sum', 'ts_kern_sum','-v7.3');
    score_test = svm_one_vs_all(tr_kern_sum, ts_kern_sum, trainLabels', max(classlabel));
    [~, predict_labels] = max(score_test');
    %save(sprintf('score_test%d.mat',split),'score_test','testLabels','predict_labels');
    [cm,avg_acc,~] = get_cm(testLabels',predict_labels',1);
    %save(sprintf('split%d_cm.mat',split),'cm');
    des_accs(end) = avg_acc;
    fprintf('split---%d, %d descriptor combination--->accuracy:\n %f\n',split, numel(descriptorType), avg_acc);
end

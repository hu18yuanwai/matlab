function [gmm] = getIdtEncoder(split, videoname,vocab_dir,descriptor_dir, gmmSize)
    if ~exist(vocab_dir,'dir'), mkdir(vocab_dir), end
    samples = 256000;
    gmm.gmmSize = gmmSize;
    pcaFactor = 0.5;
    whiten = 1;
    sampleFeatFile = fullfile(vocab_dir,sprintf('%d_featfile.mat',split));
    gmmFilePath = fullfile(vocab_dir,sprintf('%d_gmmModel_%d.mat',split,gmmSize));
    if exist(gmmFilePath,'file')
        tmp = load(gmmFilePath); %gmm
        gmm = tmp.gmm;
    end
    if ~exist(sampleFeatFile,'file')
        locationAll = zeros(samples,3);
        trjAll = zeros(samples,30);
        hogAll = zeros(samples,96);
        hofAll = zeros(samples,108);
        mbhxAll = zeros(samples,96);
        mbhyAll = zeros(samples,96);
        warning('getEncoder : generate encoder from subset of videos...')
        num_videos = 5100;
        if num_videos>numel(videoname), num_videos = numel(videoname); end
        num_samples_per_vid = ceil(samples/ num_videos);
        videoname = videoname(randperm(numel(videoname)));
        st = 1;
        for i = 1 : num_videos
            timest = tic();
            descriptorFile = fullfile(descriptor_dir,sprintf('%s.mat',videoname{i}));
            dt = load(descriptorFile);
            if ~isempty(dt)
                rnsam = randperm(size(dt.hog,1));
                if numel(rnsam) > num_samples_per_vid
                    rnsam = rnsam(1:num_samples_per_vid);
                end
                send = st + numel(rnsam) - 1;
                locationAll(st:send,:) = dt.obj(rnsam,8:10);
                trjAll(st:send,:) = dt.trj(rnsam,:);
                hogAll(st:send,:) = dt.hog(rnsam,:);
                hofAll(st:send,:) = dt.hof(rnsam,:);
                mbhxAll(st:send,:) = dt.mbhx(rnsam,:);
                mbhyAll(st:send,:) = dt.mbhy(rnsam,:);
            end
            st = st + numel(rnsam);
            timest = toc(timest);
            fprintf('%d/%d -> %s --> %1.2f sec\n',i,num_videos,videoname{(i)},timest);
        end
        if send ~= samples
            locationAll(send+1:samples,:) = [];
            trjAll(send+1:samples,:) = [];
            hogAll(send+1:samples,:) = [];
            hofAll(send+1:samples,:) = [];
            mbhxAll(send+1:samples,:) = [];
            mbhyAll(send+1:samples,:) = [];
        end
        fprintf('start computing pca\n');

        
        %gmm.pcamap.trj = pca(trjAll);
        %gmm.pcamap.hog = pca(hogAll);
        %gmm.pcamap.hof = pca(hofAll);
        %gmm.pcamap.mbhx = pca(mbhxAll);
        %gmm.pcamap.mbhy = pca(mbhyAll);
        [gmm.pcamap.trj, gmm.centre.trj] = xpca(trjAll', whiten, size(trjAll,2)*pcaFactor);
        [gmm.pcamap.hog, gmm.centre.hog] = xpca(hogAll', whiten, size(hogAll,2)*pcaFactor);
        [gmm.pcamap.hof, gmm.centre.hof] = xpca(hofAll', whiten, size(hofAll,2)*pcaFactor);
        [gmm.pcamap.mbhx, gmm.centre.mbhx] = xpca(mbhxAll', whiten, size(mbhxAll,2)*pcaFactor);
        [gmm.pcamap.mbhy, gmm.centre.mbhy] = xpca(mbhyAll', whiten, size(mbhyAll,2)*pcaFactor);

        fprintf('start saving descriptors\n');
        save(sampleFeatFile,'locationAll','trjAll','hogAll','hofAll','mbhxAll','mbhyAll','gmm','-v7.3');
        fprintf('save ends.\n')
    else
        load(sampleFeatFile);
    end
    %=========gmm & kmeans=============

    if exist(gmmFilePath,'file')
        load(gmmFilePath);
        fprintf('gmm has existed.\n');
        return;
    end
    fprintf('start create gmm  trj\n');
    trjProjected = bsxfun(@minus,trjAll,gmm.centre.trj) * gmm.pcamap.trj;
    trjProjected = [trjProjected locationAll];
    [gmm.means.trj, gmm.covariances.trj, gmm.priors.trj] = vl_gmm(trjProjected', gmmSize);
    clear trjProjected;
    clear trjAll;

    fprintf('start create gmm hog\n');
    hogProjected = bsxfun(@minus,hogAll,gmm.centre.hog) * gmm.pcamap.hog;
    hogProjected = [hogProjected locationAll];
    [gmm.means.hog, gmm.covariances.hog, gmm.priors.hog] = vl_gmm(hogProjected', gmmSize);
    clear hogAll;

    fprintf('start create gmm  hof\n');
    hofProjected = bsxfun(@minus,hofAll,gmm.centre.hof) * gmm.pcamap.hof;
    hofProjected = [hofProjected locationAll];
    [gmm.means.hof, gmm.covariances.hof, gmm.priors.hof] = vl_gmm(hofProjected', gmmSize);
    clear hofProjected;
    clear hofAll;

    fprintf('start create gmm  mbhx\n');
    mbhxProjected = bsxfun(@minus,mbhxAll,gmm.centre.mbhx) * gmm.pcamap.mbhx;
    mbhxProjected = [mbhxProjected locationAll];
    [gmm.means.mbhx, gmm.covariances.mbhx, gmm.priors.mbhx] = vl_gmm(mbhxProjected', gmmSize);
    clear mbhxProjected;
    clear mbhxAll;

    fprintf('start create gmm  mbhy\n');
    mbhyProjected = bsxfun(@minus,mbhyAll,gmm.centre.mbhy) * gmm.pcamap.mbhy;
    mbhyProjected = [mbhyProjected locationAll];
    [gmm.means.mbhy, gmm.covariances.mbhy, gmm.priors.mbhy] = vl_gmm(mbhyProjected', gmmSize);
    clear mbhyProjected;
    clear mbhyAll;

    fprintf('start saving gmm and codebook\n');
    save(gmmFilePath,'gmm','-v7.3');
end

function [gmm] = getIdtEncoder(split, videoname,vocab_dir,descriptor_dir, gmmSize)
    if ~exist(vocab_dir,'dir'), mkdir(vocab_dir), end
    samples = 256000;
    numWords = 4000;
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
        trjAll = zeros(samples,30);
        hogAll = zeros(samples,96);
        hofAll = zeros(samples,108);
        mbhxAll = zeros(samples,96);
        mbhyAll = zeros(samples,96);
        warning('getEncoder : generate encoder from subset of videos...')
        num_videos = 5200;
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
            trjAll(send+1:samples,:) = [];
            hogAll(send+1:samples,:) = [];
            hofAll(send+1:samples,:) = [];
            mbhxAll(send+1:samples,:) = [];
            mbhyAll(send+1:samples,:) = [];
        end
        fprintf('start computing pca\n');

        gmm.pcamap.trj = pca(trjAll);
        gmm.pcamap.hog = pca(hogAll);
        gmm.pcamap.hof = pca(hofAll);
        gmm.pcamap.mbhx = pca(mbhxAll);
        gmm.pcamap.mbhy = pca(mbhyAll);

        fprintf('start saving descriptors\n');
        save(sampleFeatFile,'trjAll','hogAll','hofAll','mbhxAll','mbhyAll','gmm','-v7.3');
        fprintf('save ends.')
    else
        load(sampleFeatFile);
    end
    %=========gmm & kmeans=============

    if exist(gmmFilePath,'file')
        load(gmmFilePath);
        fprintf('gmm has existed.\n');
        return;
    end
    fprintf('start create gmm & kmeans trj\n');

    trjProjected = trjAll * gmm.pcamap.trj(:,1:size(gmm.pcamap.trj,1)*pcaFactor);
    [gmm.means.trj, gmm.covariances.trj, gmm.priors.trj] = vl_gmm(trjProjected', gmmSize);

    fprintf('start create gmm & kmeans hog\n');

    hogProjected = hogAll * gmm.pcamap.hog(:,1:size(gmm.pcamap.hog,1)*pcaFactor);
    [gmm.means.hog, gmm.covariances.hog, gmm.priors.hog] = vl_gmm(hogProjected', gmmSize);

    fprintf('start create gmm & kmeans hof\n');
    hofProjected = hofAll * gmm.pcamap.hof(:,1:size(gmm.pcamap.hof,1)*pcaFactor);
    [gmm.means.hof, gmm.covariances.hof, gmm.priors.hof] = vl_gmm(hofProjected', gmmSize);

    fprintf('start create gmm & kmeans mbhx\n');
    mbhxProjected = mbhxAll * gmm.pcamap.mbhx(:,1:size(gmm.pcamap.mbhx,1)*pcaFactor);
    [gmm.means.mbhx, gmm.covariances.mbhx, gmm.priors.mbhx] = vl_gmm(mbhxProjected', gmmSize);

    fprintf('start create gmm & kmeans mbhy\n');
    mbhyProjected = mbhyAll * gmm.pcamap.mbhy(:,1:size(gmm.pcamap.mbhy,1)*pcaFactor);
    [gmm.means.mbhy, gmm.covariances.mbhy, gmm.priors.mbhy] = vl_gmm(mbhyProjected', gmmSize);

    fprintf('start saving gmm and codebook\n');
    save(gmmFilePath,'gmm','-v7.3');

    clear trjAll hogAll hofAll mbhxAll mbhyAll gmm;
end

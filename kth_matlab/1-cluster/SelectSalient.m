function [gmm] =  SelectSalient(gmmSize,samples,fullvideoname,descriptor_path,vocabDir)
    sampleFeatFile = fullfile(vocabDir,'SampleFeatfile.mat');
    gmmFilePath = fullfile(vocabDir,'gmm.mat');

    if exist(gmmFilePath,'file')
        temp = load(gmmFilePath);
        gmm = temp.gmm;
        return
    end

    pcaFactor = 0.5;
    whiten = 1;

    start_index = 1;
    end_index = 1;
    if ~exist(sampleFeatFile,'file')
        objAll = zeros(samples,10)
        trjAll = zeros(samples,30);
        hogAll = zeros(samples,96);
        hofAll = zeros(samples,108);
        mbhxAll = zeros(samples,96);
        mbhyAll = zeros(samples,96);

        num_videos = 3000;
        if num_videos>size(fullvideoname,1), num_videos = size(fullvideoname,1); end

        num_samples_per_vid = round(samples / num_videos);

		for i = 1:num_videos
	        timest = tic();
	        [~,partfile,~] = fileparts(fullvideoname{i});
	        descriptorFile = fullfile(descriptor_path,sprintf('%s.mat',partfile));
		        if exist(descriptorFile,'file')
		            load(descriptorFile);
				else
					fprintf('%s not exist !!!',descriptorFile);
                    [obj,trj,hog,hof,mbhx,mbhy] = extract_improvedfeatures(fullvideoname{i});
                    save(descriptorFile,'obj','trj','hog','hof','mbhx','mbhy');
		        end

		        hog = sqrt(hog); hof = sqrt(hof); mbhx = sqrt(mbhx);mbhy = sqrt(mbhy);

	        	rnsam = randperm(size(mbhx,1));
				if numel(rnsam) > num_samples_per_vid
		            rnsam = rnsam(1:num_samples_per_vid);
		        end

		        end_index = start_index + numel(rnsam) - 1;

                objAll(start_index:end_index,:) = obj(rnsam,:);
                trjAll(start_index:end_index,:) = trj(rnsam,:);
                hogAll(start_index:end_index,:) = hog(rnsam,:);
                hofAll(start_index:end_index,:) = hof(rnsam,:);
		        mbhxAll(start_index:end_index,:) = mbhx(rnsam,:);
                mbhyAll(start_index:end_index,:) = mbhy(rnsam,:);

		        start_index = start_index + numel(rnsam);
		        timest = toc(timest);
		        fprintf('sampling %d/%d -> %s --> %1.2f sec\n',i,num_videos,fullvideoname{(i)},timest);
                clear trj;
                clear hog;
                clear hof;
                clear mbhx;
                clear mbhy;
    	end
    	if end_index ~= samples
            trjAll(end_index+1:samples,:) = [];
            hogAll(end_index+1:samples,:) = [];
            hofAll(end_index+1:samples,:) = [];
    		mbhxAll(end_index+1:samples,:) = [];
            mbhyAll(end_index+1:samples,:) = [];
    	end

        fprintf('start computing pca\n');

        [gmm.pcamap.trj, gmm.centre.trj] = xpca(trjAll', whiten, size(trjAll,2)*pcaFactor);
        [gmm.pcamap.hog, gmm.centre.hog] = xpca(hogAll', whiten, size(hogAll,2)*pcaFactor);
        [gmm.pcamap.hof, gmm.centre.hof] = xpca(hofAll', whiten, size(hofAll,2)*pcaFactor);
        [gmm.pcamap.mbhx, gmm.centre.mbhx] = xpca(mbhxAll', whiten, size(mbhxAll,2)*pcaFactor);
        [gmm.pcamap.mbhy, gmm.centre.mbhy] = xpca(mbhyAll', whiten, size(mbhyAll,2)*pcaFactor);

        fprintf('start saving descriptors\n');
        save(sampleFeatFile,'objAll','trjAll','hogAll','hofAll','mbhxAll','mbhyAll');
        fprintf('sampled features saved.\n');
     else
     	load(sampleFeatFile);
    end

    % start to generating kmeans.

    numData = size(trjAll,1);

    fprintf('There are %d descriptors.',numData);
    %=========gmm & kmeans=============
    fprintf('start create gmm \n');
    trjProjected = bsxfun(@minus,trjAll,gmm.centre.trj) * gmm.pcamap.trj;
    trjProjected = [trjProjected objAll(:,8:10)];
    [gmm.means.trj, gmm.covariances.trj, gmm.priors.trj] = vl_gmm(trjProjected', gmmSize);

    fprintf('start create gmm & kmeans hog\n');
    hogProjected = bsxfun(@minus,hogAll,gmm.centre.hog) * gmm.pcamap.hog;
    hogProjected = [hogProjected objAll(:,8:10)];
    [gmm.means.hog, gmm.covariances.hog, gmm.priors.hog] = vl_gmm(hogProjected', gmmSize);

    fprintf('start create gmm & kmeans hof\n');
    hofProjected = bsxfun(@minus,hofAll,gmm.centre.hof) * gmm.pcamap.hof;
    hofProjected = [hofProjected objAll(:,8:10)];
    [gmm.means.hof, gmm.covariances.hof, gmm.priors.hof] = vl_gmm(hofProjected', gmmSize);

    fprintf('start create gmm & kmeans mbhx\n');
    mbhxProjected = bsxfun(@minus,mbhxAll,gmm.centre.mbhx) * gmm.pcamap.mbhx;
    mbhxProjected = [mbhxProjected objAll(:,8:10)];
    [gmm.means.mbhx, gmm.covariances.mbhx, gmm.priors.mbhx] = vl_gmm(mbhxProjected', gmmSize);

    fprintf('start create gmm & kmeans mbhy\n');
    mbhyProjected = bsxfun(@minus,mbhyAll,gmm.centre.mbhy) * gmm.pcamap.mbhy;
    mbhyProjected = [mbhyProjected objAll(:,8:10)];
    [gmm.means.mbhy, gmm.covariances.mbhy, gmm.priors.mbhy] = vl_gmm(mbhyProjected', gmmSize);

    fprintf('start saving gmm and codebook\n');
    save(gmmFilePath,'gmm');
end

function feat_all = encodeVideos(videoname,gmm,fv_dir,descriptor_dir)
%ENCODEVIDEOS:   encode all video IDT features with 'encode' method.
% For simplity, we only integrate Fisher vector method here
    pcaFactor = 0.5;
    if ~exist(fv_dir,'dir'), mkdir(fv_dir), end
    [path, ~, ~]=fileparts(videoname{1});
    if ~exist(fullfile(fv_dir,path),'dir')
        for i = 1 : numel(videoname)
            [path, ~, ~]=fileparts(videoname{i});
            if ~exist(fullfile(fv_dir,path), 'dir')
                mkdir(fullfile(fv_dir,path));
            end
        end
    end

    fv_trj = zeros( numel(videoname),size(gmm.pcamap.trj,2)*2*size(gmm.means.trj,2)*pcaFactor);
    for i = 1 : numel(videoname)
        timest = tic();
        savefile = fullfile(fv_dir, sprintf('%s_trj.mat',videoname{i}));
        if ~exist(savefile, 'file')
            descriptorFile = fullfile(descriptor_dir,sprintf('%s.mat',videoname{i}));
            dt = load(descriptorFile);
            if ~isempty(dt)
                trjProjected = dt.trj * gmm.pcamap.trj(:,1:size(gmm.pcamap.trj,1)*pcaFactor);
                fv_trj(i,:) = vl_fisher( trjProjected', gmm.means.trj, gmm.covariances.trj, gmm.priors.trj);
            else
                fv_trj(i,:) = 1/size(fv_trj,2);
            end
            save_fv_trj(savefile, fv_trj(i,:));
        else
            load(savefile);
            fv_trj(i,:) = fvec_trj;
        end
        timest = toc(timest);
        fprintf('trj-> %d -> %s -->  %1.1f sec.\n',i,videoname{i},timest);
    end
    save_all_fv(sprintf('%s/fv_trj.mat',fv_dir),fv_trj);
    clear fv_trj;

    fv_hog = zeros( numel(videoname),size(gmm.pcamap.hog,2)*2*size(gmm.means.hog,2)*pcaFactor);
    for i = 1 : numel(videoname)
        timest = tic();
        savefile = fullfile(fv_dir, sprintf('%s_hog.mat',videoname{i}));
        if ~exist(savefile, 'file')
            descriptorFile = fullfile(descriptor_dir,sprintf('%s.mat',videoname{i}));
            dt = load(descriptorFile);
            if ~isempty(dt)
                hogProjected = dt.hog * gmm.pcamap.hog(:,1:size(gmm.pcamap.hog,1)*pcaFactor);
                fv_hog(i,:) = vl_fisher( hogProjected', gmm.means.hog, gmm.covariances.hog, gmm.priors.hog);
            else
                fv_hog(i,:) = 1/size(fv_hog,2);
            end
            save_fv_hog(savefile, fv_hog(i,:));
        else
            load(savefile);
            fv_hog(i,:) = fvec_hog;
        end
        timest = toc(timest);
        fprintf('hog-> %d -> %s -->  %1.1f sec.\n',i,videoname{i},timest);
    end
    save_all_fv(sprintf('%s/fv_hog.mat',fv_dir),fv_hog);
    clear fv_hog;

    fv_hof = zeros( numel(videoname),size(gmm.pcamap.hof,2)*2*size(gmm.means.hof,2)*pcaFactor);
    for i = 1 : numel(videoname)
        timest = tic();
        savefile = fullfile(fv_dir, sprintf('%s_hof.mat',videoname{i}));
        if ~exist(savefile, 'file')
            descriptorFile = fullfile(descriptor_dir,sprintf('%s.mat',videoname{i}));
            dt = load(descriptorFile);
            if ~isempty(dt)
                hofProjected = dt.hof * gmm.pcamap.hof(:,1:size(gmm.pcamap.hof,1)*pcaFactor);
                fv_hof(i,:) = vl_fisher( hofProjected', gmm.means.hof, gmm.covariances.hof, gmm.priors.hof);
            else
                fv_hof(i,:) = 1/size(fv_hof,2);
            end
            save_fv_hof(savefile, fv_hof(i,:));
        else
            load(savefile);
            fv_hof(i,:) = fvec_hof;
        end
        timest = toc(timest);
        fprintf('hof->%d -> %s -->  %1.1f sec.\n',i,videoname{i},timest);
    end
    save_all_fv(sprintf('%s/fv_trj.mat',fv_dir),fv_hof);
    clear fv_hof;

    fv_mbhx = zeros( numel(videoname),size(gmm.pcamap.mbhx,2)*2*size(gmm.means.mbhx,2)*pcaFactor);
    for i = 1 : numel(videoname)
        timest = tic();
        savefile = fullfile(fv_dir, sprintf('%s_mbhx.mat',videoname{i}));
        if ~exist(savefile, 'file')
            descriptorFile = fullfile(descriptor_dir,sprintf('%s.mat',videoname{i}));
            dt = load(descriptorFile);
            if ~isempty(dt)
                mbhxProjected = dt.mbhx * gmm.pcamap.mbhx(:,1:size(gmm.pcamap.mbhx,1)*pcaFactor);
                fv_mbhx(i,:) = vl_fisher( mbhxProjected', gmm.means.mbhx, gmm.covariances.mbhx, gmm.priors.mbhx);
            else
                fv_mbhx(i,:) = 1/size(fv_mbhx,2);
            end
            save_fv(savefile, fv_mbhx(i,:));
        else
            load(savefile);
            fv_mbhx(i,:) = fvec_mbhx;
        end
        timest = toc(timest);
        fprintf('mbhx-> %d -> %s -->  %1.1f sec.\n',i,videoname{i},timest);
    end
    save_all_fv(sprintf('%s/fv_mbhx.mat',fv_dir),fv_mbhx);


    fv_mbhy = zeros( numel(videoname),size(gmm.pcamap.mbhy,2)*2*size(gmm.means.mbhy,2)*pcaFactor);
    for i = 1 : numel(videoname)
        timest = tic();
        savefile = fullfile(fv_dir, sprintf('%s_mbhy.mat',videoname{i}));
        if ~exist(savefile, 'file')
            descriptorFile = fullfile(descriptor_dir,sprintf('%s.mat',videoname{i}));
            dt = load(descriptorFile);
            if ~isempty(dt)
                mbhyProjected = dt.mbhy * gmm.pcamap.mbhy(:,1:size(gmm.pcamap.mbhy,1)*pcaFactor);
                fv_mbhy(i,:) = vl_fisher( mbhyProjected', gmm.means.mbhy, gmm.covariances.mbhy, gmm.priors.mbhy);
            else
                fv_mbhy(i,:) = 1/size(fv_mbhy,2);
            end
            save_fv_mbhy(savefile,fv_mbhy(i,:));
        else
            load(savefile);
            fv_mbhy(i,:) = fvec_mbhy;
        end
        timest = toc(timest);
        fprintf('%d -> %s -->  %1.1f sec.\n',i,videoname{i},timest);
    end
    save_all_fv(sprintf('%s/fv_mbhy.mat',fv_dir),fv_mbhy);
    feat_all = {fv_mbhx, fv_mbhy};
end

function save_all_fv(filepath,fvecs)
   save(filepath,'fvecs','-v7.3');
end

function save_fv_trj(filepath,fvec_trj)
   save(filepath,'fvec_trj');
end

function save_fv_hog(filepath,fvec_hog)
   save(filepath,'fvec_hog');
end

function save_fv_hof(filepath,fvec_hof)
   save(filepath,'fvec_hof');
end

function save_fv_mbhx(filepath,fvec_mbhx)
   save(filepath,'fvec_mbhx');
end

function save_fv_mbhy(filepath,fvec_mbhy)
   save(filepath,'fvec_mbhy');
end




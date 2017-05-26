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
    fv_mbhx = zeros( numel(videoname),size(gmm.pcamap.mbhx,2)*2*size(gmm.means.mbhx,2)*pcaFactor);
    fv_mbhy = zeros( numel(videoname),size(gmm.pcamap.mbhy,2)*2*size(gmm.means.mbhy,2)*pcaFactor);
    for i = 1 : numel(videoname)
        timest = tic();
        savefile = fullfile(fv_dir, sprintf('%s.mat',videoname{i}));
        if ~exist(savefile, 'file')
            descriptorFile = fullfile(descriptor_dir,sprintf('%s.mat',videoname{i}));
            dt = load(descriptorFile);
            if ~isempty(dt)
                fv_mbhx(i,:) = vl_fisher( (dt.mbhx * gmm.pcamap.mbhx(:,1:size(gmm.pcamap.mbhx,1)*pcaFactor))', gmm.means.mbhx, gmm.covariances.mbhx, gmm.priors.mbhx);
                fv_mbhy(i,:) = vl_fisher( (dt.mbhy * gmm.pcamap.mbhy(:,1:size(gmm.pcamap.mbhy,1)*pcaFactor))', gmm.means.mbhy, gmm.covariances.mbhy, gmm.priors.mbhy);
            else
                fv_mbhx(i,:) = 1/size(fv_mbhx,2);
                fv_mbhy(i,:) = 1/size(fv_mbhy,2);
            end
            save_fv(savefile,fv_mbhx(i,:), fv_mbhy(i,:));
        else
            load(savefile);
            fv_mbhx(i,:) = fvec_mbhx; fv_mbhy(i,:) = fvec_mbhy;
        end
        timest = toc(timest);
        fprintf('%d -> %s -->  %1.1f sec.\n',i,videoname{i},timest);
    end
    feat_all = {fv_mbhx, fv_mbhy};
    save_all_fv(sprintf('%s/fv_mbhx.mat',fv_dir),fv_mbhx);
    save_all_fv(sprintf('%s/fv_mbhy.mat',fv_dir),fv_mbhy);
end

function save_all_fv(filepath,fvecs)
   save(filepath,'fvecs','-v7.3');
end
function save_fv(filepath, fvec_mbhx, fvec_mbhy)
   save(filepath,'fvec_mbhx', 'fvec_mbhy');
end

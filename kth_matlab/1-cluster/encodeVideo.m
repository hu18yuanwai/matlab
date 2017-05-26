function  encodeVideos(videoname,gmm,codebook,fv_dir,descriptor_dir,class_category)
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

    fv_trj = zeros( 1,(size(gmm.pcamap.hog,2)+3)*2*size(gmm.means.hog,2));
    fv_hog = zeros( 1,(size(gmm.pcamap.hog,2)+3)*2*size(gmm.means.hog,2));
    fv_hof = zeros( 1,(size(gmm.pcamap.hof,2)+3)*2*size(gmm.means.hof,2));
    fv_mbhx = zeros( 1,(size(gmm.pcamap.mbhx,2)+3)*2*size(gmm.means.mbhx,2));
    fv_mbhy = zeros( 1,(size(gmm.pcamap.mbhy,2)+3)*2*size(gmm.means.mbhy,2));

    video_dir = '~/remote/KTH/';
    category = dir(video_dir);
    for i = 3:length(category) % 1-6 actions
    	timest = tic();
    	for j = 1:25
    		for k = 1:4 % for clips
    			descriptorFile = [];
				clipName = 'person';
				clipName = sprintf('%s%02d',clipName,j);
				clipName = sprintf('%s_%s_d%d_uncomp',clipName,category(i).name,k);
				descriptorFile = fullfile(descriptor_path,sprintf('%s.mat',clipName));
                savefile_trj = fullfile(fv_dir,sprintf('trj_%d.mat',j));
                savefile_hog = fullfile(fv_dir,sprintf('hog_%d.mat',j));
                savefile_hof = fullfile(fv_dir,sprintf('hof_%d.mat',j));
                savefile_mbhx = fullfile(fv_dir,sprintf('mbhx_%d.mat',j));
                savefile_mbhy = fullfile(fv_dir,sprintf('mbhy_%d.mat',j));

                if ~exist(savefile_trj,'file')
				    fprintf('%s is descriptorFile \n', descriptorFile);
				    if exist(descriptorFile,'file')
					    dt=load(descriptorFile);
					    fprintf('load %s complete \n',descriptorFile);
        		    else
        			    fprintf('%s not exist !!!\n',descriptorFile);
        			    return;
                    end

                    if ~isempty(dt)
                        fv_trj(1,:) = vl_fisher( [(bsxfun(@minus,dt.trj,gmm.centre.trj)*gmm.pcamap.hog) dt.obj(:,8:10)]', gmm.means.trj, gmm.covariances.trj, gmm.priors.trj);
                        fv_hog(1,:) = vl_fisher( [(bsxfun(@minus,dt.hog,gmm.centre.hog)*gmm.pcamap.hog) dt.obj(:,8:10)]', gmm.means.hog, gmm.covariances.hog, gmm.priors.hog);
                        fv_hof(1,:) = vl_fisher( [(bsxfun(@minus,dt.hof,gmm.centre.hof)*gmm.pcamap.hof) dt.obj(:,8:10)]', gmm.means.hof, gmm.covariances.hof, gmm.priors.hof);
                        fv_mbhx(1,:) = vl_fisher( [(bsxfun(@minus,dt.mbhx,gmm.centre.mbhx)*gmm.pcamap.mbhx) dt.obj(:,8:10)]', gmm.means.mbhx, gmm.covariances.mbhx, gmm.priors.mbhx);
                        fv_mbhy(1,:) = vl_fisher( [(bsxfun(@minus,dt.mbhy,gmm.centre.mbhy)*gmm.pcamap.mbhy) dt.obj(:,8:10)]', gmm.means.mbhy, gmm.covariances.mbhy, gmm.priors.mbhy);
                    else
                        fv_trj(1,:) = 1/size(fv_trj,2);
                        fv_hog(1,:) = 1/size(fv_hog,2);
                        fv_hof(1,:)= 1/size(fv_hof,2);
                        fv_mbhx(1,:)= 1/size(fv_mbhx,2);
                        fv_mbhy(1,:)= 1/size(fv_mbhy,2);
                    end
                    class_label = class_category{i};
                    fv_hogTerm = [class_label, fv_hog];
                    fv_hofTerm = [class_label, fv_hof];
                    fv_mbhxTerm = [class_label, fv_mbhx];
                    fv_mbhyTerm = [class_label, fv_mbhy];

                    dlmwrite(savefile_trj,fv_trjTerm, '-append');
                    dlmwrite(savefile_hog,fv_hogTerm, '-append');
                    dlmwrite(savefile_hof,fv_hofTerm, '-append');
                    dlmwrite(savefile_mbhx,fv_mbhxTerm, '-append');
                    dlmwrite(savefile_mbhy,fv_mbhyTerm, '-append');
                else
                    fprintf('the encoder of the video %s exists', clipName );
                end
                timest = toc(timest);
                fprintf('%d/%d -> %s --> %1.2f sec\n',i-2,length(category)-2,category(i-2).name,timest);
            end
        end
    end
end

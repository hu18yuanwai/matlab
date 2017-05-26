function getSalient(st,send,fullvideoname,descriptor_path)
    totalSize = 0;
    num_videos = size(fullvideoname,1);
    for i = st : min(send,numel(fullvideoname))
        timest = tic();
        [~,partfile,~] = fileparts(fullvideoname{i});
        descriptorFile = fullfile(descriptor_path,sprintf('%s.mat',partfile));
        if exist(descriptorFile,'file')
            load(descriptorFile);
        else
            try
            [obj,trj,hog,hof,mbhx,mbhy] = extract_improvedfeatures(fullvideoname{i});
            save(descriptorFile,'obj','trj','hog','hof','mbhx','mbhy');
            catch e
                fprintf('ERROR %s\n');
                e
            end
        end
        sDescription = dir(descriptorFile);
        sizef = sDescription.bytes / (1024 * 1024);
        timest = toc(timest);
        fprintf('%d -> %s --> size %1.1f Mb--> %1.1f sec.\n',i,descriptorFile,sizef,timest);
    end
end

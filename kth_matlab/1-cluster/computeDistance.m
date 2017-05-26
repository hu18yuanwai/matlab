function computeDistance(fullvideoname,featDir)
	st = 1;

	%send = size(fullvideoname);
	send = 10;
	video_dir = '~/remote/KTH/';
    category = dir(video_dir);


	for itest = 1:25
		timest = tic();

		trainfeatFile = {};
		trainlabel = {};
		testfeatFile = {};
		testlabel = {};

		for i=3 :length(category)
			for iTrain = 1:25
				for k = 1:4
					FeatFile = [];
					clipName = 'person';
					clipName = sprintf('%s%02d',clipName,iTrain);
					clipName = sprintf('%s_%s_d%d_uncomp',clipName,category(i).name,k);
					FeatFile = fullfile(featDir,sprintf('%s.mat',clipName));
					fprintf('%s is features file. \n', FeatFile);
					if itest == iTrain
						testfeatFile = [testfeatFile FeatFile];
						testlabel = [testlabel category(i)];
					else
						trainfeatFile = [trainfeatFile FeatFile];
						trainlabel = [trainlabel category(i)];
					end
				end
			end
		end
	end
end

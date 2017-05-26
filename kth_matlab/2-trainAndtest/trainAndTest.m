function trainAndTest(fullvideoname,featDir)
	video_dir = '~/remote/KTH/';
    category = dir(video_dir);
    nClasses = 6;
    nCorrect = 0;
    result = zeros(nClasses,nClasses);
	for j = 1:25
		timest = tic();
		distanceFile = fullfile(featDir,sprintf('/distance/%d.mat',j));
		fprintf('%s is distanceFile\n', distanceFile);
		dlmread(distanceFile);

		trainlabel = total(1:trainSize,:);
		testlabel = total(1:testSize,:);

		total = total(2:size(total,1),:);
		nTotal = 0;
		nTrain = 1:trainSize;
		nTest = trainSize+1:trainSize+testSize;
		trainData = [nTrain' total(trainSize,:)];
		testData = [nTest' total(testSize,:)];
		C = [1 10 100 500 1000 ];
        for ci = 1 : numel(C)
             model(ci) = svmtrain(trainlabel, trainData, sprintf('-t 4 -c %1.6f -v 2 -q ',C(ci)));               
        end        
        [~,max_indx]=max(model);
        
        C = C(max_indx);
 		
 		for ci = 1 : numel(C)
             model = svmtrain(TrainClass, TrainData_Kern, sprintf('-t 4 -c %1.6f  -q ',C(ci)));
             [predicted_label{ci}, acc, scores{ci}] = svmpredict(testlabel, testData ,model);	                 
             accuracy(ci) = acc(1,1);
        end
        [acc,cindx] = max(accuracy); 
        best_predicted_label =  predicted_label{cindx};
        
    	for i = 1: testSize
    		nTotal += 1;
    		if best_predicted_label(i) == testlabel(i)
    			nCorrect = nCorrect + 1;
    		end
    		result(testlabel(i),best_predicted_label(i)) = result(testlabel(i),best_predicted_label(i))+1;
    	end
	end

	average_accuracy = 0;
	for i = 1:nClasses
		nsequences = sum(result(i));
		average_accuracy = average_accuracy + result(i,i)/nsequences;
	end    	
	average_accuracy /= nClasses;
	accuracy = nCorrect / nTotal;
	save(resultFile,result,average_accuracy,accuracy);
	fprintf('average_accuracy is %f, and accuracy is %f',average_accuracy,accuracy);
end
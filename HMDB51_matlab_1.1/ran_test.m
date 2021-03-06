function [TRAIN,X] = ran_test(TRAIN,X,subset)
	%N = size(TRAIN,1);
    n = randperm( size(TRAIN,1));
    b = TRAIN(n(1: subset),:);
    for i = 1: size(X,1)
        i
        Y = [X(i,:);b];
        for j = 1:size(Y,2);
            [~,pos]=sort(Y(:,j));
            for k = 1: size(Y,1);
                if pos(k,1) == 1
                    X(i,j) = k/subset;
                end
            end
        end
    end

    for i = 1: size(TRAIN,1)
        i
        Y = [TRAIN(i,:);b];
        for j = 1:size(Y,2);
            [~,pos]=sort(Y(:,j));
            for k = 1: size(Y,1);
                if pos(k,1) == 1
                    TRAIN(i,j) = k/subset;
                end
            end
        end
    end
end


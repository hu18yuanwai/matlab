function X = normalize(X, METHOD, ncomponent,ADD)
    switch METHOD
        case 'Power-L2'
            X = sign(X).*sqrt(abs(X));
            X = normalizeL2(X);
        case 'Power-Intra-L2'
            X = sign(X).*sqrt(abs(X));
            X = normalizeL2(X);
            n = size(X,1);
            X = (normalizeIntra(X', ncomponent,n))';
        case 'L2'
            X = normalizeL2(X);
        case 'Intra'
            n = size(X,1);
            X = (normalizeIntra(X', ncomponent, n))';
        case 'RootSift'
            X = normalizeL1(X);
            X = sign(X).*sqrt(abs(X));
            X = normalizeL2(X);
        case 'RaN'
            X=normalizeRaN(X);
        case 'RaNp'
            for i = 1:size(X,1)
                i
                B = [X(i,:); ADD];
                B = normalizeRaN(B);
                X(i,:) = B(1,:);
            end
    end
end

function X = normalizeRaN(X)
    N = size(X,1);
    for i = 1: size(X,2)
        [c,pos] = sort( X(:,i) );
        for j = 1:N
            X(pos(j,1),i) = j/N;
        end
        %X(:,i) = pos*(c(N,1)-c(1,1)) / N;
        %X(:,i) = pos / N;
    end
end

function X = normalizeL2(X)
	for i = 1 : size(X,1)
		X(i,:) = X(i,:) / (eps+sqrt(sum(X(i,:).^2)));
    end
end

function X = normalizeL1(X)
    for i = 1: size(X,1)
        X(i,:) = X(i,:) / (eps+sum(abs(X(i,:))));
    end
end

function X = normalizeIntra(X, ncomponent,n)
% X in colume-wise
    X = reshape(X,[size(X,1)/ncomponent,ncomponent*n]);
    X = bsxfun(@rdivide,X,eps+sqrt(sum(X.^2)));
    X = reshape(X,[size(X,1)*ncomponent,n]);
end
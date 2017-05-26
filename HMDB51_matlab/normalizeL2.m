function X = normalizeL2(X)
	for i = 1 : size(X,1)
		X(i,:) = X(i,:) / (eps+sqrt(sum(X(i,:).^2)));
    end
end
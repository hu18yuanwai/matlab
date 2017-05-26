function X = normalize(X, METHOD, ncomponent)
    switch METHOD
        case 'Power-L2'
            X = normalizeL2(X);
            X = sign(X).*sqrt(abs(X));
        case 'Power-Intra-L2'
            X = sign(X).*sqrt(abs(X));
            n = size(X,2);
            X = normalizeIntra(X, ncomponent,n);
            X = normalizeL2(X);
        case 'L2'
            x = normalizeL2(X);
    end
end

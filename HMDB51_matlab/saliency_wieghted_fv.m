function code = encodeFeatures(descrs,encoder,use_vlfeat,saliency)
	P = get_posteriors(descrs, encoder.means, encoder.covariances, encoder.priors);
    P(find(P<1e-4)) = 0;
    P = bsxfun(@rdivide,P,sum(P,1));
    dimension = size(descrs,1); numData = size(descrs,2);
    sqrtInvSigma = sqrt(1./encoder.covariances);
    uprefix = 1./(sqrt(encoder.priors));%numData*
    vprefix = 1./(sqrt(2*encoder.priors));%numData*
    z = zeros(2*dimension*encoder.numWords,1);
    for gmm_i = 1:encoder.numWords
       if encoder.priors(gmm_i)<1e-6,
       	 continue;
       end
       diff = bsxfun(@minus,descrs,encoder.means(:,gmm_i));
       diff = bsxfun(@times, diff, sqrtInvSigma(:,gmm_i));
       % mean
       z((gmm_i-1)*dimension+1:gmm_i*dimension) = ...
            uprefix(gmm_i)*sum(bsxfun(@times,P(gmm_i,:), diff),2);
              % var
       z(encoder.numWords*dimension+(gmm_i-1)*dimension+1:encoder.numWords*dimension+gmm_i*dimension) = ...
                  vprefix(gmm_i)*sum(bsxfun(@times,P(gmm_i,:), diff.^2 - 1),2);
    end
    code = z(:)';
end

function posteriors = get_posteriors(descrs,means,covariances,priors)
  dimension = size(descrs,1);
  numData = size(descrs,2); numClusters = length(priors);
  posteriors = zeros(numClusters, numData,'single');
  logWeights = log(priors);
  logCovariances = log(covariances); logCovariances = sum(logCovariances,1);
  invCovariances = 1./covariances;
  halfDimLog2Pi = (dimension / 2.0) * log(2.0*pi);

  for i = 1:numClusters
      tmp = bsxfun(@minus,descrs,means(:,i));
      p = logWeights(i) - halfDimLog2Pi - 0.5 * logCovariances(i) - ...
          0.5 * sum(bsxfun(@times,bsxfun(@times, tmp,invCovariances(:,i)),tmp),1);
      posteriors(i,:) = p;    
  end
  maxPosterior = max(posteriors,[],1);
  posteriors2 = bsxfun(@minus,posteriors,maxPosterior);
  posteriors2 = exp(posteriors2);
  posteriors = bsxfun(@times,posteriors2,1./sum(posteriors2,1));
end    
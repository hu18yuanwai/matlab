function  p = reranking(score_test)
	%score_test = normToOne(score_test);
	beta = 1;
	gamma = 0.5;
	iter = 5;
	p = score_test;
	for it = 1: iter
		for i = 1 : size(p,1)
			w = sort(p(i,:),'descend');
			r = (1:size(p,2));
			for j = 1 : size(p,2)
				tag = find(w == p(i,j));
				ww = [w(1:tag-1) w(tag+1:size(w,2))];
				ww = [ww(1:j-1) p(i,j) ww(j:size(ww,2))];

                suma = sum(ww.*exp(-beta*r));
                sumb = p(i,j)*exp(-beta*j);
				p(i,j) = p(i,j) - gamma.^(it-1)* ( suma - sumb);
				clear ww;
			end
			clear w;
		end 
	end
end

function p = normToOne(p)
	for i = 1:size(p,1)
		maxp = max(p(i,:));
		minp = min(p(i,:));
		p(i,:) = (p(i,:) - minp)./(maxp - minp);
	end
end

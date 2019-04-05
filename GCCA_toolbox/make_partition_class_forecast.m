function [test_predictions] = make_partition_class_forecast(X);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is uses the stored output from "bayes_partition_class.m" to make predictions on a test set
%
%
%  Version 1.0 	Date: 8th October 2002
%
%	Writen by Chris Holmes: c.holmes@ic.ac.uk, for academic purposes only.
%									please contact me if you intend to use this for commercial work
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%	USAGE:
%			
%        Samples from the post burn Markov chain are stored in subdirectory ./PARTITION_MULT_MCMC_samples
%
%			THIS PROGRAM MUST BE RUN IN THE PARENT DIRECTORY OF "./PARTITION_MULT_MCMC_samples"
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% INPUTS:
%
%		covariates X, n by p matrix, n data points, p predictors
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%
% OUTPUTS:
%			
%     test_predictions is an object containing:
%				pred_store is mean prediction on the test set
%				credibles is 95% credible intervals around pred_store
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% get data dimensions
[n p]=size(X);

% change directory to where MCMC samples are stored
cd PARTITION_MULT_MCMC_samples

% get options
load options;

load number_of_categories.dat
q = number_of_categories;

pred_store = zeros(n,q); % mean predictions
test_predictions.pred_store = zeros(n,q); % the predictions stored
test_predictions.credible_lower = zeros(n,q); % the 95% credible interval below predictions
test_predictions.credible_upper = zeros(n,q); % the 95% credible interval above predictions


its = options.mcmc_samples;

% get standardising factors
load mx; load sx;
for j=1:p
   X(:,j) = (X(:,j)-mx(j))/sx(j);
end


% how many samples must we store for the mcmc 95% credible intervals?
cred_n = ceil(0.025*its);
cred_upper = zeros(n,cred_n);
cred_lower = zeros(n,cred_n);

sample = 0;

for i=1:its
   
   if rem(i,100)==0
      fprintf('Calculating %d / %d samples \n', i, its);
   end
   
   sample = sample+1;
   
   str = ['mcmc_model_' int2str(i)];
   load(str);
   
   Dist = get_dist(model.theta(1:model.k,:),X); % get distance between centres and test points
   
   if model.k==1
      W = ones(1,n);
   else
      [dum W] = min(Dist); % W stores indicator of which centres are closest
   end
   
   
   aa = zeros(n,q); % used to store predictions
   
   for jj=1:model.k  % for each partition
      indx = find(W==jj); % find the points associated with this partition
      aa(indx,:) = ones(length(indx),1)*model.p_j_given_x(jj,:); % make predictions from the j'th partition
   end
   
   pred_store = pred_store + aa; % add on mean prediction
   
   % now check and store credibles
   for j=1:q
      a = aa(:,j);
      if sample <=cred_n % if we still have'nt filled the credible store
         cred_class_upper(:,sample,j)=a;
         cred_class_lower(:,sample,j)=a;
         if sample==cred_n % get min and max if this is the last sample to fill the store
            [min_class_cred_upper(:,j) min_class_indx(:,j)] = min(cred_class_upper(:,:,j),[],2);
            [max_class_cred_lower(:,j) max_class_indx(:,j)] = max(cred_class_lower(:,:,j),[],2);
         end
      else % we have filled up the credible store
         % check to see if any current predictions are in upper band
         find_upper = find(a > min_class_cred_upper(:,j));
         % if there are any we must insert them into the store
         for jj=1:length(find_upper)
            row_indx = find_upper(jj);
            cred_class_upper(row_indx,min_class_indx(row_indx,j),j) = a(row_indx);
            % recalculate the minimal upper and it's index
            [min_class_cred_upper(row_indx,j) min_class_indx(row_indx,j)] = min(cred_class_upper(row_indx,:,j));
         end
         % now do the same for the lower credibles.....
         % check to see if any in lower band
         find_lower = find(a < max_class_cred_lower(:,j));
         for jj=1:length(find_lower)
            row_indx = find_lower(jj);
            cred_class_lower(row_indx,max_class_indx(row_indx,j),j) = a(row_indx);
            [max_class_cred_lower(row_indx,j) max_class_indx(row_indx,j)] = max(cred_class_lower(row_indx,:,j));
         end   
      end
   end
   
end

% get MCMC mean
test_predictions.pred_store = pred_store/its;

% calculate credibles
test_predictions.credible_upper = min_class_cred_upper;
test_predictions.credible_lower = max_class_cred_lower;

cd ..

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [dist] = get_dist(centre,data)
% This calculates the distance between objects
% INPUTS
%		centre - points to calcluate distance to...
%	   data - this is the data matrix
%
%  OUTPUTS
%
%		dist - Euclidean distance between centre and data

[n p]=size(data);

[nc pc] = size(centre);

if (p ~= pc) 
   error('CENTRE DIMENSION AND INPUT DIMENSION MUST MATCH ');
end

% calculate distance matrix from knot points to data points
dist = zeros(nc,n);
x_knot = sum(centre.^2,2)*ones(1,n);
x_data = sum(data.^2,2)*ones(1,nc);
xx = centre*data';
dist = x_knot + x_data' - 2*xx;
dist_sq = dist;
dist = sqrt(dist_sq);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


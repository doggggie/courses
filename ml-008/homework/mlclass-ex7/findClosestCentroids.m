function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

m = size(X,1);
for i=1:m
    % look at example X(i,:)
    centroid_idx = 1;
    d2centroid = norm(X(i,:) - centroids(1,:), 2);
    for k = 2:K
        dist = norm(X(i,:) - centroids(k,:), 2);
        if dist < d2centroid
            d2centroid = dist;
            centroid_idx = k;
        end
    end
    idx(i) = centroid_idx;
end





% =============================================================

end


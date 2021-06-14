%% Computing the exact Gram Matrix K(*, *)
% X: The first sampling positions of the distribution, size (d, n)
% X_dash: The second sampling positions of the distribution, size (d, m)

function gramMatrix = calculateExactGramMatrixCurl(X, X_dash, magnitude_scale_const, magnitude_scale_SE, length_scale)

d = size(X, 1);
n = size(X, 2);
m = size(X_dash, 2);

gramMatrix = zeros(3*n, 3*m);

Identity = repmat(eye(3), 1, m);

for i = 1 : n
     gramMatrix(i*3-2 : i*3, :) = curlFreeKernel(X(:, i), X_dash, magnitude_scale_SE, length_scale) + Identity * magnitude_scale_const^2; 
end

end
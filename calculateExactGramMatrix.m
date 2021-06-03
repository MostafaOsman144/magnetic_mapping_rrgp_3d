%% Computing the exact Gram Matrix K(*, *)
% X: The first sampling positions of the distribution, size (d, n)
% X_dash: The second sampling positions of the distribution, size (d, m)

function gramMatrix = calculateExactGramMatrix(X, X_dash, magnitude_scale_lin, magnitude_scale_SE, length_scale)

d = size(X, 1);
n = size(X, 2);
m = size(X_dash, 2);

gramMatrix = zeros(n, m);

for i = 1 : n
     gramMatrix(i, :) = squaredExponentialKernel(X(:, i), X_dash, magnitude_scale_SE, length_scale) + linearKernel(X(:, i), X_dash, magnitude_scale_lin); 
%     gramMatrix(i, :) = linearKernel(X(:, i), X_dash(:, i), magnitude_scale_lin); 
%     gramMatrix(i, :) = squaredExponentialKernel(X(:, i), X_dash, magnitude_scale_SE, length_scale);
end

end
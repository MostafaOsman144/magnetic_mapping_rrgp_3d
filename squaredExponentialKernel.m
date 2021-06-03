%% Implementation of the Squared Exponential Kernel
% first_position: the value of the first argument to the kernel. (d, 1)
% second_position: the value of the second argument to the kernel. (d, m)
% d : size
% magnitude_scale: the magnitude hyperparameter of the kernel. Known as
% sigma_SE in the mathematical context.
% length_scale: known as l_SE in the mathematical context.

function result = squaredExponentialKernel(first_position, measurements_positions, magnitude_scale, length_scale)

difference = first_position - measurements_positions;
euclidean_norm_squared = difference(1, :).^2 + difference(2, :).^2 + difference(3, :).^2;
result = magnitude_scale^2 * exp(-euclidean_norm_squared.^2 / (2 * length_scale^2));
end
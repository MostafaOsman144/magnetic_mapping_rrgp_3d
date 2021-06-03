%% Impelemtation of the linear kernel
% first_position: the value of the first argument to the kernel. (d, 1)
% measurements_positions: the value of the second argument to the kernel. (d, m)
% d : size
% magnitude_scale: the magnitude hyperparameter of the kernel.

function result = linearKernel(first_position, measurements_positions, magnitude_scale)
result = magnitude_scale ^ 2 * first_position' * measurements_positions;
end
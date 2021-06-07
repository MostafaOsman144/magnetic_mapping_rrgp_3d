%% Implementation of the batch estimation of the Gaussian Process regression. 

function [mean, cov] = batchEstimation(eigenfunctions_training, spectral_eigenvalues, eigenfunctions_test, measurements, measurement_noise)
m = size(spectral_eigenvalues, 1); % Number of basis functions
d = size(measurements, 1); % number of dimensions of the measurement vector

vec_measurements = reshape(measurements, size(measurements, 1) * size(measurements, 2), 1);
spectral_eigenvalues = spectral_eigenvalues + 1e-8 * eye(size(spectral_eigenvalues, 1)); % Added a small value for numerical stability when computing the inverse. 

mean = eigenfunctions_test * ((eigenfunctions_training' * eigenfunctions_training + (measurement_noise^2*eye(m)/spectral_eigenvalues)) \ eigenfunctions_training') * vec_measurements;
cov = measurement_noise^2 * eigenfunctions_test * (eigenfunctions_training' * eigenfunctions_training + (measurement_noise^2*eye(m)/spectral_eigenvalues)) * eigenfunctions_test';

number_of_training_positions = size(eigenfunctions_test, 1)/d;
mean = reshape(mean, d, number_of_training_positions);
end
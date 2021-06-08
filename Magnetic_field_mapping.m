%% Magnetic Field Mapping in 3D
% Implementation using the reduced rank Gaussian process regression 
% described in the paper titled "Modeling and Interpolation of the Ambient 
% Magnetic Field by Gaussian Processes" by Arno Solin, Manon Kok, Niklas
% Wahlstrom, Thomas B. Schon, and Simo Sarkka. Umlauts in the names were
% changed to normal letters for convenience. 
% 
% The magnetic field approach adopted is through using the gradient of
% the scalar potential (see the paper for more information).

clear; close all; clc;


%% Configurable parameters needed in the script for the ease of testing
load('measurements_aligned_simple_trajectory.mat'); % Simple scenario measurements
% load('measurements_aligned_complicated_trajectory.mat'); % More complex scenario measurements

dimensions = 3;

downscale = 20; % Downsampling of the data for reducing running time and avoiding RAM overflow.
training_set_factor = 100/100; % Percentage for dividing the dataset into test and training datasets for the batch estimation problem.

% Initializing the GP hyperparameters.
magnitude_scale_SE = 1;
length_scale_SE = 0.3;
magnitude_scale_lin = 1;
measurement_noise = 0.1;

space_margin = 0.5; % Defining the margin in the dirichlet boundary conditions.

number_of_basis_functions = 2000; % The number of basis functions for the approximation of the gram matrix.

% Learning rate for the hyperparameters optimization problem (not working
% yet).
learning_rate = 0.1;

%% Organizing the data read from the data file. 
sampling_time = T;
magnetic_measurements = u(7:end, :);
positions = p_opti;
orientations_quat = q_opti;
gravity = g;

permutation_index = generateIndexMat(number_of_basis_functions); % generating the set of permutations for indexing the eigenfunctions and eigenvalues

positions = positions(:, 1 : downscale : end);
orientations_quat = orientations_quat(:, 1 : downscale : end);
orientation_mat = quat2rotm(orientations_quat');
number_of_measurements = size(positions, 2);
magnetic_measurements = magnetic_measurements(:, 1 : downscale : end);
sampling_time = sampling_time * downscale;

% Correcting the orientations of magnetic field measurements
for i = 1 : number_of_measurements
   magnetic_measurements(:, i) = orientation_mat(:, :, i) * magnetic_measurements(:, i); 
end

% Normalizing the measurements data (The normalization here is accomplished with an independence assumption)
means = zeros(size(magnetic_measurements, 1));
stds = zeros(size(magnetic_measurements, 2));

for i = 1 : length(means)
   means(i) = mean(magnetic_measurements(i, :)); 
   stds(i) = std(magnetic_measurements(i, :));
   magnetic_measurements(i, :) = (magnetic_measurements(i, :) - means(i)) / stds(i);
end

% Forming the training and test datasets for the batch estimation problem.
training_size = ceil(number_of_measurements * training_set_factor);
if(training_set_factor == 1) % For the case of using the whole data for training and testing
    test_size = training_size;
else
    test_size = floor(number_of_measurements * (1 - training_set_factor));
end

positions_train = positions(:, 1 : training_size);
magnetic_measurements_train = magnetic_measurements(:, 1 : training_size);
if(training_set_factor == 1)
    positions_test = positions_train;
    magnetic_measurements_test = magnetic_measurements_train;
else
    positions_test = positions(:, training_size+1: end);
    magnetic_measurements_test = magnetic_measurements(:, training_size+1:end);
end

%% Defining the Dirichlet boundary conditions of the eigendecomposition problem of the Gaussian Process
x_u = max(positions(1, :)) + space_margin; % Upper bound of the X-coordinates 
y_u = max(positions(2, :)) + space_margin; % Upper bound of the Y-coordinates
z_u = max(positions(3, :)) + space_margin; % Upper bound of the Z-coordinates

x_l = abs(min(positions(1, :))) + space_margin; % Lower bound of the X-coordinates (notice the abs operator)
y_l = abs(min(positions(2, :))) + space_margin; % Lower bound of the Y-coordinates (notice the abs operator)
z_l = abs(min(positions(3, :))) + space_margin; % Lower bound of the Z-coordinates (notice the abs operator)

boundaries = [max(x_u, x_l); 
              max(y_u, y_l); 
              max(z_u, z_l)]; % The boundaries of the eigendecomposition problem based on the actual boundaries of the data. 

%% Compute the exact gram matrix for the GP prior
fprintf('Calculating the exact gram matrix: \n');
tic
exact_gram_matrix = calculateExactGramMatrix(positions_train, positions_train, magnitude_scale_lin, magnitude_scale_SE, length_scale_SE);
toc
%% Compute the approximated gram matrix for the GP prior using the eigenfunctions of the laplacian operator for the scalar potential
fprintf('Calculating the Pot approximation: \n');
tic
[pot_eigenfunctions, pot_eigenvalues] = calculatePotBasisFunctionsAndValues(positions_train, number_of_basis_functions, boundaries, permutation_index, true);
pot_spectral_eig_values = computeSpectralEigValsMat(pot_eigenvalues, magnitude_scale_SE, length_scale_SE, magnitude_scale_lin, dimensions);
pot_approx_gram_mat = pot_eigenfunctions * pot_spectral_eig_values * pot_eigenfunctions';
toc

%% Compute the approximated gram matrix for the GP prior using the eigenfunctions of the laplacian operator for the Magnetic field
fprintf('Calculating the Mag approximation: \n');
tic
[mag_eigenfunctions, mag_eigenvalues] = calculateMagBasisFunctionsAndValues(positions_train, number_of_basis_functions, boundaries, permutation_index, true);
mag_spectral_eig_values = computeSpectralEigValsMat(mag_eigenvalues, magnitude_scale_SE, length_scale_SE, magnitude_scale_lin, dimensions);
mag_approx_gram_mat = mag_eigenfunctions * mag_spectral_eig_values * mag_eigenfunctions';
toc

%% Optimizing the log marginal likelihood function to get the optimal hyperparamters. (Not working properly yet)
% hyperparameters_optimization = HyperparametersOptimization(learning_rate);
% hyperparameters_optimization.initializeHyperparameters(length_scale_SE, magnitude_scale_SE, magnitude_scale_lin, measurement_noise);
% hyperparameters_optimization.setEigenfunctionsAndValues(mag_eigenfunctions, mag_eigenvalues, 3);
% hyperparameters_optimization.setMeasurementsVector(magnetic_measurements_train);
% hyperparameters_optimization.setPositionsVector(positions_train);
% [length_scale_SE, magnitude_scale_SE, magnitude_scale_lin, measurement_noise] = hyperparameters_optimization.optimizeHyperparameters(0.001, 1000);

%% Batch Estimation for the Magnetic Field using GP
fprintf('Calculating the Posterior of Magnetic field batch estimation problem: \n');
tic 
[mag_eigenfunctions_test, ~] = calculateMagBasisFunctionsAndValues(positions_test, number_of_basis_functions, boundaries, permutation_index, false);
[mean_mag, cov_mag] = batchEstimation(mag_eigenfunctions, mag_spectral_eig_values, mag_eigenfunctions_test, magnetic_measurements_train, measurement_noise);
toc

%% Plotting the results of the batch estimation against the actual measurements
figure; hold;
plot(magnetic_measurements_test(1, :)); % Plotting the actual measurements of the magnetic field in the X direction
plot(mean_mag(1, :)); % Plotting the estimated measurements of the magnetic field in the X direction
title('Plotting Magnetic field in X-direction');
xlabel('Timesteps');
ylabel('Magnetic Field Magnitude');
legend('Measurements', 'Estimated');

figure; hold;
plot(magnetic_measurements_test(2, :)); % Plotting the actual measurements of the magnetic field in the X direction
plot(mean_mag(2, :)); % Plotting the estimated measurements of the magnetic field in the X direction
title('Plotting Magnetic field in Y-direction');
xlabel('Timesteps');
ylabel('Magnetic Field Magnitude');
legend('Measurements', 'Estimated');

figure; hold;
plot(magnetic_measurements_test(3, :)); % Plotting the actual measurements of the magnetic field in the X direction
plot(mean_mag(3, :)); % Plotting the estimated measurements of the magnetic field in the X direction
title('Plotting Magnetic field in Z-direction');
xlabel('Timesteps');
ylabel('Magnetic Field Magnitude');
legend('Measurements', 'Estimated');

%% Sequential Estimation for Magnetic Field using GP
% For explanation of the variables names, please check the paper mentioned
% in the beginning of the file.
% This implementationc an be conisdered as a recursive least squares
% estimation of the magnetic field. 
% It is also completely equivalent to the update steps fo the Kalman
% filter.
sigma = mag_spectral_eig_values;
mu = zeros(number_of_basis_functions + dimensions, 1);

fprintf('Executing the Sequential Estimation \n');
tic
for i = 1 : training_size
   [current_eigenfunctions, ~] = calculateMagBasisFunctionsAndValues(positions_train(:, i), number_of_basis_functions, boundaries, permutation_index, false);
   S = current_eigenfunctions *  sigma * current_eigenfunctions' + measurement_noise^2 * eye(3);
   K = sigma * current_eigenfunctions' / S;
   mu = mu + K * (magnetic_measurements_train(:, i) - current_eigenfunctions * mu);
   sigma = sigma - K * S * K';   
   sigma = 1/2 * (sigma + sigma');
end
toc

fprintf('Estimating the Magnetic field using the estimated mean and covariance \n')
tic
predictions = zeros(dimensions, test_size);
for i = 1 : test_size
    [current_eigenfunctions, ~] = calculateMagBasisFunctionsAndValues(positions_test(:, i), number_of_basis_functions, boundaries, permutation_index, false);
    predictions(:, i) = current_eigenfunctions * mu;
    predictions_cov = current_eigenfunctions * sigma * current_eigenfunctions';
end
toc

%% Plotting the results of the sequential estimation against the actual measurements
figure; hold;
plot(magnetic_measurements_test(1, :)); % Plotting the actual measurements of the magnetic field in the X direction
plot(predictions(1, :)); % Plotting the estimated measurements of the magnetic field in the X direction
title('Plotting Magnetic field in X-direction');
xlabel('Timesteps');
ylabel('Magnetic Field Magnitude');
legend('Measurements', 'Estimated');

figure; hold;
plot(magnetic_measurements_test(2, :)); % Plotting the actual measurements of the magnetic field in the X direction
plot(predictions(2, :)); % Plotting the estimated measurements of the magnetic field in the X direction
title('Plotting Magnetic field in Y-direction');
xlabel('Timesteps');
ylabel('Magnetic Field Magnitude');
legend('Measurements', 'Estimated');

figure; hold;
plot(magnetic_measurements_test(3, :)); % Plotting the actual measurements of the magnetic field in the X direction
plot(predictions(3, :)); % Plotting the estimated measurements of the magnetic field in the X direction
title('Plotting Magnetic field in Z-direction');
xlabel('Timesteps');
ylabel('Magnetic Field Magnitude');
legend('Measurements', 'Estimated');

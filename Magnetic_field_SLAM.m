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

load('measurements_aligned_simple_trajectory.mat'); % Simple scenario measurements
% load('measurements_aligned_complicated_trajectory.mat'); % More complex scenario measurements

sampling_time = T;
magnetic_measurements = u(7:end, :);
positions = p_opti;
orientations = q_opti;
gravity = g;

downscale = 20;
positions = positions(:, 1 : downscale : end);
orientations = orientations(:, 1 : downscale : end);
number_of_measurements = size(positions, 2);
magnetic_measurements = magnetic_measurements(:, 1 : downscale : end);
sampling_time = sampling_time * downscale;

%% Initializing the hyperparameters of the Gaussian Process
magnitude_scale_SE = 1;
length_scale_SE = 0.3;
magnitude_scale_lin = 1;

space_margin = 10.0;
x_u = max(positions(1, :)) + space_margin; % Upper bound of the X-coordinates 
y_u = max(positions(2, :)) + space_margin; % Upper bound of the Y-coordinates
z_u = max(positions(3, :)) + space_margin; % Upper bound of the Z-coordinates

x_l = abs(min(positions(1, :))) + space_margin; % Lower bound of the X-coordinates
y_l = abs(min(positions(2, :))) + space_margin; % Lower bound of the Y-coordinates
z_l = abs(min(positions(3, :))) + space_margin; % Lower bound of the Z-coordinates

boundaries = [max(x_u, x_l); 
              max(y_u, y_l); 
              max(z_u, z_l)]; % The boundaries of the eigendecomposition problem based on the actual boundaries of the data. 
          
number_of_basis_functions = 500;

%% Compute the exact gram matrix for the GP prior
exact_gram_matrix = calculateExactGramMatrix(positions, positions, magnitude_scale_lin, magnitude_scale_SE, length_scale_SE);

%% Compute the approximated gram matrix for the GP prior using the eigenfunctions of the laplacian operator for the scalar potential
tic
[pot_eigenfunctions, pot_eigenvalues] = calculatePotBasisFunctionsAndValues(positions, number_of_basis_functions, boundaries);
pot_spectral_eig_values = computeSpectralEigValsMat(pot_eigenvalues, magnitude_scale_SE, length_scale_SE, magnitude_scale_lin, 3);
pot_approx_gram_mat = pot_eigenfunctions * pot_spectral_eig_values * pot_eigenfunctions';
toc

%% Compute the approximated gram matrix for the GP prior using the eigenfunctions of the laplacian operator for the Magnetic field
tic
[mag_eigenfunctions, mag_eigenvalues] = calculateMagBasisFunctionsAndValues(positions, number_of_basis_functions, boundaries);
mag_spectral_eig_values = computeSpectralEigValsMat(mag_eigenvalues, magnitude_scale_SE, length_scale_SE, magnitude_scale_lin, 3);
mag_approx_gram_mat = mag_eigenfunctions * mag_spectral_eig_values * mag_eigenfunctions';
toc
%% Batch Estimation of the Magnetic field

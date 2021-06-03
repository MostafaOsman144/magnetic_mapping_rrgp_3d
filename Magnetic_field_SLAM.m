%% Magnetic Field Mapping 
% Implementation using the reduced rank Gaussian process regression 
% described in the paper titled Modeling and Interpolation of the Ambient 
% Magnetic Field by Gaussian Processes
% 
% The magnetic field approach adopted is through using the gradient of
% the scalar potential

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

space_margin = 0.5;
x_u = max(positions(1, :)) + space_margin;
y_u = max(positions(2, :)) + space_margin;
z_u = max(positions(3, :)) + space_margin;

x_l = abs(min(positions(1, :))) + space_margin;
y_l = abs(min(positions(2, :))) + space_margin;
z_l = abs(min(positions(3, :))) + space_margin;

boundaries = [max(x_u, x_l); 
              max(y_u, y_l); 
              max(z_u, z_l)];
          
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

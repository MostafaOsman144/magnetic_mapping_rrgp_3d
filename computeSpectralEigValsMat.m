%% Calculating the diagonal matrix of eigenvalues
% Calulating the diagonal matrix of the spectral spectral density of the
% eigenvalues

function spectral_eigenvalues_mat = computeSpectralEigValsMat(eigenvalues, magnitude_scale_SE, length_scale, magnitude_scale_lin, input_dimension)
 spectral_eigvals = magnitude_scale_SE^2 * (2 * pi * length_scale^2)^(3/2) * exp(-(eigenvalues * length_scale^2) / 2);
 
 spectral_eigenvalues_part_mat = diag(spectral_eigvals);
 
 spectral_eigenvalues_mat = zeros(input_dimension + length(eigenvalues), input_dimension + length(eigenvalues));
 spectral_eigenvalues_mat(1:input_dimension, 1:input_dimension) = magnitude_scale_lin^2 * eye(input_dimension);
 spectral_eigenvalues_mat(input_dimension+1 : end, input_dimension+1 : end) = spectral_eigenvalues_part_mat;
end
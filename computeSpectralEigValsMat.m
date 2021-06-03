%% Calculating the diagonal matrix of eigenvalues
% Calulating the diagonal matrix of the spectral spectral density of the
% eigenvalues

function spectral_eigenvalues_mat = computeSpectralEigValsMat(eigenvalues_squared, magnitude_scale_SE, length_scale, magnitude_scale_lin, input_dimension)
number_of_eigenfunctions = size(eigenvalues_squared, 1);

spectral_eigvals = zeros(number_of_eigenfunctions + input_dimension, 1);
spectral_eigvals(1:input_dimension) = magnitude_scale_lin^2;
spectral_eigvals(input_dimension+1:end) = magnitude_scale_SE^2 * (2 * pi * length_scale^2)^(3/2) * exp(-(eigenvalues_squared * length_scale^2) / 2);
 
spectral_eigenvalues_mat = diag(spectral_eigvals);
end
%% %% Calculating the Magnetic field basis function as described in the function
% positions: size (d, n)
% space_upper_boundaries: the boundaries of the omega space which defines
% the eigendecomposition problem. size (d, 1)
% This function returns an (3n, m+3) matrix of basis functions evaluated at
% the specified positions.
% Notice that here the basis values returned by the function is the
% eigenvalues sqaured.
function [mag_basis_functions, basis_values]= calculateMagBasisFunctionsAndValues(positions, number_of_basis_functions, space_upper_boundaries)
positions = positions';

n = size(positions, 1);
m = number_of_basis_functions;
d = size(positions, 2);

permutation_index = generateIndexMat(m);

mag_basis_functions = zeros(3*n, d + m);
% TODO: Try to vectorize this for loop over n to optimize the computation
% time.
for i = 1 : n
   mag_basis_function_col = calculateGradient(permutation_index, positions(i, :)', space_upper_boundaries);
   mag_basis_functions(i*3 - 2 : i*3, :) = mag_basis_function_col;
end

basis_values = ones(m, 1);
for k = 1 : d
   basis_values =  basis_values .* (pi .* permutation_index(:, k) / (2 * space_upper_boundaries(k))).^2;
end
end
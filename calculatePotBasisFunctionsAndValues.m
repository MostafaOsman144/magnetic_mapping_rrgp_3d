%% Calculating the potential basis function as described in the function
% positions: size (d, n)
% space_upper_boundaries: the boundaries of the omega space which defines
% the eigendecomposition problem. size (d, 1)
% This function returns an (n, m+3) matrix of basis functions evaluated at
% the specified positions.
% Notice that here the basis values returned by the function is the
% eigenvalues sqaured.
function [pot_basis_functions, basis_values] = calculatePotBasisFunctionsAndValues(positions, number_of_basis_functions, space_upper_boundaries)
positions = positions';

n = size(positions, 1);
m = number_of_basis_functions;
d = size(positions, 2);

permutation_index = generateIndexMat(m);

pot_basis_functions = [];
for i = 1 : n
    pot_basis_function_col = ones(m, 1);
   for k = 1 : d
        sine_argument = pi .* permutation_index(:, k) .* (positions(i, k) + space_upper_boundaries(k)) / (2*space_upper_boundaries(k));
        basis_function_term = (1/(sqrt(space_upper_boundaries(k)))) * sin(sine_argument);
        pot_basis_function_col = pot_basis_function_col .* basis_function_term;
   end
   pot_basis_function_col = [positions(i, :)'; pot_basis_function_col];
   pot_basis_functions = [pot_basis_functions; pot_basis_function_col'];
end

basis_values = ones(m, 1);
for k = 1 : d
   basis_values =  basis_values .* (pi .* permutation_index(:, k) / (2 * space_upper_boundaries(k))).^2;
end
end
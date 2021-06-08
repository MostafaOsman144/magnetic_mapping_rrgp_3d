%% Calculating the potential basis function as described in the function
% positions: size (d, n)
% space_upper_boundaries: the boundaries of the omega space which defines
% the eigendecomposition problem. size (d, 1)
% This function returns an (n, m+3) matrix of basis functions evaluated at
% the specified positions.
% Notice that here the basis values returned by the function is the
% eigenvalues sqaured.
function [pot_basis_functions, basis_values] = calculatePotBasisFunctionsAndValues(positions, number_of_basis_functions, space_upper_boundaries, permutation_index, calculate_values)
positions = positions';

n = size(positions, 1);
m = number_of_basis_functions;
d = size(positions, 2);

pot_basis_functions = zeros(n, m+d);
% TODO: Make this snippet of code use pot-basis_function_row directly to
% avoid ambiguities in debugging the code due to the difference sizes and
% the many transposes. 

% TODO: Try to vectorize this for loop over n to optimize the computation
% time.
for i = 1 : n
    pot_basis_function_col = ones(m+3, 1);
    pot_basis_function_col(1 : 3, 1) = positions(i, :)';
   for k = 1 : d
        sine_argument = pi .* permutation_index(:, k) .* (positions(i, k) + space_upper_boundaries(k)) / (2*space_upper_boundaries(k));
        basis_function_term = (1/(sqrt(space_upper_boundaries(k)))) * sin(sine_argument);
        pot_basis_function_col(4:end, 1) = pot_basis_function_col(4:end, 1) .* basis_function_term;
   end
   pot_basis_functions(i, :)= pot_basis_function_col';
end

basis_values = ones(m, 1);
if(calculate_values)
    for k = 1 : d
        basis_values =  basis_values .* (pi .* permutation_index(:, k) / (2 * space_upper_boundaries(k))).^2;
    end
end

end
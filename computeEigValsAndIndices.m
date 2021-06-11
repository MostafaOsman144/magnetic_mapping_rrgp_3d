function [eigvals, indices] = computeEigValsAndIndices(permutation_list, number_of_basis_functions, space_upper_boundaries, d)

basis_values = zeros(size(permutation_list, 1), 1);
for k = 1 : d
    basis_values =  basis_values + (pi .* permutation_list(:, k) / (2 * space_upper_boundaries(k))).^2;
end

[sorted_eigenvalues, eigenvalues_indices] = sort(basis_values, 'ascend');
indices = permutation_list(eigenvalues_indices, :);

indices = indices(1:number_of_basis_functions, :);
eigvals = sorted_eigenvalues(1:number_of_basis_functions);

end
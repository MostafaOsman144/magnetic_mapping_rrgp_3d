%% Function for calculating the gradient of the basis functions
% as discussed in the paper

function grad = calculateGradient(permutation_index, position, upper_bound)

m = size(permutation_index, 1);
d = length(position);

grad_of_position = eye(d);

grad = zeros(d, m+d);
% TODO: Make this snippet of code use mag_basis_function_row directly to
% avoid ambiguities in debugging the code due to the difference sizes and
% the many transposes. 
for grad_position = 1 : d
    grad_col = ones(m, 1);
    for i = 1 : d
        if i == grad_position
            argument = pi .* permutation_index(:, i) .* (position(i) + upper_bound(i)) ./ (2*upper_bound(i));
            grad_term = (pi .* permutation_index(:, i) ./ (2 * upper_bound(i)^(3/2))) .* cos(argument);
            grad_col = grad_col .* grad_term;
        else
            argument = pi .* permutation_index(:, i) .* (position(i) + upper_bound(i)) ./ (2*upper_bound(i));
            const_term = (1/(sqrt(upper_bound(i)))) .* sin(argument);
            grad_col = grad_col .* const_term;
        end
    end
    grad(grad_position, d+1:end) = grad_col';

end


grad(:, 1:d) = grad_of_position;

end
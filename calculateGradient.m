%% Function for calculating the gradient of the basis functions
% as discussed in the paper

function grad = calculateGradient(permutation_index, position, upper_bound)

m = size(permutation_index, 1);
d = length(position);

grad_of_position = eye(d);

grad = [];
for grad_position = 1 : d
    for i = 1 : d
        grad_col = ones(m, 1);
        if i == grad_position
            argument = pi .* permutation_index(:, i) .* (position(i) + upper_bound(i)) / (2*upper_bound(i));
            grad_term = pi .* permutation_index(:, i) / (2 * upper_bound(i)^(3/2)) .* cos(argument);
            grad_col = grad_col .* grad_term;
        else
            argument = pi .* permutation_index(:, i) .* (position(i) + upper_bound(i)) / (2*upper_bound(i));
            const_term = (1/(sqrt(upper_bound(i)))) * sin(argument);
            grad_col = grad_col .* const_term;
        end
    end
    grad = [grad; grad_col'];
end

grad = [grad_of_position, grad];
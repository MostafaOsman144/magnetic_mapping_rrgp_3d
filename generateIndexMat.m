%% Generating the index set of permutations for computing the eigenfunctions and the eigenvalues
% This function generates a set of permutations to be used for calculating
% the eigenfunctions and eigenvalues of the kernel for reduced-rank
% Gaussian process regression based on the methodology proposed in the
% paper "Modeling and interpolation of the ambient magnetic field by
% Gaussian Process".
% Notice that this function only work for 3D vectors. It needs to be
% generalized.

function index_mat = generateIndexMat(basis_number)

% index_mat = zeros(basis_number, 3);
% max_test_val = 20;
% 
% n = 1;
% i = 1;
% j = 1;
% k = 1;
% while n <= basis_number
%     index_mat(n, :) = [i, j, k];
%     n = n + 1;
%     
%     k = k + 1;
%     if(k > max_test_val)
%         k = 1;
%         j = j + 1;
%         if(j > max_test_val)
%             j = 1;
%             i = i + 1;
%         end
%     end
% end

index_mat = [];
max_test_val = 20;
for i = 1 : max_test_val
    for j = 1 : max_test_val
        for k = 1 : max_test_val
        index_mat = [index_mat; i, j, k];    
        end
    end
end

end
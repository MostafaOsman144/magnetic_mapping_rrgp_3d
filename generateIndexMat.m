%% Generating the index set of permutations for computing the eigenfunctions and the eigenvalues
% This function generates a set of permutations to be used for calculating
% the eigenfunctions and eigenvalues of the kernel for reduced-rank
% Gaussian process regression based on the methodology proposed in the
% paper "Modeling and interpolation of the ambient magnetic field by
% Gaussian Process".
% Notice that this function only work for 3D vectors. It needs to be
% generalized.

function index_mat = generateIndexMat(basis_number)

index_mat = [];

for i = 1 : basis_number
    for j = 1 : basis_number
        for k = 1 : basis_number
            index_mat = [index_mat; i, j, k];
            if size(index_mat, 1) == basis_number
                break;
            end
        end
        if size(index_mat, 1) == basis_number
            break;
        end
    end
    if size(index_mat, 1) == basis_number
        break;
    end
end

end
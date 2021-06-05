%% This class is responsible for determining the optimal hyperparameters for the batch estimation GP. 
% The hyperparameters are determined through solving an optimization
% problem which aims at maximizing the log marginal likelihood.
% The optimization problem is solved through using gradient descent. 
% The gradient descent is implemented from scratch in this class.
%
% L: The loss function (log marginal likelihood)
% dLdl: the partial derivative of the loss function with respect to the
% length scale.
% dLdsSE: the partial derivative of the loss function with respect to the
% magnitude_scale of the SE kernel.
% dLdsLin: the partial derivative of the loss function with respect to the
% magnitude_scale of the linear kernel.
% dLdsn: the partial derivative of the loss function with respect to the
% variance of the measurements noise. 
%
% The class will compute the marginal likelihood through using the
% reduced rank approximation defined in "Hilbert Space Methods for
% Reduced_Rank Gaussian Process Regression" by Arno Solin and Simo
% Sarkka. 
% Please see this paper for more information on the variables naming. 

classdef HyperparametersOptimization
    properties
        % Defining the loss function and its derivatives
        L = Inf;
        dLdl;
        dLdsSE;
        dLdsLin;
        dLdsn;
        
        % Defining the hyperparameters.
        length_scale_SE = 0;
        magnitude_scale_SE = 0;
        magnitude_scale_lin = 0;
        measurement_noise_variance = 0;
        
        % Defining the learning rate for the gradient descent
        learning_rate = 0.1;
        
        % Defining the eigenfunctions and eigenvalues matrices
        eigenfunctions;
        eigenvalues;
        
        % Defining the size of the problem's vectors
        n; % size of the training set
        m; % number of eigenfunctions 
        d; % number of dimensions of input (positions)
        
        % Defining the measurements vector vec_y (3n, 1)
        measurements;
    end
    
    methods 
        function init_obj = HyperparametersOptimization(learning_rate)
            init_obj.learning_rate = learning_rate;
            init_obj.length_scale_SE = length_scale_se;
            init_obj.magnitude_scale_SE = magnitude_scale_se;
            init_obj.magnitude_scale_lin = magnitude_scale_lin;
            init_obj.measurement_noise_variance = measurement_noise_variance;
            
        end
        
        function obj = initializeHyperparameters(obj, length_scale_se, magnitude_scale_se, magnitude_scale_lin, noise_variance)
            obj.length_scale_SE = length_scale_se;
            obj.magnitude_scale_SE = magnitude_scale_se;
            obj.magnitude_scale_lin = magnitude_scale_lin;
            obj.measurement_noise_variance = noise_variance;
        end
        
        % This function initializes the hyperparameters to random values
        % taken from a Gaussian distribution defined by the mean and
        % standard deviation defined in the modes vectors. 
        % length_scale_modes: [mean; std] for length_scales
        % magnitude_scale_modes: [mean; std] for magntiude_scales
        % noise_variance_modes: [mean; std] for the measurements noise
        % variance
        function obj = initializeHyperparametersRandomly(obj, length_scale_modes, magnitude_scale_modes, noise_variance_modes)
           obj.length_scale_SE = length_scale_modes(1) + length_scale_modes(2) * randn;
           obj.magnitude_scale_SE = magnitude_scale_modes(1) + magnitude_scale_modes(2) * randn;
           obj.magnitude_scale_lin = magntiude_scale_modes(1) + magnitude_scale_modes(2) * randn;
           obj.measurement_noise_variance = noise_variance_modes(1) + noise_variance_modes(2) * randn;
        end
        
        function obj = setEigenfunctionsAndValues(obj, eigenfunctions, eigenvalues, dimensions)
           obj.eigenfunctions = eigenfunctions;
           obj.eigenvalues = eigenvalues;
           obj.n = size(eigenfunctions, 1); % This value is d * actual number of positions.
           obj.m = size(eigenvalues, 1);
           obj.d = dimensions;
           
        end
        
        function obj = setMeasurementsVector(obj, measurements)
            obj.measurements = reshape(measurements, size(measurements, 1) * size(measurements, 2), 1);
        end
        
        
        % See the paper mentioned in the top of the file for more
        % information.
        function obj = computeLossFunction(obj, spectral_eigenvalues, Z)
            log_Q_tilde = (obj.n - obj.m) * log(obj.measurement_noise_variance) + log(det(Z)) + trace(log(spectral_eigenvalues));
            
            vec_y_Q_tilde_vec_y = (1/obj.measurement_noise_variance) * (obj.measurements' * obj.measurements - obj.measurements' * obj.eigenfunctions / Z * obj.eigenfunctions' * obj.measurements );
            
            obj.L = 0.5 * log_Q_tilde + 0.5 * vec_y_Q_tilde_vec_y + (obj.n/2)*log(2*pi);
        end
        
        function dSdl = partialEigenPartialLengthScaleSE(obj)
            l_se = obj.length_scale_SE;
            s_se = obj.magnitude_scale_SE;
            dSdl = s_se^2 * (3/2) * (2*pi*l_se^2)^(1/2) * exp(-(obj.eigenvalues * l_se^2)/2) * (4*pi*l_se) + s_se^2 * (2*pi*l_se^2)^(3/2) * exp(-(obj.eigenvalues * l_se^2)/2) * (-obj.eigenvalues * l_se);
            
            dSdl = diag(dSdl);
            dSdl(1:obj.d, 1:obj.d) = zeros(obj.d, obj.d);
        end
        
        function dSds = partialEigenPartialMagnitudeScaleSE(obj)
            l_se = obj.length_scale_SE;
            s_se = obj.magnitude_scale_SE;
            dSds = 2 * s_se * (2*pi*l_se^2)^(3/2) * exp(-(obj.eigenvalues * l_se^2)/2);
            
            dSds = diag(dSds);
            dSds(1:obj.d, 1:obj.d) = zeros(obj.d, obj.d);
        end
        
        function dSdlin = partialEigenPartialMagnitudeScaleLin(obj)
           dSdlin = zeros(obj.m+obj.d, obj.m+obj.d);
           dSdlin(1:obj.d, 1:obj.d) = 2 * obj.magnitude_scale_lin;
        end
        
        function obj = computePartials(obj, spectral_eigenvalues, Z)
            dLambda_dl = obj.partialEigenPartialLengthScaleSE();
            dLambda_ds = obj.partialEigenPartialMagnitudeScaleSE();
            dLambda_dlin = obj.partialEigenPartialMagnitudeScaleLin();
            
            d_logQtilde_dl = trace(spectral_eigenvalues(obj.d+1:end, obj.d+1:end) \ dLambda_dl(obj.d+1:end, obj.d+1:end)) - obj.measurement_noise_variance^2 * trace(Z \ spectral_eigenvalues^2 \ dLambda_dl);
            d_logQtilde_ds = trace(spectral_eigenvalues(obj.d+1:end, obj.d+1:end) \ dLambda_ds(obj.d+1:end, obj.d+1:end)) - obj.measurement_noise_variance^2 * trace(Z \ spectral_eigenvalues^2 \ dLambda_ds);
            d_logQtilde_dlin = trace(spectral_eigenvalues(obj.d+1:end, obj.d+1:end) \ dLambda_dlin(obj.d+1:end, obj.d+1:end)) - obj.measurement_noise_variance^2 * trace(Z \ spectral_eigenvalues^2 \ dLambda_dlin);
            
            d_logQtilde_dn = ((obj.n - obj.m) / obj.measurement_noise_variance^2) + trace(Z \ inv(spectral_eigenvalues));
            
            d_yQtildey_dl = - obj.measurements' * obj.eigenfunctions / Z * (spectral_eigenvalues^2 \ dLambda_dl) / Z * obj.eigenfunctions' * obj.measurements;
            d_yQtildey_ds = - obj.measurements' * obj.eigenfunctions / Z * (spectral_eigenvalues^2 \ dLambda_ds) / Z * obj.eigenfunctions' * obj.measurements;
            d_yQtildey_dlin = - obj.measurements' * obj.eigenfunctions / Z * (spectral_eigenvalues^2 \ dLambda_dlin) / Z * obj.eigenfunctions' * obj.measurements;
            
            d_yQtildey_dn = (1/obj.measurement_noise_variance^2) * obj.measurements' * obj.eigenfunctions / Z / spectral_eigenvalues / Z * obj.eigenfunctions' * obj.measurements - (1/obj.measurement_noise_variance^4) * obj.measurements' * obj.measurements;
            
            obj.dLdl = 0.5 * d_logQtilde_dl + 0.5 * d_yQtildey_dl;
            obj.dLdsSE = 0.5 * d_logQtilde_ds + 0.5 * d_yQtildey_ds;
            obj.dLdsLin = 0.5 * d_logQtilde_dlin + 0.5 * d_yQtildey_dlin;
            obj.dLdsn = 0.5 * d_logQtilde_dn + 0.5 * d_yQtildey_dn;
        end
        
        % See the paper mentioned in the top of the file for more
        % information.
        function Q_tilde = computeQtilde(obj, spectral_eigvalues)
            Q_tilde = obj.eigenfunctions * spectral_eigvalues * obj.eigenfunctions' + obj.measurement_noise_variance * eye(obj.n);
        end
        
        % See the paper mentioned in the top of the file for more
        % information.
        function Z = computeZ(obj, spectral_eigvalues)
            Z = obj.measurement_noise_variance * eye(obj.m) / spectral_eigvalues + obj.eigenfunctions' * obj.eigenfunctions; 
        end
    end
end
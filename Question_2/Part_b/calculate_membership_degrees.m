 % File: calculate_membership_degrees.m
function [active_mf_indices, active_mf_values] = calculate_membership_degrees(crisp_input, cores_vec)
    n_cores = length(cores_vec);
    active_mf_indices = [];
    active_mf_values = [];

    % Clip input to the universe of discourse
    crisp_input = max(cores_vec(1), min(cores_vec(end), crisp_input));

    % If crisp_input is exactly on a core
    for k_core = 1:n_cores
        if abs(crisp_input - cores_vec(k_core)) < 1e-9 % Check if on a core
            active_mf_indices = [k_core];
            active_mf_values = [1.0];
            return;
        end
    end

    % Find which two MFs are active
    for i = 1:(n_cores - 1)
        if crisp_input > cores_vec(i) && crisp_input < cores_vec(i+1)
            % Input is between core i and core i+1
            % MF_i: (cores_vec(i+1) - crisp_input) / (cores_vec(i+1) - cores_vec(i))
            % MF_{i+1}: (crisp_input - cores_vec(i)) / (cores_vec(i+1) - cores_vec(i))
            
            val_i = (cores_vec(i+1) - crisp_input) / (cores_vec(i+1) - cores_vec(i));
            val_i_plus_1 = (crisp_input - cores_vec(i)) / (cores_vec(i+1) - cores_vec(i));
            
            active_mf_indices = [i, i+1];
            active_mf_values = [val_i, val_i_plus_1];
            return; 
        end
    end
end
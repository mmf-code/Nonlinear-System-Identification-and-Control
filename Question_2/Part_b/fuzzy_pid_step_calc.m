% File: fuzzy_pid_step_calc.m
function [u_cn, u_FCn] = fuzzy_pid_step_calc(r_n, y_n_minus_1, e_trn_prev_val, u_FC_prev_val, Ke, Kde, alpha_param, beta_param, h_sampling)
    % 1. Calculate e_trn and delta_e_trn
    e_trn_n = r_n - y_n_minus_1;
    delta_e_trn_n = (e_trn_n - e_trn_prev_val) / h_sampling;

    % 2. Scale errors and clip
    en = Ke * e_trn_n;
    en_dot = Kde * delta_e_trn_n;
    en = max(-1, min(1, en));
    en_dot = max(-1, min(1, en_dot));

    % 3. Fuzzification
    cores = [-1, -0.4, 0, 0.4, 1];
    [active_indices_e, active_mf_values_e] = calculate_membership_degrees(en, cores);
    [active_indices_edot, active_mf_values_edot] = calculate_membership_degrees(en_dot, cores);

    % 4. Fuzzy Inference (Equation 5)
    u_FCn = 0;
    rule_table_outputs = get_rule_table();

    for i_loop = 1:length(active_indices_e)
        idx_e = active_indices_e(i_loop);
        val_A = active_mf_values_e(i_loop);
        for j_loop = 1:length(active_indices_edot)
            idx_edot = active_indices_edot(j_loop);
            val_B = active_mf_values_edot(j_loop);
            
            u_rule = rule_table_outputs(idx_e, idx_edot);
            u_FCn = u_FCn + val_A * val_B * u_rule;
        end
    end
    
    % 5. Calculate u_cn (Equation 3)
    u_cn = alpha_param * u_FCn + beta_param * (u_FCn + u_FC_prev_val);
end
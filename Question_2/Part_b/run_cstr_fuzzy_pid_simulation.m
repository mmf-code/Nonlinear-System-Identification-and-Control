% File: run_cstr_fuzzy_pid_simulation.m
clear; clc; close all;

% --- 1. Initialization ---
% CSTR Parameters (Table 2 from assignment)
Da1 = 3.0; Da2 = 0.5; Da3 = 1.0; d2 = 1.0;
h = 0.1; % Sampling time

x1_init = 0.31;
x2_init = 0.71;
x3_init = 0.5; % Plant output initial value (y_initial)

% Fuzzy PID Controller Hyperparameters (Table 3 from assignment)
Ke = 0.15;
Kde = 1.5;
alpha_fpid = 0.7; 
beta_fpid = 1.2;

% Simulation Setup
t_end = 45; % End time from Figure 3
time_vec = 0:h:t_end-h; % Time vector for simulation
N_steps = length(time_vec);

% Reference Signal (Construct based on Figure 3 from assignment)
r_vec = zeros(1, N_steps);
% t_points for changes: 0, 10, 20, 30, 40, 45
% r_values at these intervals: 0.4, 0.5, 0.58, 0.5, 0.4
for i = 1:N_steps
    t_current = time_vec(i);
    if t_current < 10
        r_vec(i) = 0.4;
    elseif t_current < 20
        r_vec(i) = 0.5;
    elseif t_current < 30
        r_vec(i) = 0.58;
    elseif t_current < 40
        r_vec(i) = 0.5;
    else % t_current >= 40
        r_vec(i) = 0.4;
    end
end


% Initialize Plant States (current values for the loop)
x1_current = x1_init;
x2_current = x2_init;
x3_current = x3_init; % This is also y(0) or y_initial

% Initialize Controller States
e_trn_prev = 0;         % e_trn(n-1) for derivative calculation
u_FC_prev = 0;          % u_FC(n-1) from FLC output
% Initial u_FPID. A common choice for u_FPID(0) is based on r(0)
% or the initial control needed. Let's start with a value that might keep
% the output near r(0) or x3_init.
% From the problem, u_FPID(n+1) = u_FPID(n) + u_cn.
% So u_FPID(n) is the control input applied at step n.
u_FPID_current = x3_init; % Control input u_FPID that will be applied at current step n

% History Arrays to store data for plotting
y_history = zeros(1, N_steps);
u_FPID_history = zeros(1, N_steps);
e_plot_history = zeros(1, N_steps); % For r(n) - y(n)

% Store initial values in history
y_history(1) = x3_current; % y(0)
u_FPID_history(1) = u_FPID_current; % u_FPID applied at t=0
e_plot_history(1) = r_vec(1) - y_history(1);
% For the first step (n=1), y_n_minus_1 will be x3_init.
% e_trn_prev is 0. u_FC_prev is 0.

% --- 2. Main Simulation Loop ---
fprintf('Starting CSTR simulation with Fuzzy PID...\n');
for n = 1:N_steps
    r_n_current = r_vec(n); % Reference r(n) for current time step
    
    % Output from previous step y(n-1) needed for error calculation
    if n == 1
        y_n_minus_1 = x3_init;
    else
        y_n_minus_1 = y_history(n-1); % Output from the actual plant at step n-1
    end

    % Call fuzzy_pid_step_calc to get u_cn and u_FCn for current step
    [u_cn_n, u_FC_n_current] = fuzzy_pid_step_calc(r_n_current, y_n_minus_1, ...
                                           e_trn_prev, u_FC_prev, ...
                                           Ke, Kde, alpha_fpid, beta_fpid, h);
    
    % Update overall fuzzy PID control signal for *this* step (Equation 4)
    % u_FPID(n) = u_FPID(n-1) + u_cn(n-1) if u_cn comes from previous error.
    % Or u_FPID(n) = u_FPID(n-1) + u_cn(n) if u_cn is calculated based on current error to determine current u_FPID.
    % The problem states u_FPID(n+1) = u_FPID(n) + u_cn.
    % This means u_FPID to be applied *now* is u_FPID_current + u_cn_n.
    % Let's adjust: u_FPID_to_apply = u_FPID_prev_applied + u_cn_n.
    if n == 1
        u_FPID_prev_applied = u_FPID_current; % Initial control effort
    else
        u_FPID_prev_applied = u_FPID_history(n-1);
    end
    u_FPID_to_apply = u_FPID_prev_applied + u_cn_n;
    
    % Apply control input v1 = u_FPID_to_apply to CSTR Plant
    % The function CSTR_runga_kutta_new updates x1, x2, x3 internally
    % So we pass current states and get back next states.
    [x1_next, x2_next, x3_next] = CSTR_runga_kutta_new(x1_current, x2_current, x3_current, ...
                                                      Da1, Da2, Da3, d2, u_FPID_to_apply, h);
    
    y_n_actual = x3_next; % Plant output y(n) after applying u_FPID_to_apply

    % Store History for plotting
    y_history(n) = y_n_actual;
    u_FPID_history(n) = u_FPID_to_apply;
    e_plot_history(n) = r_n_current - y_n_actual;

    % Update States for Next Iteration
    e_trn_prev = r_n_current - y_n_minus_1; % Error based on y(n-1) used for this step's D-term calc
    u_FC_prev = u_FC_n_current;             % FLC output from this step becomes previous for next
    % Update current plant states for the next loop iteration
    x1_current = x1_next;
    x2_current = x2_next;
    x3_current = x3_next;
end
fprintf('Simulation finished.\n');

% --- 3. Plotting Results ---
figure;
subplot(2,1,1);
plot(time_vec, r_vec, 'k--', 'LineWidth', 1.5, 'DisplayName', 'Reference r(n)');
hold on;
plot(time_vec, y_history, 'b-', 'LineWidth', 1.5, 'DisplayName', 'System Output y(n) (x_3)');
hold off;
xlabel('Time (s)');
ylabel('Output x_3');
title('CSTR Control with PID-Type Fuzzy Controller');
legend('show', 'Location', 'best');
grid on;
ylim_vals = ylim; % Get current y-axis limits
padding = 0.1 * (ylim_vals(2)-ylim_vals(1));
ylim([ylim_vals(1)-padding, ylim_vals(2)+padding]);


subplot(2,1,2);
plot(time_vec, u_FPID_history, 'r-', 'LineWidth', 1.5, 'DisplayName', 'Control Signal u_{FPID}(n)');
xlabel('Time (s)');
ylabel('Control Input v_1');
legend('show', 'Location', 'best');
grid on;
ylim_vals = ylim;
padding = 0.1 * (ylim_vals(2)-ylim_vals(1));
ylim([ylim_vals(1)-padding, ylim_vals(2)+padding]);

sgtitle('Fuzzy PID Control of CSTR System');
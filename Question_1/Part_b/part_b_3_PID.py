import sys
import pathlib
import importlib.util
import numpy as np
import matplotlib.pyplot as plt


STUDENT_ID = 35

# --- Load System2 Class ---
def load_system_class_definition(system_name="System2"):
    pyc_filename = f"{system_name}.cpython-38.pyc"
    pyc_path = pathlib.Path("./python") / pyc_filename 
    if not pyc_path.exists():
        print(f"Error: {pyc_filename} not found at {pyc_path}")
        sys.exit()
    spec = importlib.util.spec_from_file_location(system_name.lower(), pyc_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, system_name)

System2_class = load_system_class_definition("System2")
s = System2_class(STUDENT_ID) # Plant

# --- Controller & Simulation Parameters ---
print("--- Q1.b.3: Neuron-Based Nonlinear PID Controller (Improved Tuning Attempt) ---")

ETA = 0.005  # Learning rate for weight updates

# Nonlinear Activation Function (scaled tanh)
ACTIVATION_AMP = 1.0 # Max/min output of activation if input is large
def nonlinear_activation(value): # B is effectively merged into GAIN_X
    return ACTIVATION_AMP * np.tanh(value)

# Gains for P, I, D terms (applied BEFORE activation)
GAIN_P = 1.2
GAIN_I = 0.3 
GAIN_D = 0.8

# Initial weights for the "neurons" (P, I, D components)
w_p_init = 0.2
w_i_init = 0.1 
w_d_init = 0.1

# Control signal limits (act as saturation for controller output)
U_MIN = -1.5 # Assumming System2 input is somewhat bounded
U_MAX = 1.5  

# Simulation settings
N_SIM_STEPS = 700  # Increased simulation time to see more adaptation
T_STEPS = np.arange(N_SIM_STEPS)

# Reference Signal r(n) - more varied
reference_signal = np.zeros(N_SIM_STEPS)
reference_signal[50:200] = 1.0
reference_signal[250:400] = -0.5 # Added a negative reference
reference_signal[450:600] = 1.2
reference_signal[600:N_SIM_STEPS] = 0.2

# History tracking
y_history = np.zeros(N_SIM_STEPS)
u_history = np.zeros(N_SIM_STEPS)
e_update_history = np.zeros(N_SIM_STEPS) # Stores e_actual_for_update = r(n) - y(n)

w_p_history = np.zeros(N_SIM_STEPS)
w_i_history = np.zeros(N_SIM_STEPS)
w_d_history = np.zeros(N_SIM_STEPS)

# Initialize controller state
s.reset() # Reset plant state
y_prev = 0.0  # y(n-1), initial output of the system is assumed 0
e_prev = 0.0  # e(n-1)
I_sum = 0.0   # Integral sum

w_p, w_i, w_d = w_p_init, w_i_init, w_d_init

# --- Simulation Loop ---
print(f"Starting simulation with ETA={ETA}, GAIN_P={GAIN_P}, GAIN_I={GAIN_I}, GAIN_D={GAIN_D}")
for n in range(N_SIM_STEPS):
    r_n = reference_signal[n] # Current reference r(n)

    # 1. Calculate error for PID terms (based on y_prev = y(n-1))
    e_n_control = r_n - y_prev 

    # 2. Create PID terms
    P_term = e_n_control
    
    # Integral term with basic anti-windup logic
    # I_sum is only updated if control signal is not saturated or error is reducing saturation
    # I_term uses previous accumulation (I_sum from n-1)
    
    # D_term
    D_term = e_n_control - e_prev

    # 3. Applying Nonlinear Activation to scaled PID terms
    x_p = nonlinear_activation(GAIN_P * P_term)
    x_i = nonlinear_activation(GAIN_I * I_sum) # I_sum is from previous step's accumulation
    x_d = nonlinear_activation(GAIN_D * D_term)

    # 4. Calculate Tentative Control Signal u_tentative(n)
    u_tentative_n = w_p * x_p + w_i * x_i + w_d * x_d
    
    # 5. Apply Saturation to get Actual Control Signal u(n)
    u_n = np.clip(u_tentative_n, U_MIN, U_MAX)

    # 6. Update Integral Sum (I_sum for NEXT iteration) with anti-windup
    # If u_n was saturated and e_n_control would make saturation worse
    if (u_tentative_n >= U_MAX and e_n_control > 0) or \
       (u_tentative_n <= U_MIN and e_n_control < 0):
        pass # Do not update I_sum (Hold integral)
    else:
        I_sum += e_n_control # This I_sum will be used for I_term in iteration n+1


    # 7. Applying Control Signal to Plant to get y(n)
    y_n = s.output(u_n)

    # 8. Calculate Actual Error for Weight Update: e_update(n) = r(n) - y(n)
    e_n_update = r_n - y_n
    
    # 9. Update Weights (Online Learning)
    w_p += ETA * e_n_update * x_p
    w_i += ETA * e_n_update * x_i
    w_d += ETA * e_n_update * x_d

    # 10. Store History
    y_history[n] = y_n
    u_history[n] = u_n
    e_update_history[n] = e_n_update 
    w_p_history[n] = w_p
    w_i_history[n] = w_i
    w_d_history[n] = w_d

    # 11. Updating states for next iteration
    y_prev = y_n
    e_prev = e_n_control 

print("Simulation finished.")

# --- Plotting Results ---
plt.figure(figsize=(12, 10))
plt.suptitle(f'Adaptive Neuron PID: Eta={ETA}, Gp={GAIN_P}, Gi={GAIN_I}, Gd={GAIN_D}', fontsize=14)

plt.subplot(4, 1, 1)
plt.plot(T_STEPS, reference_signal, 'k--', linewidth=1.5, label='Reference r(n)')
plt.plot(T_STEPS, y_history, 'b-', linewidth=1.5, label='System Output y(n)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

plt.subplot(4, 1, 2)
plt.plot(T_STEPS, u_history, 'r-', label='Control Signal u(n)')
plt.plot(T_STEPS, np.full_like(T_STEPS, U_MAX), 'r:', linewidth=1, label='U_max')
plt.plot(T_STEPS, np.full_like(T_STEPS, U_MIN), 'r:', linewidth=1, label='U_min')
plt.ylabel('Control Input')
plt.legend()
plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(T_STEPS, e_update_history, 'g-', label='Error e(n) = r(n)-y(n)')
plt.ylabel('Error')
plt.legend()
plt.grid(True)

plt.subplot(4, 1, 4)
plt.plot(T_STEPS, w_p_history, label=f'Weight w_p(n) (final: {w_p:.2f})')
plt.plot(T_STEPS, w_i_history, label=f'Weight w_i(n) (final: {w_i:.2f})')
plt.plot(T_STEPS, w_d_history, label=f'Weight w_d(n) (final: {w_d:.2f})')
plt.ylabel('Weights')
plt.xlabel('Time step n')
plt.legend()
plt.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjusting rect to make space for suptitle
# --- Save the plot as PNG ---
output_dir = pathlib.Path("Question_1/Part_b")
output_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(output_dir / "q1b3_adaptive_neuron_pid.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved: q1b3_adaptive_neuron_pid.png")


# Calculate some performance metrics
mse_tracking = np.mean(e_update_history**2)
print(f"\nTracking Performance:")
print(f"  Mean Squared Error (r(n)-y(n)): {mse_tracking:.4f}")
final_abs_error_avg = np.mean(np.abs(e_update_history[-50:])) # Avg abs error in last 50 steps
print(f"  Avg absolute error in last 50 steps: {final_abs_error_avg:.4f}")
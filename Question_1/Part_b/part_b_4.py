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
# ... (same load_system_class_definition function as before) ...
System2_class = load_system_class_definition("System2")
s = System2_class(STUDENT_ID) # Plant instance

# --- Classic PID Controller & Simulation Parameters ---
print("--- Q1.b.4: Classic PID Controller ---")

# PID Gains
KP = 0.8   # Proportional Gain
KI = 0.1   # Integral Gain 
KD = 0.2   # Derivative Gain

# Control signal limits
U_MIN = -1.5 
U_MAX = 1.5  

# Simulation settings (same as b.3 for comparison)
N_SIM_STEPS = 700  
T_STEPS = np.arange(N_SIM_STEPS)
reference_signal = np.zeros(N_SIM_STEPS)
reference_signal[50:200] = 1.0
reference_signal[250:400] = -0.5 
reference_signal[400:600] = 1.2
reference_signal[600:N_SIM_STEPS] = 0.2

# History tracking
y_history_pid = np.zeros(N_SIM_STEPS)
u_history_pid = np.zeros(N_SIM_STEPS)
e_history_pid = np.zeros(N_SIM_STEPS) # Stores e_actual = r(n) - y(n)

# Initial conditions for controller state
s.reset() 
y_prev = 0.0  
e_prev = 0.0  
I_sum = 0.0   

# --- Simulation Loop ---
print(f"Starting simulation with KP={KP}, KI={KI}, KD={KD}")
for n in range(N_SIM_STEPS):
    r_n = reference_signal[n] 

    e_n_control = r_n - y_prev 

    P_term = e_n_control

    
    # D_term
    D_term = e_n_control - e_prev

    # Calculate Control Signal u(n)
    # Note: P_term, I_sum (as I_term), D_term are used directly (linear)
    u_tentative_n = KP * P_term + KI * I_sum + KD * D_term # I_sum is from previous accumulations
    
    u_n = np.clip(u_tentative_n, U_MIN, U_MAX)

    # Update Integral Sum (I_sum for NEXT iteration) with anti-windup
    if (u_tentative_n >= U_MAX and e_n_control > 0) or \
       (u_tentative_n <= U_MIN and e_n_control < 0):
        pass # Hold integral
    else:
        I_sum += e_n_control 

    y_n = s.output(u_n)
    e_n_actual = r_n - y_n
    
    y_history_pid[n] = y_n
    u_history_pid[n] = u_n
    e_history_pid[n] = e_n_actual

    y_prev = y_n
    e_prev = e_n_control 

print("Simulation finished.")

# --- Plotting Results ---
plt.figure(figsize=(12, 8))
plt.suptitle(f'Classic PID Control: Kp={KP}, Ki={KI}, Kd={KD}', fontsize=14)

plt.subplot(3, 1, 1)
plt.plot(T_STEPS, reference_signal, 'k--', linewidth=1.5, label='Reference r(n)')
plt.plot(T_STEPS, y_history_pid, 'b-', linewidth=1.5, label='System Output y(n)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(T_STEPS, u_history_pid, 'r-', label='Control Signal u(n)')
plt.plot(T_STEPS, np.full_like(T_STEPS, U_MAX), 'r:', linewidth=1, label='U_max')
plt.plot(T_STEPS, np.full_like(T_STEPS, U_MIN), 'r:', linewidth=1, label='U_min')
plt.ylabel('Control Input')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(T_STEPS, e_history_pid, 'g-', label='Error e(n) = r(n)-y(n)')
plt.ylabel('Error')
plt.xlabel('Time step n')
plt.legend()
plt.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96]) 

# --- Save the figure ---
output_dir = pathlib.Path("Question_1/Part_b")
output_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(output_dir / "q1b4_classic_pid.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved: q1b4_classic_pid.png")


mse_tracking_pid = np.mean(e_history_pid**2)
print(f"\nTracking Performance (Classic PID):")
print(f"  Mean Squared Error (r(n)-y(n)): {mse_tracking_pid:.4f}")
final_abs_error_avg_pid = np.mean(np.abs(e_history_pid[-50:])) 
print(f"  Avg absolute error in last 50 steps: {final_abs_error_avg_pid:.4f}")
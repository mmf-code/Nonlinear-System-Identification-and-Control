import sys
import pathlib
import importlib.util
import numpy as np
import torch
import torch.nn as nn
# Note: I am not using optim here as we are not training the NN, but using it.
from sklearn.preprocessing import StandardScaler # To load/use scalers
import matplotlib.pyplot as plt

# --- Configuration ---
STUDENT_ID = 35
# These MUST match the NARX orders of the TRAINED NN model from part_b_2_2
N_u_model = 2  # u(k), u(k-1)
N_y_model = 1  # y(k-1)
INPUT_FEATURES_MODEL = N_u_model + N_y_model

# Trained NN model path and scaler paths

MODEL_PATH = f"best_model_system2_Nu{N_u_model}_Ny{N_y_model}_ID{STUDENT_ID}.pth"

dataset_dir = pathlib.Path("Question_1/Part_b")
dataset_filename_X_for_scaler = dataset_dir / f"system2_narx_X_data_Nu{N_u_model}_Ny{N_y_model}_ID{STUDENT_ID}.npy"
dataset_filename_Y_for_scaler = dataset_dir / f"system2_narx_Y_data_Nu{N_u_model}_Ny{N_y_model}_ID{STUDENT_ID}.npy"


# NN Hyperparameters (must match the architecture of the loaded model)
HIDDEN_SIZE_MODEL = 32 # Or whatever was used for the saved model

# Control Parameters
U_MIN_CTRL = -1.5 # Min control input
U_MAX_CTRL = 1.5  # Max control input
NUM_ITER_U = 50   # Number of iterations to find u(k)
STEP_SIZE_U = 0.05 # Step size for searching u(k)

# Simulation settings
N_SIM_STEPS_CTRL = 500
T_STEPS_CTRL = np.arange(N_SIM_STEPS_CTRL)

# Reference Signal r(n)
reference_signal_ctrl = np.zeros(N_SIM_STEPS_CTRL)
reference_signal_ctrl[50:200] = 1.0
reference_signal_ctrl[250:400] = -0.5
reference_signal_ctrl[400:N_SIM_STEPS_CTRL] = 0.8

# --- Load System Module ---
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

System2_class_ctrl = load_system_class_definition("System2")
s_ctrl = System2_class_ctrl(STUDENT_ID) # Plant instance

# --- Define Neural Network Model  ---
HIDDEN_SIZE_1 = 32
HIDDEN_SIZE_2 = 16

class System2NN_Ctrl(nn.Module):
    def __init__(self, input_size=INPUT_FEATURES_MODEL, hidden_size1=HIDDEN_SIZE_1, hidden_size2=HIDDEN_SIZE_2, output_size=1):
        super(System2NN_Ctrl, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc_out = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc_out(x)
        return x

# --- Load Trained NN Model and Scalers ---
print("--- Loading Trained NN Model and Scalers ---")
nn_model_ctrl = System2NN_Ctrl()
try:
    nn_model_ctrl.load_state_dict(torch.load(MODEL_PATH))
    nn_model_ctrl.eval() # Set to evaluation mode
    print(f"Loaded trained NN model from {MODEL_PATH}")
except FileNotFoundError:
    print(f"Error: Trained model file {MODEL_PATH} not found. Train the NN first (Q1.b.2).")
    sys.exit()

# Load dataset to re-fit scalers
try:
    X_data_for_scaler = np.load(dataset_filename_X_for_scaler)
    Y_data_for_scaler = np.load(dataset_filename_Y_for_scaler)
    scaler_X_ctrl = StandardScaler().fit(X_data_for_scaler)
    scaler_Y_ctrl = StandardScaler().fit(Y_data_for_scaler)
    print("Scalers re-fitted from the dataset.")
except FileNotFoundError:
    print("Error: Dataset files for scalers not found. Cannot proceed.")
    sys.exit()


# --- History tracking ---
y_history_ctrl = np.zeros(N_SIM_STEPS_CTRL)
u_history_ctrl = np.zeros(N_SIM_STEPS_CTRL)
r_history_ctrl = reference_signal_ctrl # Same as reference_signal_ctrl
e_history_ctrl = np.zeros(N_SIM_STEPS_CTRL)

# --- Initial conditions for controller state ---
s_ctrl.reset()
y_prev_ctrl = [0.0] * N_y_model # y(k-1), y(k-2)...
u_prev_ctrl = [0.0] * (N_u_model - 1) # u(k-1), u(k-2)...
current_u_k = 0.0 # Initial guess for u(k)

print("--- Starting NN Model-Based Control Simulation ---")
for n in range(N_SIM_STEPS_CTRL):
    r_n_ctrl = reference_signal_ctrl[n]

    # Iteratively u(k) that makes NN_output(u(k), ...) close to r(k)
    # This is a simple gradient ascent / search method on u(k)
    # Starting with u(k) = u(k-1)
    if n > 0: current_u_k = u_history_ctrl[n-1] 
    else: current_u_k = 0.0 
    
    best_u_k = current_u_k
    # Target output for NN (reference, scaled)
    r_n_scaled = scaler_Y_ctrl.transform(np.array([[r_n_ctrl]])) 

    # Iterative search for u_k
    for _iter in range(NUM_ITER_U):
        # Positive and negative steps from current_u_k
        u_candidates = [current_u_k, current_u_k + STEP_SIZE_U, current_u_k - STEP_SIZE_U]
        u_candidates = [np.clip(u_c, U_MIN_CTRL, U_MAX_CTRL) for u_c in u_candidates] # Clip candidates
        
        min_error_iter = float('inf')
        
        for u_try in u_candidates:
            # NN input features: [u_try, u_prev_ctrl..., y_prev_ctrl...]
            nn_input_features_list = [u_try] + u_prev_ctrl + y_prev_ctrl
            nn_input_np = np.array(nn_input_features_list).reshape(1, -1)
            nn_input_scaled_np = scaler_X_ctrl.transform(nn_input_np)
            nn_input_tensor = torch.tensor(nn_input_scaled_np, dtype=torch.float32)

            with torch.no_grad():
                predicted_y_k_scaled_tensor = nn_model_ctrl(nn_input_tensor)
            
            # Error between scaled reference and scaled NN prediction
            # I want predicted_y_k_scaled_tensor to be close to r_n_scaled
            current_iter_error = ((predicted_y_k_scaled_tensor.numpy() - r_n_scaled)**2).sum()

            if current_iter_error < min_error_iter:
                min_error_iter = current_iter_error
                best_u_k_iter = u_try # This u_try gives prediction closest to reference
        
        current_u_k = best_u_k_iter # Updating current_u_k for next search iteration

    # Final chosen u(k) for this time step
    u_n_final = np.clip(current_u_k, U_MIN_CTRL, U_MAX_CTRL)

    # Applying u_n_final to the actual plant
    y_n_actual_ctrl = s_ctrl.output(u_n_final)

    # Storing history
    y_history_ctrl[n] = y_n_actual_ctrl
    u_history_ctrl[n] = u_n_final
    e_history_ctrl[n] = r_n_ctrl - y_n_actual_ctrl

    # Update history for NEXT time step's NARX model input
    # y_prev_ctrl uses actual plant output
    new_y_history = [y_n_actual_ctrl] + y_prev_ctrl[:-1] if N_y_model > 0 else []
    y_prev_ctrl = new_y_history[:N_y_model]

    # u_prev_ctrl uses the u_n_final we just applied
    new_u_history = [u_n_final] + u_prev_ctrl[:-1] if (N_u_model -1) > 0 else []
    u_prev_ctrl = new_u_history[:(N_u_model-1)]

print("NN Model-Based Control Simulation finished.")

# --- Plotting Results ---
plt.figure(figsize=(12, 8))
plt.suptitle(f'NN Model-Based Control (Iterative Inversion, Nu={N_u_model}, Ny={N_y_model})', fontsize=14)

plt.subplot(3, 1, 1)
plt.plot(T_STEPS_CTRL, r_history_ctrl, 'k--', linewidth=1.5, label='Reference r(n)')
plt.plot(T_STEPS_CTRL, y_history_ctrl, 'b-', linewidth=1.5, label='System Output y(n)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(T_STEPS_CTRL, u_history_ctrl, 'r-', label='Control Signal u(n)')
plt.plot(T_STEPS_CTRL, np.full_like(T_STEPS_CTRL, U_MAX_CTRL), 'r:', linewidth=1, label='U_max')
plt.plot(T_STEPS_CTRL, np.full_like(T_STEPS_CTRL, U_MIN_CTRL), 'r:', linewidth=1, label='U_min')
plt.ylabel('Control Input')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(T_STEPS_CTRL, e_history_ctrl, 'g-', label='Error e(n) = r(n)-y(n)')
plt.ylabel('Error')
plt.xlabel('Time step n')
plt.legend()
plt.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
# --- Save the plot as PNG ---
output_dir = pathlib.Path("Question_1/Part_b")
output_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(output_dir / "q1b3_nn_model_based_control.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved: q1b3_nn_model_based_control.png")


mse_tracking_ctrl = np.mean(e_history_ctrl**2)
print(f"\nTracking Performance (NN Model-Based Control):")
print(f"  Mean Squared Error (r(n)-y(n)): {mse_tracking_ctrl:.4f}")
final_abs_error_avg_ctrl = np.mean(np.abs(e_history_ctrl[-50:]))
print(f"  Avg absolute error in last 50 steps: {final_abs_error_avg_ctrl:.4f}")
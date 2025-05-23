"""
Script: part_b_2_1.py
Purpose: Generates a rich NARX input-output dataset by exciting System2 with various input types.
Components:
  - Signal generator with configurable types
  - Data collection using NARX structure
  - Visualization and saving of a sample signal-response pair
  - Export of dataset as .npy files
"""

import sys
import pathlib
import importlib.util
import numpy as np
import matplotlib.pyplot as plt


# --- Configuration ---
STUDENT_ID = 35
N_u = 2
N_y = 1

# --- Load System2 ---
def load_system_module(system_name="System2"):
    pyc_filename = f"{system_name}.cpython-38.pyc"
    pyc_path = pathlib.Path("./python") / pyc_filename
    if not pyc_path.exists():
        print(f"Error: {pyc_filename} not found at {pyc_path}")
        sys.exit()
    spec = importlib.util.spec_from_file_location(system_name.lower(), pyc_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, system_name)

System2 = load_system_module("System2")
s = System2(STUDENT_ID)

# --- Signal Generator ---
def generate_input_signal(length, signal_type="random_steps", low=-1.0, high=1.0, num_steps=20, freq=1.0):
    t = np.linspace(0, (length-1)*0.1, length)
    
    if signal_type == "random_steps":
        changes = np.random.uniform(low, high, num_steps)
        signal = np.repeat(changes, length // num_steps)
        if len(signal) < length:
            signal = np.pad(signal, (0, length - len(signal)), 'edge')
        return signal[:length]

    elif signal_type == "sinusoid":
        return high * np.sin(2 * np.pi * freq * t * np.random.uniform(0.5, 1.5))

    elif signal_type == "ramp":
        return np.linspace(low, high, length)

    elif signal_type == "steps":
        # Constant amplitude steps at certain intervals
        signal = np.zeros(length)
        step_times = np.linspace(0, length, num_steps + 1, dtype=int)
        amplitudes = np.random.uniform(low, high, num_steps)
        for i in range(num_steps):
            signal[step_times[i]:step_times[i+1]] = amplitudes[i]
        return signal

    else:
        return np.random.uniform(low, high, length)


# --- Collect NARX Data ---
def collect_narx_data_from_sequence(system_instance, u_sequence, n_u, n_y):
    system_instance.reset()
    sequence_X, sequence_Y = [], []
    
    # Circular buffer-like structure to hold input and output histories
    u_history = [0.0] * (n_u - 1)
    y_history = [0.0] * n_y
    max_lag = max(n_u - 1, n_y)

    for k in range(len(u_sequence)):
        u_k = u_sequence[k]
        y_k = system_instance.output(u_k)

        # Once enough history is accumulated (lag period is complete), create NARX input-output pair
        if k >= max_lag:
            u_features = [u_k]  # current input u(k)
            if n_u > 1:
                u_features.extend(u_history[-(n_u - 1):])  # past entries u(k-1), u(k-2)...

            y_features = []
            if n_y > 0:
                y_features.extend(y_history[-n_y:])  # past outputs y(k-1), y(k-2)...

            # input vector (X): [u(k), u(k-1), ..., y(k-1), ...]
            sequence_X.append(u_features + y_features)

            # Target output (Y): y(k)
            sequence_Y.append([y_k])

        # Updating login and logout histories
        if n_u > 1:
            u_history.append(u_k)
            if len(u_history) > (n_u - 1): u_history.pop(0)
        if n_y > 0:
            y_history.append(y_k)
            if len(y_history) > n_y: y_history.pop(0)

    return np.array(sequence_X), np.array(sequence_Y)


# --- Data Generation ---
print(f"--- Generating NARX Dataset for System 2 (ID: {STUDENT_ID}) ---")
all_X_data_list, all_Y_data_list = [], []
num_total_samples_target = 2000
samples_per_sequence = 200
num_sequences_to_generate = int(np.ceil(num_total_samples_target / (samples_per_sequence - max(N_u - 1, N_y))))

signal_configs = [
    {"type": "random_steps", "low": -0.5, "high": 1.5, "num_steps": 20},
    {"type": "random_steps", "low": -1.0, "high": 1.0, "num_steps": 40},
    {"type": "sinusoid", "low": 0, "high": 1.0, "freq": 0.1},
    {"type": "sinusoid", "low": 0, "high": 0.5, "freq": 0.5},
    {"type": "sinusoid", "low": 0, "high": 1.5, "freq": 0.05},
    {"type": "ramp", "low": -0.5, "high": 1.5},
    {"type": "ramp", "low": 1.0, "high": 0.0},
    {"type": "steps", "low": -1.0, "high": 1.0, "num_steps": 10},
]


# --- Dataset Generation Loop ---
# Data is collected from the system with different signals
# Purpose: To produce a sufficient variety of input-output samples for the NARX structure
for i in range(num_sequences_to_generate):
    config = signal_configs[i % len(signal_configs)]
    print(f"Generating sequence {i+1}/{num_sequences_to_generate} with type: {config['type']}")
    u_seq = generate_input_signal(samples_per_sequence,
                                  signal_type=config["type"],
                                  low=config["low"],
                                  high=config["high"],
                                  num_steps=config.get("num_steps", 20),
                                  freq=config.get("freq", 1.0))
    # NARX input-output samples are collected
    X_seq, Y_seq = collect_narx_data_from_sequence(s, u_seq, N_u, N_y)
    
    # If there is enough data, it will be added to the list
    if X_seq.shape[0] > 0: 
        all_X_data_list.append(X_seq)
        all_Y_data_list.append(Y_seq)

if all_X_data_list:
    X_data_np = np.concatenate(all_X_data_list, axis=0)
    Y_data_np = np.concatenate(all_Y_data_list, axis=0)

    print(f"\n--- Dataset Summary ---")
    print(f"Total NARX samples generated: {X_data_np.shape[0]}")
    print(f"X shape: {X_data_np.shape}, Y shape: {Y_data_np.shape}")

    # --- Plotting ---
    last_u_seq = u_seq
    s.reset()
    last_y_response = [s.output(u_val) for u_val in last_u_seq]

    plt.figure(figsize=(12, 6))
    plt.subplot(2,1,1)
    plt.plot(last_u_seq, label='Input Signal u(k)')
    plt.ylabel('u(k)')
    plt.grid(True)
    plt.legend()

    plt.subplot(2,1,2)
    plt.plot(last_y_response, label='System Response y(k)')
    plt.xlabel('Time step k')
    plt.ylabel('y(k)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    # Save plot
    output_dir = pathlib.Path("Question_1/Part_b")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_save_path = output_dir / "q1b2_generated_signal_example.png"
    plt.savefig(fig_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to {fig_save_path}")

    # --- Save dataset ---
    np.save(output_dir / f"system2_narx_X_data_Nu{N_u}_Ny{N_y}_ID{STUDENT_ID}.npy", X_data_np)
    np.save(output_dir / f"system2_narx_Y_data_Nu{N_u}_Ny{N_y}_ID{STUDENT_ID}.npy", Y_data_np)
    print("Dataset files saved.")
else:
    print("No data generated.")

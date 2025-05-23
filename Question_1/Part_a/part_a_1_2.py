import sys, pathlib, importlib.util
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# ------------------------ Load the system class ------------------------
pyc_path = pathlib.Path("./") / "python/System1.cpython-38.pyc"
spec = importlib.util.spec_from_file_location("system1", pyc_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
System1 = module.System1
# -----------------------------------------------------------------------

s = System1(35)  # last two digits of student ID:110190235
output_dir = pathlib.Path("Question_1")
output_dir.mkdir(parents=True, exist_ok=True)

# a.1) STEP RESPONSE
t_vec = np.linspace(0, 5, 200)
t, y_step = s.step(t_vec)

step_mean = np.mean(y_step)
step_std = np.std(y_step)

plt.figure()
plt.plot(t, y_step)
plt.title("Step Response")
plt.grid()
plt.savefig(output_dir / "q1a_step_response.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"🔹 Step Response Mean: {step_mean:.3f}")
print(f"🔹 Step Response Std Dev (noise): {step_std:.3f}")

# a.1) RAMP RESPONSE
t, y_ramp = s.ramp(t_vec)
ramp_gain = (y_ramp[-1] - y_ramp[0]) / (t_vec[-1] - t_vec[0])

plt.figure()
plt.plot(t, y_ramp)
plt.title("Ramp Response")
plt.grid()
plt.savefig(output_dir / "q1a_ramp_response.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"🔹 Ramp Gain Estimate: {ramp_gain:.3f}")

# a.1) INPUT-OUTPUT LINEAR FIT
input_vec = np.sort(np.random.rand(10))
output_vec = s.output(input_vec)
slope, intercept, r_value, _, _ = linregress(input_vec, output_vec)

plt.figure()
plt.plot(input_vec, output_vec, 'o-', label="System Output")
plt.plot(input_vec, slope * input_vec + intercept, '--', label=f"Fit: y={slope:.2f}x+{intercept:.2f}")
plt.title("System Outputs (Linear Fit)")
plt.legend()
plt.grid()
plt.savefig(output_dir / "q1a_linear_fit.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"🔹 Linear Fit Gain: {slope:.3f}")
print(f"🔹 Linear Fit Offset: {intercept:.3f}")
print(f"🔹 R² Score: {r_value**2:.4f}")

K_est = slope
C_est = intercept

# a.2) CONTROLLER DESIGN
print(f"\n--- Controller Design (a.2) ---")
print(f"Using K_est = {K_est:.3f}, C_est = {C_est:.3f}")

t_control = np.linspace(0, 5, 100)
r_desired_value = 10.0
reference_signal = np.full_like(t_control, r_desired_value)

u_ff = (r_desired_value - C_est) / K_est
print(f"For r_desired = {r_desired_value}, calculated u_ff = {u_ff:.3f}")

controlled_output_series = s.output(np.full_like(t_control, u_ff))

plt.figure(figsize=(10, 6))
plt.plot(t_control, reference_signal, 'r--', label=f'Reference Signal (r = {r_desired_value})')
plt.plot(t_control, controlled_output_series, 'b-', label='Controlled System Output (y)')
plt.axhline(np.mean(controlled_output_series), color='g', linestyle=':', label=f'Mean Output ({np.mean(controlled_output_series):.2f})')
plt.title('System Response with Feedforward Controller (ID: 35)')
plt.xlabel('Time (s)')
plt.ylabel('Output')
plt.legend()
plt.grid(True)
plt.ylim(r_desired_value - 1, r_desired_value + 1)
plt.savefig(output_dir / "q1a2_control_response.png", dpi=300, bbox_inches='tight')
plt.close()

tracking_error = np.mean(controlled_output_series) - r_desired_value
print(f"Mean controlled output: {np.mean(controlled_output_series):.3f}")
print(f"Steady-state tracking error: {tracking_error:.3f}")
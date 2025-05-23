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

s = System1(35)  # last two digits of student ID

# STEP RESPONSE
t_vec = np.linspace(0, 5, 200)
t, y_step = s.step(t_vec)

step_mean = np.mean(y_step)
step_std = np.std(y_step)

plt.figure()
plt.plot(t, y_step)
plt.title("Step Response")
plt.grid()
plt.show()

print(f"🔹 Step Response Mean: {step_mean:.3f}")
print(f"🔹 Step Response Std Dev (noise): {step_std:.3f}")

# RAMP RESPONSE
t, y_ramp = s.ramp(t_vec)

ramp_gain = (y_ramp[-1] - y_ramp[0]) / (t_vec[-1] - t_vec[0])

plt.figure()
plt.plot(t, y_ramp)
plt.title("Ramp Response")
plt.grid()
plt.show()

print(f"🔹 Ramp Gain Estimate: {ramp_gain:.3f}")

# INPUT-OUTPUT TEST
input_vec = np.sort(np.random.rand(10))
output_vec = s.output(input_vec)

slope, intercept, r_value, _, _ = linregress(input_vec, output_vec)

plt.figure()
plt.plot(input_vec, output_vec, 'o-', label="System Output")
plt.plot(input_vec, slope * input_vec + intercept, '--', label=f"Fit: y={slope:.2f}x+{intercept:.2f}")
plt.title("System Outputs (Linear Fit)")
plt.legend()
plt.grid()
plt.show()

print(f"🔹 Linear Fit Gain: {slope:.3f}")
print(f"🔹 Linear Fit Offset: {intercept:.3f}")
print(f"🔹 R² Score: {r_value**2:.4f}")

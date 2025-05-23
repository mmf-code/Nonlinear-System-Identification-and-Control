import sys, pathlib, importlib.util
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress # For asymptote fit

############################ Load the system class ############################
pyc_path = pathlib.Path("./") / "python/System2.cpython-38.pyc"

if not pyc_path.exists():
    print(f"Error: System2 compiled file not found at {pyc_path}")
    print("Please ensure the pyc_path variable is set correctly.")
    sys.exit()

spec = importlib.util.spec_from_file_location("system2", pyc_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
System2 = module.System2
##############################################################################

s = System2(35) # Last two digits student number

# Initial Plots as in example code
print("--- Generating Initial Diagnostic Plots for System 2 ---")
# 1 - Step input
s.reset()
t_step, y_step = s.step() # Using default t_final=5. Store for fitting.

plt.figure(figsize=(8,5))
plt.plot(t_step, y_step)
plt.title("System 2: Original Step Response (ID: 35)")
plt.xlabel("Time (s)")
plt.ylabel("Output")
plt.grid()

output_dir = pathlib.Path("Question_1/Part_b")
output_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(output_dir / "q1b1_step_response.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved: q1b1_step_response.png")


# 2 - Ramp input results
s.reset()
t_ramp, y_ramp = s.ramp() # Using default t_final=5

plt.figure(figsize=(8,5))
plt.plot(t_ramp, y_ramp)
plt.title("System 2: Original Ramp Response (ID: 35)")
plt.xlabel("Time (s)")
plt.ylabel("Output")
plt.grid()

plt.savefig(output_dir / "q1b1_ramp_response.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved: q1b1_ramp_response.png")


# 3 - Test with arbitrary input sequence
s.reset() 
inputs_seq = np.linspace(0, 1, 100) # A slow ramp-like sequence of inputs
outputs_seq = [s.output(float(u)) for u in inputs_seq] # System evolves with each call

plt.figure(figsize=(8,5))
plt.plot(inputs_seq, outputs_seq)
plt.title("System 2: Output for Varying Input Sequence (ID: 35)")
plt.xlabel("Sequential Input Value")
plt.ylabel("Output")
plt.grid()

plt.savefig(output_dir / "q1b1_input_output_sequence.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved: q1b1_input_output_sequence.png")


# --- System Identification for b.1 ---
print("\n--- System 2 Identification (b.1) ---")
print("Model hypothesis: G(s) = K / (s*(tau*s + 1))")
print("Theoretical step response: y(t) = K * (t - tau + tau * exp(-t/tau))")

# Theoretical step response function for curve_fit
def integrated_first_order_step_response(t, K_param, tau_param):
    # Ensuring tau is positive and not too small to avoid numerical issues
    tau_abs = np.abs(tau_param)
    if tau_abs < 1e-6:  # Preventing division by zero
        tau_abs = 1e-6
    
    response = K_param * (t - tau_abs + tau_abs * np.exp(-t / tau_abs))
    # Response should be non-negative; that's a physical constraint (often is for step responses from 0)
    return np.maximum(0, response)

# Initial guesses for K and tau for curve_fit
# From visual inspection of the step response:
#   - The final output y(5) is approximately 2.75–2.8
#   - This implies asymptote: y_as = K*(5 - tau) ≈ 2.8
# Trying values:
#   K_guess = 0.65
#   tau_guess = 0.85
# Then: y_as(5) = 0.65 * (5 - 0.85) = 0.65 * 4.15 ≈ 2.70, which closely matches the observed value
K_guess_cf = 0.65 
tau_guess_cf = 0.85
print(f"Initial guesses for curve_fit: K={K_guess_cf:.3f}, tau={tau_guess_cf:.3f}")

K_cf, tau_cf = None, None  # initializing parameters


try:
    # Bounds for parameters: K > 0, tau > 0. 
    # max_tau_bound = t_step[-1] / 2.0 if t_step[-1] / 2.0 > 1e-3 else 5.0 # Avoid too large tau
    max_tau_bound = 5.0 # Since t_final is 5s for default step
    
    popt_cf, pcov_cf = curve_fit(
        integrated_first_order_step_response, 
        t_step, 
        y_step, 
        p0=[K_guess_cf, tau_guess_cf],
        bounds=([1e-4, 1e-4], [10.0, max_tau_bound]),
        maxfev=5000
    )
    K_cf, tau_cf = popt_cf
    
    print(f"\nCurve Fit Parameters (for G(s) = K/(s(τs+1))):")
    print(f"  K_curve_fit = {K_cf:.4f}")
    print(f"  tau_curve_fit = {tau_cf:.4f}")

    y_fitted_model_step = integrated_first_order_step_response(t_step, K_cf, tau_cf)

    plt.figure(figsize=(10,6))
    plt.plot(t_step, y_step, 'b-', label='Observed Step Response')
    plt.plot(t_step, y_fitted_model_step, 'r--', linewidth=2, label=f'Fitted Model: K={K_cf:.2f}, τ={tau_cf:.2f}')
    plt.title("System 2: Step Response & Fitted Model G(s) = K/(s(τs+1)) (ID: 35)")
    plt.xlabel("Time (s)")
    plt.ylabel("Output")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "q1b1_curve_fit_model.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: q1b1_curve_fit_model.png")


except RuntimeError as e:
    print(f"Curve fit failed: {e}.")
    print("Try adjusting initial guesses (K_guess_cf, tau_guess_cf) or bounds if results are poor.")
except ValueError as e:
    print(f"ValueError during curve_fit: {e}.")
    print("This might be due to bounds or initial parameters leading to invalid math operations.")

# Optional: Asymptote fit from the linear part of the step response
print("\n--- Optional: Asymptote Fit from Step Response ---")

time_for_linear_asymptote = 2.5
if tau_cf is not None and tau_cf > 0:
    time_for_linear_asymptote = max(2.5, 2.5 * tau_cf)  # More robust linear start estimate

start_indices_linear = np.where(t_step >= time_for_linear_asymptote)[0]
K_asym_fit, tau_asym_fit = None, None

if len(start_indices_linear) > 0:
    start_idx_linear = start_indices_linear[0]
    t_linear_part = t_step[start_idx_linear:]
    y_linear_part = y_step[start_idx_linear:]

    if len(t_linear_part) > 5:  # Need enough points for a reliable fit
        slope_K_asym, intercept_negKtau_asym, r_val_asym, _, _ = linregress(t_linear_part, y_linear_part)

        if np.abs(slope_K_asym) > 1e-6:
            K_asym_fit = slope_K_asym
            tau_asym_fit = -intercept_negKtau_asym / K_asym_fit

            print(f"Asymptote Fit Parameters (less robust):")
            print(f"  K_asymptote_fit = {K_asym_fit:.4f}")
            print(f"  tau_asymptote_fit = {tau_asym_fit:.4f} (R_squared for linear part: {r_val_asym**2:.3f})")

            # Plot and save
            plt.figure(figsize=(10, 6))
            plt.plot(t_step, y_step, 'b-', label='Observed Step Response')

            if K_cf is not None and tau_cf is not None:
                plt.plot(t_step, integrated_first_order_step_response(t_step, K_cf, tau_cf), 
                         'm:', linewidth=2, label=f'Curve Fit Model (preferred)')

            asymptote_line = K_asym_fit * t_step - K_asym_fit * tau_asym_fit
            plt.plot(t_step, asymptote_line, 'g--', linewidth=2, 
                     label=f'Asymptote Fit: K={K_asym_fit:.2f}, τ={tau_asym_fit:.2f}')
            plt.plot(t_linear_part, y_linear_part, 'kx', markersize=5, label='Data for asymptote fit')
            plt.title("System 2: Step Response, Model Fits, & Asymptote (ID: 35)")
            plt.xlabel("Time (s)")
            plt.ylabel("Output")
            plt.legend()
            plt.grid(True)

            # Save to file
            output_dir = pathlib.Path("Question_1/Part_b")
            output_dir.mkdir(parents=True, exist_ok=True)
            save_path = output_dir / "q1b1_asymptote_fit_model.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved: {save_path}")

        else:
            print("Slope of the linear part is too close to zero for a reliable asymptote fit.")
    else:
        print("Not enough data points in the presumed linear region for asymptote fit.")
else:
    print("Could not determine a suitable linear region for asymptote fit based on t_step values.")



# --- Final Summary for b.1 ---
print("\n--- Summary of System 2 Identification (b.1) ---")
final_K, final_tau = None, None

if K_cf is not None and tau_cf is not None:
    final_K, final_tau = K_cf, tau_cf
    print(f"Primary parameters (from curve_fit):")
elif K_asym_fit is not None and tau_asym_fit is not None:
    final_K, final_tau = K_asym_fit, tau_asym_fit
    print(f"Fallback parameters (from asymptote fit - less reliable for this model):")
else:
    print("Parameter estimation failed or was not completed successfully.")

if final_K is not None and final_tau is not None:
    print(f"  Estimated K = {final_K:.4f}")
    print(f"  Estimated τ = {final_tau:.4f}")
    print(f"Identified Transfer Function G(s) = {final_K:.4f} / (s * ({final_tau:.4f}s + 1))")
    # Differential equation: τ d²y/dt² + dy/dt = K u(t)
    print(f"Equivalent Differential Equation: {final_tau:.4f} * d²y/dt² + dy/dt = {final_K:.4f} * u(t)")
else:
    print("Could not provide final parameters for System 2.")
import sys, pathlib, importlib.util
import numpy as np
import matplotlib.pyplot as plt

# Example usage of the System2 class. If you write your own code, copy the 
# part below.
# You need Python 3.8 to use the compiled module.

############################ Load the system class ############################
pyc_path = pathlib.Path("./") / "python/System2.cpython-38.pyc"
spec = importlib.util.spec_from_file_location("system2", pyc_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

System2 = module.System2
##############################################################################

s = System2(35) # Last two digits of your student ID (last digit if it is 0x)

# 1 - Step input
s.reset()
t, y = s.step() # Optionally, you can give t_final input

plt.figure()
plt.plot(t, y)
plt.title("Step Response")
plt.grid()
plt.show()

# 2 - Ramp input results
s.reset()
t, y = s.ramp() # Optionally, you can give t_final input

plt.figure()
plt.plot(t, y)
plt.title("Ramp Response ")
plt.grid()
plt.show()

# 3 - Test with arbitrary input sequence
# This system takes one input at a time, so we'll simulate a sequence
s.reset() # Dont forget to reset
inputs = np.linspace(0, 1, 100)
outputs = [s.output(float(u)) for u in inputs]

plt.figure()
plt.plot(inputs, outputs)
plt.title("System Output for Varying Input")
plt.grid()
plt.show()

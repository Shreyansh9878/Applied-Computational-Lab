import subprocess
import pandas as pd
import re
import os
import numpy as np

# Change to your folder in WSL/Linux
os.chdir("/mnt/c/Users/shrey/Applied Computational Lab/photon_gas_mc")

# Generate 1000 BetaEpsilon values from 0 to 100
beta_eps_values = np.linspace(0, 50, 1001)

data = []

print("🚀 Starting Photon Gas simulations...")

for i, val in enumerate(beta_eps_values, start=1):
    input_data = f"1000\n500\n{val}\n"  # string input for the executable

    # Run the executable
    result = subprocess.run(
        ["./run"],          # Use "./run" in WSL/Linux
        input=input_data,   # pass string directly
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True           # stdout/stderr as strings
    )
    output = result.stdout

    # Extract numbers using regex
    avg_match = re.search(r"Average Value\s*:\s*([\d\.\-E+]+)", output)
    theo_match = re.search(r"Theoretical Value\s*:\s*([\d\.\-E+]+)", output)
    rel_err_match = re.search(r"Relative Error\s*:\s*([\d\.\-E+]+)", output)

    if avg_match and theo_match:
        avg_val = float(avg_match.group(1))
        theo_val = float(theo_match.group(1))
        rel_err = float(rel_err_match.group(1)) if rel_err_match else None

        data.append({
            "BetaEpsilon": val,
            "Calc_Avg_Val": avg_val,
            "Theo_Avg_Val": theo_val,
            "Relative_Error": rel_err
        })

    # Print progress every 50 runs
    if i % 50 == 0 or i == 1 or i == len(beta_eps_values):
        print(f"🔹 Running {i}/{len(beta_eps_values)} — BetaEpsilon = {val:.2f}")

# Save results to CSV
df = pd.DataFrame(data)
df.to_csv("Photon_gas.csv", index=False)

print("\n✅ All simulations completed.")
print("✅ Results saved to Photon_gas.csv")
import matplotlib.pyplot as plt
import os
import pandas as pd
import sys


if len(sys.argv) <= 1:
    sys.exit("Please provide a file to read from")
csv_filename = sys.argv[1]

# Read the file
df = pd.read_csv(csv_filename)

groups = df.groupby("n")
plt.figure(figsize=(10, 6))
for name, group in groups:
    plt.semilogy(group["iteration"], group["residual"], label=f"n = {name}")
plt.xlabel(r"Iteration $m=\frac{n}{2}$")
plt.ylabel(r"Normalised Residual $\|r_k\|_2/\|b\|_2$")
plt.title("GMRES Convergence for Different Matrix Sizes")
plt.legend()
plt.grid(True)

# Generate the output filename
base_name = os.path.splitext(csv_filename)[0]
output_filename = base_name + ".png"
plt.savefig(output_filename)

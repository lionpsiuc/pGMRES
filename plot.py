import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv("residuals.csv")
groups = df.groupby("n")
plt.figure(figsize=(10, 6))
for name, group in groups:
    plt.semilogy(group["iteration"], group["residual"], label=f"n = {name}")
plt.xlabel(r"Iteration $m=\frac{n}{2}$")
plt.ylabel(r"Normalised Residual $\|r_k\|_2/\|b\|_2$")
plt.title("GMRES Convergence for Different Matrix Sizes")
plt.legend()
plt.grid(True)
plt.savefig("residuals.png")

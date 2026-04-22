# Save this as plot_matrix.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys

# Pass the csv filename as an argument: python plot_matrix.py pairwise_kld_matrix_123.csv
csv_file = "pairwise_kld_matrix.csv"

# Load matrix, setting the first column as the row index
df = pd.read_csv(csv_file, index_col=0)

# Clean up row/col names for the graph (lora_0_5 -> 0.5)
clean_names = [name.replace('lora_', '').replace('_', '.') for name in df.columns]
df.columns = clean_names
df.index = clean_names

plt.figure(figsize=(10, 8))
sns.heatmap(df, annot=True, cmap="YlOrRd", fmt=".4f", cbar_kws={'label': 'KL Divergence'})
plt.title("Adapter Pairwise KL Divergence", fontsize=14, fontweight='bold')
plt.xlabel("Target Adapter (Model B)", fontsize=12)
plt.ylabel("Baseline Adapter (Model A)", fontsize=12)
plt.tight_layout()
plt.savefig("kld_heatmap.png", dpi=300)
print("Saved heatmap to kld_heatmap.png")
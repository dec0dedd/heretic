import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Create output directory for the graphs
os.makedirs('plots', exist_ok=True)

# Find all CSV files in the metrics directory
csv_files = glob.glob('metrics/*.csv')

if not csv_files:
    print("No CSV files found in 'metrics/' directory.")
    exit()

print(f"Found {len(csv_files)} models to compare. Generating comparative plot...")

# Set up a figure with 2 subplots stacked vertically, sharing the X-axis
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# Generate a distinct color palette based on the number of CSVs
colors = cm.get_cmap('tab10', len(csv_files))

for i, file in enumerate(csv_files):
    # Extract a clean name from the filename for the legend
    model_name = os.path.basename(file).replace('.csv', '').replace('adapter_metrics_', '')
    
    # Read and prep data
    df = pd.read_csv(file)
    df['Scale'] = df['Adapter'].str.replace('lora_', '', regex=False).str.replace('_', '.', regex=False).astype(float)
    df['Refusal_Rate'] = (df['Refusals'] / df['Total_Prompts']) * 100
    df = df.sort_values('Scale')
    
    color = colors(i)
    
    # Top subplot: Refusal Rate Phase Transition
    ax1.plot(df['Scale'], df['Refusal_Rate'], color=color, marker='o', 
             linewidth=2.5, markersize=7, label=model_name)
    
    # Bottom subplot: KL Divergence (Capability Degradation)
    ax2.plot(df['Scale'], df['KL_Divergence'], color=color, marker='s', 
             linewidth=2.5, markersize=7, label=model_name)

# ---------------------------------------------------------
# Formatting Top Subplot (Refusals)
# ---------------------------------------------------------
ax1.set_ylabel('Refusal Rate (%)', fontsize=12, fontweight='bold')
ax1.set_ylim(-5, 105)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.set_title('Cross-Model Ablation: Alignment vs. Capability', fontsize=14, fontweight='bold', pad=15)
ax1.legend(title='Models', fontsize=10, loc='best')

# ---------------------------------------------------------
# Formatting Bottom Subplot (KL Divergence)
# ---------------------------------------------------------
ax2.set_xlabel('Adapter Scale', fontsize=12, fontweight='bold')
ax2.set_ylabel('KL Divergence', fontsize=12, fontweight='bold')
# Start Y-axis slightly below 0 for aesthetic padding
ax2.set_ylim(bottom=-0.01)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(title='Models', fontsize=10, loc='best')

# Adjust layout to prevent overlapping labels
plt.tight_layout()

# Save the comparative plot at 300 DPI
out_path = 'plots/comparative_metrics.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ Saved high-res comparative plot: {out_path}")
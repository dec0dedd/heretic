import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# Create output directory for the graphs
os.makedirs('plots', exist_ok=True)

# Find all CSV files in the metrics directory
csv_files = glob.glob('metrics/*.csv')

if not csv_files:
    print("No CSV files found in 'metrics/' directory.")
else:
    print(f"Found {len(csv_files)} CSV file(s). Generating plots...")

for file in csv_files:
    # Read the data
    df = pd.read_csv(file)
    
    # Extract the numeric scale from the 'Adapter' column (e.g., 'lora_0_5' -> 0.5)
    # This handles the string manipulation safely.
    df['Scale'] = df['Adapter'].str.replace('lora_', '', regex=False).str.replace('_', '.', regex=False).astype(float)
    
    # Calculate Refusal Rate percentage
    df['Refusal_Rate'] = (df['Refusals'] / df['Total_Prompts']) * 100
    
    # Sort by scale just in case the rows are out of order
    df = df.sort_values('Scale')
    
    # Set up the figure and axis
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # ---------------------------------------------------------
    # AXIS 1: Refusal Rate (Red Line)
    # ---------------------------------------------------------
    color1 = '#d62728' # A nice academic red
    ax1.set_xlabel('Adapter Scale', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Refusal Rate (%)', color=color1, fontsize=12, fontweight='bold')
    line1 = ax1.plot(df['Scale'], df['Refusal_Rate'], color=color1, marker='o', 
                     linewidth=2.5, markersize=8, label='Refusal Rate')
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # Lock the left Y-axis to 0-105% so it's always readable
    ax1.set_ylim(-5, 105) 
    
    # Add a subtle grid
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # ---------------------------------------------------------
    # AXIS 2: KL Divergence (Blue Line)
    # ---------------------------------------------------------
    ax2 = ax1.twinx()  # Instantiate a second axes that shares the same x-axis
    color2 = '#1f77b4' # A nice academic blue
    ax2.set_ylabel('KL Divergence', color=color2, fontsize=12, fontweight='bold')
    line2 = ax2.plot(df['Scale'], df['KL_Divergence'], color=color2, marker='s', 
                     linewidth=2.5, markersize=8, label='KL Divergence')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Dynamically pad the top of the KL graph a bit so it doesn't touch the ceiling
    max_kl = df['KL_Divergence'].max()
    ax2.set_ylim(-0.01, max_kl * 1.2 if max_kl > 0 else 0.1)

    # ---------------------------------------------------------
    # Titling and Legends
    # ---------------------------------------------------------
    base_name = os.path.basename(file).replace('.csv', '')
    plt.title(f'Safety Alignment Phase Transition\n({base_name})', fontsize=14, fontweight='bold', pad=15)
    
    # Combine the legends from both axes into one box
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center left', fontsize=11, framealpha=0.9)
    
    # Make layout tight so nothing gets cut off
    plt.tight_layout()
    
    # Save the plot at 300 DPI (standard for academic papers)
    out_path = f'plots/{base_name}.png'
    plt.savefig(out_path, dpi=300)
    plt.close()
    
    print(f"✅ Saved high-res plot: {out_path}")
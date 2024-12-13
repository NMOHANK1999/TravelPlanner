import matplotlib.pyplot as plt
import numpy as np

# Data
models = ['Qwen2.5-1.5B', 'Qwen2.5-3B']
instruct_rates = [3.40, 1.20]
rlhf_rates = [6.60, 2.20]

# Bar positions and width
x = np.arange(len(models))
width = 0.35

# Plotting
fig, ax = plt.subplots(figsize=(8, 6))
bars_instruct = ax.bar(x - width/2, instruct_rates, width, label='Instruct', color='#1f77b4')
bars_rlhf = ax.bar(x + width/2, rlhf_rates, width, label='RLHF', color='#ff7f0e')

# Formatting
ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('Pass Rate (%)', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=10)
ax.legend(fontsize=10)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Remove border and save as PDF
plt.tight_layout()
plt.savefig('two_stage_pass_rates.pdf', bbox_inches='tight', transparent=True)
plt.show()
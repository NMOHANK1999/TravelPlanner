import matplotlib.pyplot as plt

# Data
x_labels = ["Instruct", "RLHF", "RLHF + ICL"]
x = range(len(x_labels))
qwen_1_5b = [59.00, 60.50, 60.60]
qwen_3b = [63.88, 63.90, 64.05]

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(x, qwen_1_5b, label="Qwen2.5-1.5B", marker="o", linewidth=2.5)
plt.plot(x, qwen_3b, label="Qwen2.5-3B", marker="s", linewidth=2.5)

# Format the axes
plt.xticks(ticks=x, labels=x_labels, fontsize=12)
plt.yticks(fontsize=12)
plt.ylabel("Commonsense (Micro) Constraints Satisfied (%)", fontsize=14)
plt.xlabel("Model Finetuning", fontsize=14)

# Add grid and legend
plt.grid(visible=True, linestyle="--", alpha=0.7)
plt.legend(fontsize=12, loc="lower right")

# Remove borders
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)

# Save the plot as a PDF
plt.tight_layout()
plt.savefig("commonsense_constraints_figure.pdf", format="pdf", bbox_inches="tight", pad_inches=0)

# Show the plot
plt.show()

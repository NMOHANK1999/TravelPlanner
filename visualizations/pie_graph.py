import os
import json
from collections import Counter
import matplotlib.pyplot as plt

# Directory containing the JSON files
json_dir = "two_stagemode_outputdir_test_qwen2.5_3B_rlhf/test"

# Initialize a counter to store the aggregated state counts
state_counter = Counter()

# Process each JSON file in the directory
for filename in os.listdir(json_dir):
    if filename.startswith("generated_plan_") and filename.endswith(".json"):
        file_path = os.path.join(json_dir, filename)
        with open(file_path, "r") as file:
            data = json.load(file)
            for item in data:
                action_logs = item.get("gpt-3.5-turbo_two-stage_action_logs", [])
                for log in action_logs:
                    state = log.get("state", "Unknown")
                    if state not in ['Successful', 'invalidAction']:
                        state = 'maxRetries'
                    state_counter[state] += 1

# Create the pie chart
labels = state_counter.keys()
sizes = state_counter.values()
colors = plt.cm.tab10(range(len(labels)))  # Use a colormap for consistent colors

# Create the pie chart
plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140, colors=colors)

# Save the pie chart as a PDF without borders
plt.savefig("step_state_distribution.pdf", bbox_inches="tight", format="pdf")
plt.close()

print("Pie chart saved as 'step_state_distribution.pdf'")
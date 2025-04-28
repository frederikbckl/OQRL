import os
import re

# Path to the logs folder
LOGS_FOLDER = "logs"

# Regular expression to find final average reward
average_reward_pattern = re.compile(r"Average Reward:\s+([0-9\.]+)")

# Initialize results
results = []

# Go through all log files
for filename in os.listdir(LOGS_FOLDER):
    if filename.startswith("output_") and filename.endswith(".txt"):
        filepath = os.path.join(LOGS_FOLDER, filename)
        job_id = filename.split("_")[1].replace(".txt", "")

        try:
            with open(filepath) as f:
                content = f.read()

            # Search for the final Average Reward
            matches = average_reward_pattern.findall(content)

            if matches:
                last_avg_reward = float(matches[-1])
                results.append((int(job_id), "SUCCESS", last_avg_reward))
            else:
                results.append((int(job_id), "SUCCESS", None))
        except Exception as e:
            results.append((int(job_id), f"ERROR: {e}", None))

# Sort results by job ID
results.sort(key=lambda x: x[0])

# Display results
print("\nSummary of SLURM Jobs:")
print("JobID\tStatus\t\tAverage Reward")
print("------------------------------------------")
for job_id, status, avg_reward in results:
    reward_str = f"{avg_reward:.2f}" if avg_reward is not None else "N/A"
    print(f"{job_id}\t{status}\t\t{reward_str}")

# Optional: Save summary to file
summary_path = os.path.join(LOGS_FOLDER, "summary.txt")
with open(summary_path, "w") as f:
    f.write("JobID\tStatus\tAverage Reward\n")
    f.write("------------------------------------------\n")
    for job_id, status, avg_reward in results:
        reward_str = f"{avg_reward:.2f}" if avg_reward is not None else "N/A"
        f.write(f"{job_id}\t{status}\t{reward_str}\n")

print(f"\nSummary saved to {summary_path}")

import os
import re

# Path to the logs folder
LOGS_FOLDER = "../logs"

# Regular expression to find final average reward
average_reward_pattern = re.compile(r"Average Reward:\s+([0-9\.]+)")

# Initialize results: will hold tuples of (job_id, seed, status, avg_reward)
results = []

# Go through all log files
for filename in os.listdir(LOGS_FOLDER):
    if filename.startswith("output_") and filename.endswith(".txt"):
        filepath = os.path.join(LOGS_FOLDER, filename)
        # Example filename: output_2748_0.txt -> job_id=2748, seed=0
        parts = filename.replace(".txt", "").split("_")
        if len(parts) >= 3:
            job_id = int(parts[1])
            seed = int(parts[2])
        else:
            job_id = int(parts[1])
            seed = -1  # Unknown seed

        try:
            with open(filepath) as f:
                content = f.read()

            # Search for the final Average Reward
            matches = average_reward_pattern.findall(content)

            if matches:
                last_avg_reward = float(matches[-1])
                results.append((job_id, seed, "SUCCESS", last_avg_reward))
            else:
                results.append((job_id, seed, "SUCCESS", None))
        except Exception as e:
            results.append((job_id, seed, f"ERROR: {e}", None))

# Sort results by jobID, then seed
results.sort(key=lambda x: (x[0], x[1]))

# Display results
print("\nSummary of SLURM Jobs:")
print("JobID\tSeed\tStatus\t\tAverage Reward")
print("------------------------------------------")
for job_id, seed, status, avg_reward in results:
    reward_str = f"{avg_reward:.2f}" if avg_reward is not None else "N/A"
    print(f"{job_id}\t{seed}\t{status}\t\t{reward_str}")

# Save summary to file
summary_path = os.path.join(LOGS_FOLDER, "summary.txt")
with open(summary_path, "w") as f:
    f.write("JobID\tSeed\tStatus\tAverage Reward\n")
    f.write("------------------------------------------\n")
    for job_id, seed, status, avg_reward in results:
        reward_str = f"{avg_reward:.2f}" if avg_reward is not None else "N/A"
        f.write(f"{job_id}\t{seed}\t{status}\t{reward_str}\n")

print(f"\nSummary saved to {summary_path}")

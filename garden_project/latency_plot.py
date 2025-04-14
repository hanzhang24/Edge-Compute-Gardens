import matplotlib.pyplot as plt

LATENCY_LOG = "latency_log.txt"

# Load latency values
with open(LATENCY_LOG, "r") as f:
    latencies = [float(line.strip()) for line in f]

# Plot histogram
plt.figure(figsize=(8, 5))
plt.hist(latencies, bins = 40, color="skyblue", edgecolor="black")
plt.xlabel("Latency (seconds)")
plt.ylabel("Frequency")
plt.title("Round Trip Latency Distribution")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read data
df = pd.read_csv("comprehensive_results.csv")

# Filter successful transactions
success_df = df[df["Success"] == True]

# --- 1. Success Rate Bar Plot (Zoomed with Labels) ---
plt.figure(figsize=(8, 5))
success_rate = df.groupby("WeightFunction")["Success"].mean().reset_index()
success_rate["Success Rate (%)"] = success_rate["Success"] * 100

barplot = sns.barplot(data=success_rate, x="WeightFunction", y="Success Rate (%)", palette="viridis", hue="WeightFunction", legend=False)
barplot.set_ylim(60, 100)
barplot.set_title("Success Rate by Weight Function")
barplot.set_ylabel("Success Rate (%)")

for i, row in success_rate.iterrows():
    barplot.text(i, row["Success Rate (%)"] + 0.5, f"{row['Success Rate (%)']:.1f}%", ha='center')

plt.tight_layout()
plt.savefig("poster_success_rate_zoomed.png")
plt.close()

# --- 2. Total Fee Distribution Box Plot ---
plt.figure(figsize=(12, 6))
sns.boxplot(data=success_df, x="WeightFunction", y="TotalFee", palette="pastel", hue="WeightFunction", legend=False)
plt.yscale("log")
plt.title("Distribution of Total Fees (Successful Transactions)")
plt.ylabel("Total Fee (sats, log scale)")
plt.tight_layout()
plt.savefig("poster_fee_distribution.png")
plt.close()

# --- 3. Summary Table CSV ---
summary = success_df.groupby("WeightFunction").agg({
    "TotalFee": "mean",
    "PathPe": "mean",
    "Success": "count"
}).rename(columns={
    "TotalFee": "AvgFee",
    "PathPe": "AvgPathPe",
    "Success": "NumSuccesses"
})

total_attempts = df.groupby("WeightFunction").size()
summary["SuccessRate (%)"] = (summary["NumSuccesses"] / total_attempts) * 100
summary = summary.reset_index()
summary.to_csv("summary_metrics_table.csv", index=False)

# --- 4. Fee and Path Success Plot (Side-by-Side Bars) ---
fig, ax1 = plt.subplots(figsize=(10, 6))

bar = sns.barplot(data=summary, x="WeightFunction", y="AvgFee", ax=ax1, color="skyblue")
ax1.set_ylabel("Average Fee (sats)", color="blue")
ax1.tick_params(axis='y', labelcolor="blue")

ax2 = ax1.twinx()
line = sns.lineplot(data=summary, x="WeightFunction", y="AvgPathPe", ax=ax2, color="green", marker="o", linewidth=2)
ax2.set_ylabel("Average Path Success Probability", color="green")
ax2.tick_params(axis='y', labelcolor="green")

plt.title("Cost vs Reliability by Weight Function")
plt.tight_layout()
plt.savefig("poster_fee_vs_pe.png")
plt.close()

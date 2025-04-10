import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read data
df = pd.read_csv("comprehensive_results.csv")

# Filter successful transactions
success_df = df[df["Success"] == True].copy()

# Calculate fee per sat
success_df.loc[:, "FeePerSat"] = success_df["TotalFee"] / success_df["Amount"]

summary = success_df.groupby("WeightFunction").agg({
    "TotalFee": "mean",
    "FeePerSat": "mean",
    "PathPe": "mean",
    "Success": "count"
}).rename(columns={
    "TotalFee": "AvgFee",
    "FeePerSat": "AvgFeePerSat",
    "PathPe": "AvgPathPe",
    "Success": "NumSuccesses"
})

# --- 1. Success Rate Bar Plot (Zoomed with Labels) ---
plt.figure(figsize=(8, 5))
success_rate = df.groupby("WeightFunction")["Success"].mean().reset_index()
success_rate["Success Rate (%)"] = success_rate["Success"] * 100

barplot = sns.barplot(data=success_rate, x="WeightFunction", y="Success Rate (%)", palette="viridis", hue="WeightFunction", legend=False)
barplot.set_title("Success Rate by Weight Function")
barplot.set_ylabel("Success Rate (%)")
plt.tight_layout()
plt.savefig("poster_success_rate.png")
plt.close()

# --- 2. Fee/Amount Distribution Box Plot ---
plt.figure(figsize=(12, 6))
sns.boxplot(data=success_df, x="WeightFunction", y="FeePerSat", palette="pastel", hue="WeightFunction", legend=False, showfliers=False)
plt.title("Distribution of Fee/Amount (Successful Transactions)")
plt.ylabel("Fee / Amount")
plt.tight_layout()
plt.savefig("poster_fee_distribution.png")
plt.close()

# --- 3. Summary Table CSV ---
total_attempts = df.groupby("WeightFunction").size()
summary["SuccessRate (%)"] = (summary["NumSuccesses"] / total_attempts) * 100
summary = summary.reset_index()
summary.to_csv("summary_metrics_table.csv", index=False)

# --- 4. Fee and Path Success Plot (Fee Only) ---
fig, ax1 = plt.subplots(figsize=(10, 6))

sns.barplot(data=summary, x="WeightFunction", y="AvgFeePerSat", ax=ax1, color="skyblue")
ax1.set_ylabel("Average Fee / Amount", color="blue")
ax1.tick_params(axis='y', labelcolor="blue")
plt.title("Fee Cost by Weight Function")
plt.tight_layout()
plt.savefig("poster_fee.png")
plt.close()

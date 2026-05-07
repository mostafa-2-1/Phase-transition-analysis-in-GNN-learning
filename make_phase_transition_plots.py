# # import os
# # import json
# # import pandas as pd
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import seaborn as sns
# # from scipy.optimize import curve_fit
# # from scipy.stats import bootstrap
# # from matplotlib.patches import Rectangle
# # import warnings

# # # Suppress specific warnings
# # warnings.filterwarnings('ignore', category=FutureWarning)

# # # ================================
# # # CONFIG — EDIT THESE PATHS ONLY
# # # ================================

# # PRUNED_ROOT = "pruned_graphs_k"
# # SOLVER_ROOT = "solver_results/seed1"
# # OUTPUT_DIR = "phase_transition_plots"

# # K_VALUES = [2, 3, 4, 5, 7, 8, 10, 15, 20, 25]

# # os.makedirs(OUTPUT_DIR, exist_ok=True)

# # # ================================
# # # Load FULL solver results
# # # ================================

# # full_results = pd.read_csv(os.path.join(SOLVER_ROOT, "full", "results.csv"))
# # full_results = full_results.rename(columns={"cost": "full_cost"})
# # full_results = full_results.set_index("instance")

# # # ================================
# # # Load PRUNED solver results
# # # ================================

# # solver_tables = {}

# # for k in K_VALUES:
# #     path = os.path.join(SOLVER_ROOT, f"k{k}", "results.csv")
# #     df = pd.read_csv(path)
# #     df = df.set_index("instance")
# #     solver_tables[k] = df

# # # ================================
# # # Load metadata JSONs
# # # ================================

# # records = []

# # for k in K_VALUES:
# #     k_dir = os.path.join(PRUNED_ROOT, f"k{k}")
# #     if not os.path.isdir(k_dir):
# #         continue
    
# #     for inst in os.listdir(k_dir):
# #         inst_dir = os.path.join(k_dir, inst)
# #         if not os.path.isdir(inst_dir):
# #             continue
        
# #         meta_path = os.path.join(inst_dir, f"{inst}_metadata.json")
# #         if not os.path.exists(meta_path):
# #             continue
        
# #         with open(meta_path, "r") as f:
# #             meta = json.load(f)
        
# #         instance = meta["instance_name"]
        
# #         # attach solver results
# #         if instance not in solver_tables[k].index:
# #             continue
        
# #         sol = solver_tables[k].loc[instance]
        
# #         record = {
# #             "instance": instance,
# #             "k": k,
# #             "n": meta["n_nodes"],
# #             "sparsity": meta["sparsity"],
# #             "avg_degree": meta["avg_degree"],
# #             "is_connected": meta["is_connected"],
# #             "feasible": meta["feasible"],
# #             "gap_rel": sol["gap_rel_percent"],
# #             "success": sol["success"]
# #         }
        
# #         records.append(record)

# # df = pd.DataFrame(records)

# # print("Loaded records:", len(df))
# # print("Data overview:")
# # print(f"Success rate: {df['success'].mean():.2%}")
# # print(f"Feasibility rate: {df['feasible'].mean():.2%}")
# # print(f"Average degree range: [{df['avg_degree'].min():.2f}, {df['avg_degree'].max():.2f}]")

# # # ================================
# # # Plot 1 — Gap vs Avg Degree
# # # ================================

# # plt.figure()
# # for connected in [True, False]:
# #     subset = df[df["is_connected"] == connected]
# #     plt.scatter(subset["avg_degree"], subset["gap_rel"],
# #                 marker="o" if connected else "x",
# #                 label=f"connected={connected}")

# # plt.xlabel("Average Degree")
# # plt.ylabel("Optimality Gap (%)")
# # plt.legend()
# # plt.title("Gap vs Avg Degree")
# # plt.savefig(os.path.join(OUTPUT_DIR, "gap_vs_avg_degree.png"), dpi=300)
# # plt.close()

# # # ================================
# # # Plot 2 — Gap vs Sparsity
# # # ================================

# # plt.figure()
# # plt.scatter(df["sparsity"], df["gap_rel"])
# # plt.xlabel("Sparsity (%)")
# # plt.ylabel("Optimality Gap (%)")
# # plt.title("Gap vs Sparsity")
# # plt.savefig(os.path.join(OUTPUT_DIR, "gap_vs_sparsity.png"), dpi=300)
# # plt.close()

# # # ================================
# # # Plot 3 — Logistic P(success) vs Avg Degree
# # # ================================

# # def logistic(x, beta, x0):
# #     z = np.clip(beta * (x - x0), -500, 500)  # prevent overflow
# #     return 1 / (1 + np.exp(-z))

# # # Filter out data with extreme values if needed
# # mask = ~df[["avg_degree", "success"]].isna().any(axis=1)
# # xdata = df.loc[mask, "avg_degree"].values
# # ydata = df.loc[mask, "success"].astype(int).values

# # print(f"\nLogistic regression data: {len(xdata)} points")

# # # Fit logistic
# # try:
# #     params, pcov = curve_fit(
# #         logistic,
# #         xdata,
# #         ydata,
# #         p0=(1.0, np.median(xdata)),
# #         maxfev=5000
# #     )
    
# #     beta_hat, x0_hat = params
# #     param_errors = np.sqrt(np.diag(pcov))
# #     print(f"Fitted parameters: beta={beta_hat:.3f} ± {param_errors[0]:.3f}, "
# #           f"x0={x0_hat:.3f} ± {param_errors[1]:.3f}")
    
# # except RuntimeError as e:
# #     print(f"Logistic fit failed: {e}")
# #     # Use median as fallback
# #     x0_hat = np.median(xdata)
# #     beta_hat = 1.0
# #     param_errors = [np.nan, np.nan]

# # # Try bootstrap with percentile method (more robust than BCa)
# # def fit_x0(x, y):
# #     """Helper function for bootstrap"""
# #     x = np.asarray(x)
# #     y = np.asarray(y)
    
# #     # Check if we have enough variation
# #     if len(x) < 10 or len(np.unique(y)) < 2:
# #         return np.nan
    
# #     try:
# #         p, _ = curve_fit(
# #             logistic,
# #             x,
# #             y,
# #             p0=(1.0, np.median(x)),
# #             maxfev=5000
# #         )
# #         return p[1]  # Return x0
# #     except (RuntimeError, ValueError):
# #         return np.nan

# # # Bootstrap with simpler percentile method
# # rng = np.random.default_rng(42)
# # n_resamples = 2000  # Reduced for speed, increase if needed
# # x0_bootstrap = []

# # for i in range(n_resamples):
# #     # Resample with replacement
# #     indices = rng.integers(0, len(xdata), len(xdata))
# #     x_resampled = xdata[indices]
# #     y_resampled = ydata[indices]
    
# #     x0_val = fit_x0(x_resampled, y_resampled)
# #     if not np.isnan(x0_val):
# #         x0_bootstrap.append(x0_val)

# # x0_bootstrap = np.array(x0_bootstrap)

# # if len(x0_bootstrap) > 0:
# #     # Calculate percentile-based CI
# #     x0_low = np.percentile(x0_bootstrap, 2.5)
# #     x0_high = np.percentile(x0_bootstrap, 97.5)
# #     print(f"Bootstrap: {len(x0_bootstrap)} successful resamples out of {n_resamples}")
# # else:
# #     x0_low, x0_high = np.nan, np.nan
# #     print("Warning: Bootstrap failed to produce any valid estimates")

# # # Smooth curve for plotting
# # x_plot = np.linspace(min(xdata), max(xdata), 200)
# # y_plot = logistic(x_plot, beta_hat, x0_hat)

# # plt.figure(figsize=(10, 6))
# # plt.scatter(xdata, ydata, alpha=0.5, label="Data points", s=30)

# # # Only plot fit if we have valid parameters
# # if not np.isnan(beta_hat):
# #     plt.plot(x_plot, y_plot, 'r-', linewidth=2, label="Logistic fit")
# #     plt.axvline(x0_hat, color='red', linestyle='--', 
# #                 label=f"Critical degree = {x0_hat:.2f}")
    
# #     # Plot CI if available
# #     if not np.isnan(x0_low):
# #         plt.axvspan(x0_low, x0_high, alpha=0.2, color='red', 
# #                    label=f"95% CI: [{x0_low:.2f}, {x0_high:.2f}]")

# # plt.xlabel("Average Degree")
# # plt.ylabel("P(success)")
# # plt.title(f"Solver Success Phase Transition (n={len(xdata)})")
# # plt.grid(True, alpha=0.3)
# # plt.legend()
# # plt.tight_layout()
# # plt.savefig(os.path.join(OUTPUT_DIR, "logistic_phase_transition.png"), dpi=300)
# # plt.close()

# # print(f"\nCritical degree: {x0_hat:.3f}")
# # if not np.isnan(x0_low):
# #     print(f"95% CI: [{x0_low:.3f}, {x0_high:.3f}]")
# # else:
# #     print("95% CI: Could not compute")

# # # ================================
# # # Plot 4 — Feasibility vs Sparsity
# # # ================================

# # # Bin sparsity with explicit observed parameter
# # bins = np.linspace(df["sparsity"].min(), df["sparsity"].max(), 8)
# # df["sparsity_bin"] = pd.cut(df["sparsity"], bins)

# # # Group with observed=False to suppress warning
# # feas = df.groupby("sparsity_bin", observed=False)["feasible"].mean()
# # bin_centers = [b.mid for b in feas.index.categories]

# # plt.figure()
# # plt.plot(bin_centers, feas.values, marker="o", linewidth=2)
# # plt.xlabel("Sparsity (%)")
# # plt.ylabel("Fraction Feasible")
# # plt.title("Feasibility vs Sparsity")
# # plt.grid(True, alpha=0.3)
# # plt.savefig(os.path.join(OUTPUT_DIR, "feasibility_vs_sparsity.png"), dpi=300)
# # plt.close()

# # # ================================
# # # Additional diagnostic plot: Success rate by avg_degree bins
# # # ================================

# # plt.figure(figsize=(10, 6))
# # # Create equal-sized bins for avg_degree
# # df_sorted = df.sort_values("avg_degree")
# # n_bins = min(10, len(df_sorted) // 5)  # Ensure enough points per bin
# # if n_bins > 1:
# #     df_sorted["degree_bin"] = pd.qcut(df_sorted["avg_degree"], n_bins, duplicates='drop')
# #     success_by_bin = df_sorted.groupby("degree_bin", observed=False)["success"].agg(['mean', 'count'])
    
# #     # Create custom x positions
# #     bin_centers = []
# #     bin_labels = []
# #     for bin_val in success_by_bin.index.categories:
# #         bin_centers.append(bin_val.mid)
# #         bin_labels.append(f"{bin_val.left:.1f}-{bin_val.right:.1f}")
    
# #     plt.bar(range(len(bin_centers)), success_by_bin['mean'].values, alpha=0.7)
# #     plt.xticks(range(len(bin_centers)), bin_labels, rotation=45, ha='right')
    
# #     # Add count labels on bars
# #     for i, (mean_val, count_val) in enumerate(zip(success_by_bin['mean'], success_by_bin['count'])):
# #         plt.text(i, mean_val + 0.02, f"n={count_val}", ha='center', fontsize=8)
    
# #     plt.xlabel("Average Degree (binned)")
# #     plt.ylabel("Success Rate")
# #     plt.title(f"Success Rate by Average Degree Bins ({n_bins} bins)")
# #     plt.ylim(0, 1.1)
# #     plt.grid(True, alpha=0.3, axis='y')
# #     plt.tight_layout()
# #     plt.savefig(os.path.join(OUTPUT_DIR, "success_rate_by_degree_bins.png"), dpi=300)
# #     plt.close()

# # print("\nAll plots saved to:", OUTPUT_DIR)
# # print(f"Total instances: {len(df)}")
# # print(f"Data summary:")
# # print(df[['avg_degree', 'sparsity', 'success', 'feasible', 'gap_rel']].describe())






# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.optimize import curve_fit

# # ============================================================
# # CONFIG
# # ============================================================

# SOLVER_ROOT = "solver_results"      # change for baseline
# PRUNED_ROOT = "pruned_graphs_k"           # change for baseline
# OUTPUT_DIR = "structural_analysis_gnn"

# K_VALUES = [2,3,4,5,6,7,8,9,10,12,15,17,20,22,25]
# N_BOOT = 1000
# SEEDS = list(range(1, 11))
# RNG = np.random.default_rng(42)

# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # ============================================================
# # LOAD DATA
# # ============================================================

# # records = []

# # for k in K_VALUES:
# #     results_path = os.path.join(SOLVER_ROOT, f"k{k}", "results.csv")
# #     if not os.path.exists(results_path):
# #         continue

# #     df_k = pd.read_csv(results_path)

# #     for _, row in df_k.iterrows():
# #         instance = row["instance"]

# #         meta_path = os.path.join(
# #             PRUNED_ROOT, f"k{k}", instance, f"{instance}_metadata.json"
# #         )

# #         if not os.path.exists(meta_path):
# #             continue

# #         meta = pd.read_json(meta_path, typ="series")

# #         records.append({
# #             "instance": instance,
# #             "k": k,
# #             "n": meta["n_nodes"],
# #             "avg_degree": meta["avg_degree"],
# #             "sparsity": meta["sparsity"],
# #             "is_connected": meta["is_connected"],
# #             "feasible": meta["feasible"],
# #             "gap": row["gap_rel_percent"],
# #             "success": row["success"]
# #         })

# # df = pd.DataFrame(records)

# # if len(df) == 0:
# #     raise ValueError("No data found.")

# # df.to_csv(os.path.join(OUTPUT_DIR, "full_structural_dataframe.csv"), index=False)

# # print("Total records:", len(df))

# records = []

# for k in K_VALUES:
#     # Collect all seeds for this k
#     seed_data = {}

#     for seed in SEEDS:
#         results_path = os.path.join(SOLVER_ROOT, f"seed{seed}", f"k{k}", "results.csv")
#         if not os.path.exists(results_path):
#             continue

#         df_k = pd.read_csv(results_path)

#         for _, row in df_k.iterrows():
#             instance = row["instance"]
#             if instance not in seed_data:
#                 seed_data[instance] = {"gaps": [], "successes": []}
#             seed_data[instance]["gaps"].append(row["gap_rel_percent"])
#             seed_data[instance]["successes"].append(row["success"])

#     # Aggregate per instance
#     for instance, data in seed_data.items():
#         meta_path = os.path.join(
#             PRUNED_ROOT, f"k{k}", instance, f"{instance}_metadata.json"
#         )
#         if not os.path.exists(meta_path):
#             continue

#         meta = pd.read_json(meta_path, typ="series")

#         records.append({
#             "instance": instance,
#             "k": k,
#             "n": meta["n_nodes"],
#             "avg_degree": meta["avg_degree"],
#             "sparsity": meta["sparsity"],
#             "is_connected": meta["is_connected"],
#             "feasible": meta["feasible"],
#             "gap": np.mean(data["gaps"]),
#             "success": np.mean(data["successes"]),
#             "n_seeds": len(data["successes"]),
#             "success_std": np.std(data["successes"])
#         })

# df = pd.DataFrame(records)

# if len(df) == 0:
#     raise ValueError("No data found.")

# df.to_csv(os.path.join(OUTPUT_DIR, "full_structural_dataframe.csv"), index=False)

# print(f"Total records: {len(df)}")
# print(f"Seeds aggregated: {len(SEEDS)}")
# print(f"Success values: {sorted(df['success'].unique())}")

# # ============================================================
# # 1️⃣ MACRO LEVEL — SUCCESS & GAP VS k (WITH CI)
# # ============================================================

# macro = df.groupby("k").agg(
#     success_rate=("success", "mean"),
#     mean_gap=("gap", "mean"),
#     std_gap=("gap", "std"),
#     count=("success", "count")
# ).reset_index()

# # Bernoulli standard error
# macro["success_se"] = np.sqrt(
#     macro["success_rate"] * (1 - macro["success_rate"]) / macro["count"]
# )

# macro["success_ci_low"] = macro["success_rate"] - 1.96 * macro["success_se"]
# macro["success_ci_high"] = macro["success_rate"] + 1.96 * macro["success_se"]

# macro.to_csv(os.path.join(OUTPUT_DIR, "macro_k_summary.csv"), index=False)

# # Success vs k
# plt.figure()
# plt.errorbar(
#     macro["k"],
#     macro["success_rate"],
#     yerr=1.96 * macro["success_se"],
#     marker="o"
# )
# plt.xlabel("k")
# plt.ylabel("Success Rate")
# plt.grid(alpha=0.3)
# plt.savefig(os.path.join(OUTPUT_DIR, "success_vs_k.png"), dpi=300)
# plt.close()

# # Gap vs k
# plt.figure()
# plt.errorbar(
#     macro["k"],
#     macro["mean_gap"],
#     yerr=macro["std_gap"],
#     marker="o"
# )
# plt.xlabel("k")
# plt.ylabel("Mean Optimality Gap (%)")
# plt.grid(alpha=0.3)
# plt.savefig(os.path.join(OUTPUT_DIR, "gap_vs_k.png"), dpi=300)
# plt.close()

# # ============================================================
# # 2️⃣ STRUCTURAL VIEW — AVG DEGREE BINNING (WITH CI)
# # ============================================================

# num_bins = 12
# bins = np.linspace(df["avg_degree"].min(), df["avg_degree"].max(), num_bins)
# df["degree_bin"] = pd.cut(df["avg_degree"], bins)

# grouped = df.groupby("degree_bin", observed=False)

# struct = grouped.agg(
#     success_rate=("success", "mean"),
#     mean_gap=("gap", "mean"),
#     std_gap=("gap", "std"),
#     count=("success", "count")
# ).reset_index()

# struct["bin_center"] = struct["degree_bin"].apply(lambda x: x.mid).astype(float)

# struct = struct[struct["count"] >= 5]

# # Bernoulli CI
# struct["success_se"] = np.sqrt(
#     struct["success_rate"] * (1 - struct["success_rate"]) / struct["count"]
# )

# struct["success_ci_low"] = struct["success_rate"] - 1.96 * struct["success_se"]
# struct["success_ci_high"] = struct["success_rate"] + 1.96 * struct["success_se"]

# struct.to_csv(os.path.join(OUTPUT_DIR, "structural_degree_summary.csv"), index=False)

# # Success vs avg_degree
# plt.figure()
# plt.errorbar(
#     struct["bin_center"],
#     struct["success_rate"],
#     yerr=1.96 * struct["success_se"],
#     marker="o"
# )
# plt.xlabel("Average Degree")
# plt.ylabel("Success Rate")
# plt.grid(alpha=0.3)
# plt.savefig(os.path.join(OUTPUT_DIR, "success_vs_avg_degree.png"), dpi=300)
# plt.close()

# # Gap vs avg_degree
# plt.figure()
# plt.errorbar(
#     struct["bin_center"],
#     struct["mean_gap"],
#     yerr=struct["std_gap"],
#     marker="o"
# )
# plt.xlabel("Average Degree")
# plt.ylabel("Mean Optimality Gap (%)")
# plt.grid(alpha=0.3)
# plt.savefig(os.path.join(OUTPUT_DIR, "gap_vs_avg_degree.png"), dpi=300)
# plt.close()

# # ============================================================
# # 3️⃣ LOGISTIC PHASE TRANSITION (RAW BERNOULLI + BOOTSTRAP)
# # ============================================================

# def logistic(x, beta, x0):
#     z = np.clip(beta * (x - x0), -500, 500)
#     return 1 / (1 + np.exp(-z))

# x_raw = df["avg_degree"].values
# y_raw = df["success"].astype(float).values

# # Initial fit
# params, _ = curve_fit(
#     logistic,
#     x_raw,
#     y_raw,
#     p0=(1.0, np.median(x_raw)),
#     maxfev=10000
# )

# beta_hat, x0_hat = params

# # Bootstrap
# beta_samples = []
# x0_samples = []

# for _ in range(N_BOOT):
#     idx = RNG.choice(len(df), len(df), replace=True)
#     x_boot = x_raw[idx]
#     y_boot = y_raw[idx]

#     try:
#         params_boot, _ = curve_fit(
#             logistic,
#             x_boot,
#             y_boot,
#             p0=(beta_hat, x0_hat),
#             maxfev=5000
#         )
#         beta_samples.append(params_boot[0])
#         x0_samples.append(params_boot[1])
#     except:
#         continue

# beta_samples = np.array(beta_samples)
# x0_samples = np.array(x0_samples)

# beta_ci = np.percentile(beta_samples, [2.5, 97.5])
# x0_ci = np.percentile(x0_samples, [2.5, 97.5])

# # Confidence band
# x_plot = np.linspace(min(x_raw), max(x_raw), 300)

# y_boot_curves = [
#     logistic(x_plot, b, x0)
#     for b, x0 in zip(beta_samples, x0_samples)
# ]

# y_boot_curves = np.array(y_boot_curves)

# y_lower = np.percentile(y_boot_curves, 2.5, axis=0)
# y_upper = np.percentile(y_boot_curves, 97.5, axis=0)
# y_plot = logistic(x_plot, beta_hat, x0_hat)

# # Plot
# plt.figure(figsize=(10, 6))

# jitter = RNG.normal(0, 0.015, size=len(y_raw))
# plt.scatter(
#     x_raw,
#     y_raw + jitter,
#     alpha=0.3,
#     s=20,
#     c="steelblue",
#     label="Data points",
#     zorder=3
# )

# plt.plot(x_plot, y_plot, "r-", linewidth=2, label="Logistic fit")

# plt.fill_between(
#     x_plot, y_lower, y_upper,
#     alpha=0.2, color="red",
#     label=f"95% CI"
# )

# plt.axvline(
#     x0_hat, color="red", linestyle="--", alpha=0.7,
#     label=f"Critical degree = {x0_hat:.1f} [{x0_ci[0]:.1f}, {x0_ci[1]:.1f}]"
# )

# plt.xlabel("Average Degree", fontsize=12)
# plt.ylabel("P(success)", fontsize=12)
# plt.title(f"Solver Success Phase Transition (n={len(df)})", fontsize=13)
# plt.legend(fontsize=10, loc="lower right")
# plt.ylim(-0.1, 1.1)
# plt.grid(alpha=0.3)
# plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_DIR, "logistic_phase_transition.png"), dpi=300)
# plt.close()

# # Save logistic parameters
# logistic_df = pd.DataFrame({
#     "beta_hat": [beta_hat],
#     "beta_ci_low": [beta_ci[0]],
#     "beta_ci_high": [beta_ci[1]],
#     "x0_hat": [x0_hat],
#     "x0_ci_low": [x0_ci[0]],
#     "x0_ci_high": [x0_ci[1]]
# })

# logistic_df.to_csv(os.path.join(OUTPUT_DIR, "logistic_parameters.csv"), index=False)

# print("Critical avg degree:", x0_hat)
# print("x0 95% CI:", x0_ci)

# # ============================================================
# # 4️⃣ PER-INSTANCE CRITICAL k + BOOTSTRAP CI
# # ============================================================

# critical_k = []

# for inst, sub in df.groupby("instance"):
#     sub_sorted = sub.sort_values("k")
#     success_ks = sub_sorted[sub_sorted["success"] >= 0.5]["k"]
#     if len(success_ks) > 0:
#         critical_k.append(success_ks.iloc[0])

# critical_df = pd.DataFrame({"critical_k": critical_k})
# critical_df.to_csv(os.path.join(OUTPUT_DIR, "per_instance_critical_k.csv"), index=False)

# # Histogram
# plt.figure()
# plt.hist(critical_df["critical_k"], bins=10)
# plt.xlabel("Critical k")
# plt.ylabel("Number of Instances")
# plt.grid(alpha=0.3)
# plt.savefig(os.path.join(OUTPUT_DIR, "critical_k_distribution.png"), dpi=300)
# plt.close()

# # Bootstrap mean critical k
# mean_samples = []
# ck_values = critical_df["critical_k"].values

# for _ in range(N_BOOT):
#     sample = RNG.choice(ck_values, len(ck_values), replace=True)
#     mean_samples.append(np.mean(sample))

# mean_samples = np.array(mean_samples)
# mean_ci = np.percentile(mean_samples, [2.5, 97.5])

# print("Mean critical k:", np.mean(ck_values))
# print("Mean critical k 95% CI:", mean_ci)

# print("\n=== Structural Navigability Analysis Complete (Statistically Rigorous) ===")


# # ============================================================
# # FINAL SUMMARY
# # ============================================================

# # Count failing instances
# n_total = df["instance"].nunique()
# n_never = n_total - len(critical_k)

# ck_values = np.array(critical_k)

# print("\n" + "=" * 60)
# print("STRUCTURAL ANALYSIS SUMMARY")
# print("=" * 60)
# print(f"Total data points: {len(df)}")
# print(f"Instances: {df['instance'].nunique()}")
# print(f"K values: {sorted(df['k'].unique())}")
# print(f"\nLogistic fit:")
# print(f"  β = {beta_hat:.4f}  CI: [{beta_ci[0]:.4f}, {beta_ci[1]:.4f}]")
# print(f"  d₀ = {x0_hat:.2f}  CI: [{x0_ci[0]:.2f}, {x0_ci[1]:.2f}]")
# print(f"  Bootstrap samples: {len(x0_samples)}/{N_BOOT}")
# print(f"\nCritical k:")
# print(f"  Mean: {np.mean(ck_values):.2f}  CI: [{mean_ci[0]:.2f}, {mean_ci[1]:.2f}]")
# print(f"  Always-failing instances: {n_never}/{n_total}")
# print(f"\nOverall success rate: {df['success'].mean()*100:.1f}%")
# print(f"Success rate at k≥10: {df[df['k']>=10]['success'].mean()*100:.1f}%")
# print(f"\nOutputs saved to: {OUTPUT_DIR}")
# print("=" * 60)









# """
# Unified analysis for GNN pruned modes and baselines.
# Generates structural plots and tables in:
#   plots/single/{mode}_{name}.png
#   plots/master/comparison_{name}.png
#   tables/single/{mode}_{name}.csv
#   tables/master/comparison_{name}.csv
# """

# import os
# import re
# import argparse
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.optimize import curve_fit
# from glob import glob

# # ============================================================
# # CONFIGURATION
# # ============================================================
# SOLVER_RESULTS_DIR = "solver_results"   # where solver outputs are stored
# OUTPUT_ROOT = "."                       # root for plots/tables
# N_BOOT = 1000
# RNG = np.random.default_rng(42)

# # Baseline k values (for filtering, not strictly needed)
# BASELINE_K_VALUES = [5, 10, 15, 20, 25]

# # For GNN modes, k values are auto-detected from data
# # ============================================================
# # Helper functions
# # ============================================================
# def logistic(x, beta, x0):
#     z = np.clip(beta * (x - x0), -500, 500)
#     return 1 / (1 + np.exp(-z))

# def load_mode_data(mode, results_dir=SOLVER_RESULTS_DIR):
#     """
#     Load all seed*k* CSV files for a given mode.
#     mode can be e.g. 'pruned_modeA' or 'pruned_baselines/nearest_k'
#     Returns a DataFrame with columns:
#         instance, k, seed, gap_rel_percent, success (0/1), avg_degree, ...
#     """
#     mode_path = os.path.join(results_dir, mode)
#     if not os.path.isdir(mode_path):
#         raise FileNotFoundError(f"Mode directory not found: {mode_path}")

#     all_files = glob(os.path.join(mode_path, "seed*_k*.csv"))
#     if not all_files:
#         raise ValueError(f"No CSV files found in {mode_path}")

#     dfs = []
#     for f in all_files:
#         # Extract seed and k from filename: seed{seed}_k{k}.csv
#         basename = os.path.basename(f)
#         match = re.search(r'seed(\d+)_k(\d+)', basename)
#         if not match:
#             continue
#         seed = int(match.group(1))
#         k = int(match.group(2))
#         df = pd.read_csv(f)
#         # Add seed and k columns if not already present
#         if 'seed' not in df.columns:
#             df['seed'] = seed
#         if 'k' not in df.columns:
#             df['k'] = k
#         dfs.append(df)

#     if not dfs:
#         raise ValueError(f"No valid seed*k* CSV files in {mode_path}")
#     data = pd.concat(dfs, ignore_index=True)

#     # Ensure success is boolean/float (0/1)
#     if 'tour_found' in data.columns:
#         data['success'] = data['tour_found'].astype(float)
#     elif 'success' in data.columns:
#         data['success'] = data['success'].astype(float)
#     else:
#         raise KeyError("No success column found")

#     # Gap column
#     if 'gap_rel_percent' not in data.columns:
#         if 'gap_rel_percent' in data.columns:
#             pass
#         else:
#             raise KeyError("No gap_rel_percent column found")

#     return data

# def aggregate_by_instance_k(data):
#     """
#     Aggregate across seeds for each (instance, k) pair.
#     Returns DataFrame with:
#         instance, k, n_nodes, avg_degree, sparsity, is_connected, feasible,
#         gap_mean, success_prob, n_seeds, success_std
#     """
#     # Required columns
#     required = ['instance', 'k', 'gap_rel_percent', 'success']
#     for col in required:
#         if col not in data.columns:
#             raise KeyError(f"Missing column: {col}")

#     # Group by instance and k
#     agg = data.groupby(['instance', 'k']).agg(
#         gap_mean=('gap_rel_percent', 'mean'),
#         success_prob=('success', 'mean'),
#         n_seeds=('success', 'count'),
#         success_std=('success', 'std')
#     ).reset_index()

#     # Add metadata from first occurrence (assuming constant per instance,k)
#     meta_cols = ['n_nodes', 'avg_degree', 'sparsity', 'is_connected', 'feasible']
#     meta_present = [c for c in meta_cols if c in data.columns]
#     if meta_present:
#         meta = data.groupby(['instance', 'k'])[meta_present].first().reset_index()
#         agg = agg.merge(meta, on=['instance', 'k'], how='left')

#     return agg

# def compute_logistic_fit(df, avg_degree_col='avg_degree', success_col='success_prob'):
#     """Fit logistic curve and bootstrap CI. Returns dict of results."""
#     x = df[avg_degree_col].values
#     y = df[success_col].values

#     # Remove NaN or infinite
#     mask = np.isfinite(x) & np.isfinite(y)
#     x = x[mask]
#     y = y[mask]

#     if len(x) < 10:
#         raise ValueError("Not enough data points for logistic fit")

#     # Initial fit
#     try:
#         params, _ = curve_fit(logistic, x, y, p0=(1.0, np.median(x)), maxfev=10000)
#     except Exception as e:
#         raise RuntimeError(f"Logistic fit failed: {e}")

#     beta_hat, x0_hat = params

#     # Bootstrap
#     beta_samples = []
#     x0_samples = []
#     n = len(x)
#     for _ in range(N_BOOT):
#         idx = RNG.choice(n, n, replace=True)
#         x_boot = x[idx]
#         y_boot = y[idx]
#         try:
#             p_boot, _ = curve_fit(logistic, x_boot, y_boot,
#                                   p0=(beta_hat, x0_hat), maxfev=5000)
#             beta_samples.append(p_boot[0])
#             x0_samples.append(p_boot[1])
#         except:
#             continue

#     beta_samples = np.array(beta_samples)
#     x0_samples = np.array(x0_samples)

#     beta_ci = np.percentile(beta_samples, [2.5, 97.5])
#     x0_ci = np.percentile(x0_samples, [2.5, 97.5])

#     # Confidence band
#     x_plot = np.linspace(x.min(), x.max(), 300)
#     y_plot = logistic(x_plot, beta_hat, x0_hat)
#     y_boot_curves = [logistic(x_plot, b, x0) for b, x0 in zip(beta_samples, x0_samples)]
#     y_boot_curves = np.array(y_boot_curves)
#     y_lower = np.percentile(y_boot_curves, 2.5, axis=0)
#     y_upper = np.percentile(y_boot_curves, 97.5, axis=0)

#     return {
#         'beta_hat': beta_hat, 'beta_ci': beta_ci,
#         'x0_hat': x0_hat, 'x0_ci': x0_ci,
#         'x_plot': x_plot, 'y_plot': y_plot, 'y_lower': y_lower, 'y_upper': y_upper,
#         'x_data': x, 'y_data': y
#     }

# def plot_success_vs_k(agg_df, mode_name, output_dir):
#     """Success rate vs k with CI."""
#     macro = agg_df.groupby('k').agg(
#         success_rate=('success_prob', 'mean'),
#         count=('success_prob', 'count')
#     ).reset_index()
#     macro['se'] = np.sqrt(macro['success_rate'] * (1 - macro['success_rate']) / macro['count'])
#     macro['ci_low'] = macro['success_rate'] - 1.96 * macro['se']
#     macro['ci_high'] = macro['success_rate'] + 1.96 * macro['se']

#     plt.figure()
#     plt.errorbar(macro['k'], macro['success_rate'], yerr=1.96*macro['se'],
#                  marker='o', capsize=3)
#     plt.xlabel('k')
#     plt.ylabel('Success Rate')
#     plt.title(f'{mode_name} - Success vs k')
#     plt.grid(alpha=0.3)
#     plt.tight_layout()
#     out_path = os.path.join(output_dir, f"{mode_name}_success_vs_k.png")
#     plt.savefig(out_path, dpi=300)
#     plt.close()
#     # Save table
#     table_path = os.path.join(output_dir.replace('plots', 'tables'),
#                               f"{mode_name}_success_vs_k.csv")
#     os.makedirs(os.path.dirname(table_path), exist_ok=True)
#     macro.to_csv(table_path, index=False)

# def plot_gap_vs_k(agg_df, mode_name, output_dir):
#     """Mean gap vs k with std error."""
#     macro = agg_df.groupby('k').agg(
#         mean_gap=('gap_mean', 'mean'),
#         std_gap=('gap_mean', 'std'),
#         count=('gap_mean', 'count')
#     ).reset_index()
#     plt.figure()
#     plt.errorbar(macro['k'], macro['mean_gap'], yerr=macro['std_gap'],
#                  marker='o', capsize=3)
#     plt.xlabel('k')
#     plt.ylabel('Mean Optimality Gap (%)')
#     plt.title(f'{mode_name} - Gap vs k')
#     plt.grid(alpha=0.3)
#     plt.tight_layout()
#     out_path = os.path.join(output_dir, f"{mode_name}_gap_vs_k.png")
#     plt.savefig(out_path, dpi=300)
#     plt.close()
#     table_path = os.path.join(output_dir.replace('plots', 'tables'),
#                               f"{mode_name}_gap_vs_k.csv")
#     os.makedirs(os.path.dirname(table_path), exist_ok=True)
#     macro.to_csv(table_path, index=False)

# def plot_success_vs_degree(agg_df, mode_name, output_dir):
#     """Binned success rate vs avg degree with CI."""
#     if 'avg_degree' not in agg_df.columns:
#         print(f"  No avg_degree column for {mode_name}, skipping degree plot")
#         return
#     # Bin by avg_degree
#     num_bins = min(12, agg_df['avg_degree'].nunique())
#     bins = np.linspace(agg_df['avg_degree'].min(), agg_df['avg_degree'].max(), num_bins)
#     agg_df['degree_bin'] = pd.cut(agg_df['avg_degree'], bins)
#     grouped = agg_df.groupby('degree_bin', observed=False).agg(
#         success_rate=('success_prob', 'mean'),
#         count=('success_prob', 'count')
#     ).reset_index()
#     grouped['bin_center'] = grouped['degree_bin'].apply(lambda x: x.mid)
#     grouped = grouped[grouped['count'] >= 5]
#     if len(grouped) == 0:
#         return
#     grouped['se'] = np.sqrt(grouped['success_rate'] * (1 - grouped['success_rate']) / grouped['count'])
#     plt.figure()
#     plt.errorbar(grouped['bin_center'], grouped['success_rate'], yerr=1.96*grouped['se'],
#                  marker='o', capsize=3)
#     plt.xlabel('Average Degree')
#     plt.ylabel('Success Rate')
#     plt.title(f'{mode_name} - Success vs Average Degree')
#     plt.grid(alpha=0.3)
#     plt.tight_layout()
#     out_path = os.path.join(output_dir, f"{mode_name}_success_vs_avg_degree.png")
#     plt.savefig(out_path, dpi=300)
#     plt.close()
#     table_path = os.path.join(output_dir.replace('plots', 'tables'),
#                               f"{mode_name}_success_vs_avg_degree.csv")
#     os.makedirs(os.path.dirname(table_path), exist_ok=True)
#     grouped.to_csv(table_path, index=False)

# def plot_gap_vs_degree(agg_df, mode_name, output_dir):
#     if 'avg_degree' not in agg_df.columns:
#         return
#     num_bins = min(12, agg_df['avg_degree'].nunique())
#     bins = np.linspace(agg_df['avg_degree'].min(), agg_df['avg_degree'].max(), num_bins)
#     agg_df['degree_bin'] = pd.cut(agg_df['avg_degree'], bins)
#     grouped = agg_df.groupby('degree_bin', observed=False).agg(
#         mean_gap=('gap_mean', 'mean'),
#         std_gap=('gap_mean', 'std'),
#         count=('gap_mean', 'count')
#     ).reset_index()
#     grouped['bin_center'] = grouped['degree_bin'].apply(lambda x: x.mid)
#     grouped = grouped[grouped['count'] >= 5]
#     if len(grouped) == 0:
#         return
#     plt.figure()
#     plt.errorbar(grouped['bin_center'], grouped['mean_gap'], yerr=grouped['std_gap'],
#                  marker='o', capsize=3)
#     plt.xlabel('Average Degree')
#     plt.ylabel('Mean Optimality Gap (%)')
#     plt.title(f'{mode_name} - Gap vs Average Degree')
#     plt.grid(alpha=0.3)
#     plt.tight_layout()
#     out_path = os.path.join(output_dir, f"{mode_name}_gap_vs_avg_degree.png")
#     plt.savefig(out_path, dpi=300)
#     plt.close()
#     table_path = os.path.join(output_dir.replace('plots', 'tables'),
#                               f"{mode_name}_gap_vs_avg_degree.csv")
#     os.makedirs(os.path.dirname(table_path), exist_ok=True)
#     grouped.to_csv(table_path, index=False)

# def plot_logistic_phase_transition(agg_df, mode_name, output_dir):
#     """Logistic fit on success_prob vs avg_degree."""
#     if 'avg_degree' not in agg_df.columns:
#         print(f"  No avg_degree for {mode_name}, skip logistic")
#         return
#     try:
#         fit = compute_logistic_fit(agg_df)
#     except Exception as e:
#         print(f"  Logistic fit failed for {mode_name}: {e}")
#         return

#     plt.figure(figsize=(10,6))
#     # Jitter raw data
#     y_raw = agg_df['success_prob'].values
#     x_raw = agg_df['avg_degree'].values
#     jitter = RNG.normal(0, 0.015, size=len(y_raw))
#     plt.scatter(x_raw, y_raw + jitter, alpha=0.3, s=20, c='steelblue', label='Data')
#     # Fit curve
#     plt.plot(fit['x_plot'], fit['y_plot'], 'r-', lw=2, label='Logistic fit')
#     plt.fill_between(fit['x_plot'], fit['y_lower'], fit['y_upper'],
#                      alpha=0.2, color='red', label='95% CI')
#     plt.axvline(fit['x0_hat'], color='red', linestyle='--', alpha=0.7,
#                 label=f"Critical degree = {fit['x0_hat']:.1f} [{fit['x0_ci'][0]:.1f}, {fit['x0_ci'][1]:.1f}]")
#     plt.xlabel('Average Degree')
#     plt.ylabel('P(success)')
#     plt.title(f'{mode_name} - Phase Transition')
#     plt.ylim(-0.1, 1.1)
#     plt.grid(alpha=0.3)
#     plt.legend(loc='lower right')
#     plt.tight_layout()
#     out_path = os.path.join(output_dir, f"{mode_name}_phase_transition.png")
#     plt.savefig(out_path, dpi=300)
#     plt.close()
#     # Save logistic parameters
#     table_dir = output_dir.replace('plots', 'tables')
#     os.makedirs(table_dir, exist_ok=True)
#     log_df = pd.DataFrame([{
#         'mode': mode_name,
#         'beta_hat': fit['beta_hat'],
#         'beta_ci_low': fit['beta_ci'][0],
#         'beta_ci_high': fit['beta_ci'][1],
#         'x0_hat': fit['x0_hat'],
#         'x0_ci_low': fit['x0_ci'][0],
#         'x0_ci_high': fit['x0_ci'][1]
#     }])
#     log_df.to_csv(os.path.join(table_dir, f"{mode_name}_logistic_params.csv"), index=False)

# def plot_critical_k_distribution(agg_df, mode_name, output_dir):
#     """Per-instance critical k: smallest k where success_prob >= 0.5."""
#     crit = []
#     for inst, grp in agg_df.groupby('instance'):
#         grp_sorted = grp.sort_values('k')
#         high = grp_sorted[grp_sorted['success_prob'] >= 0.5]
#         if not high.empty:
#             crit.append(high.iloc[0]['k'])
#     if not crit:
#         print(f"  No critical k found for {mode_name}")
#         return
#     crit = np.array(crit)
#     plt.figure()
#     plt.hist(crit, bins=min(20, len(np.unique(crit))))
#     plt.xlabel('Critical k')
#     plt.ylabel('Number of Instances')
#     plt.title(f'{mode_name} - Critical k Distribution')
#     plt.grid(alpha=0.3)
#     plt.tight_layout()
#     out_path = os.path.join(output_dir, f"{mode_name}_critical_k_dist.png")
#     plt.savefig(out_path, dpi=300)
#     plt.close()
#     # Bootstrap mean
#     mean_samples = [np.mean(RNG.choice(crit, len(crit), replace=True)) for _ in range(N_BOOT)]
#     mean_ci = np.percentile(mean_samples, [2.5, 97.5])
#     summary = pd.DataFrame([{
#         'mode': mode_name,
#         'mean_critical_k': np.mean(crit),
#         'ci_low': mean_ci[0],
#         'ci_high': mean_ci[1],
#         'n_instances': len(crit),
#         'total_instances': agg_df['instance'].nunique()
#     }])
#     table_dir = output_dir.replace('plots', 'tables')
#     os.makedirs(table_dir, exist_ok=True)
#     summary.to_csv(os.path.join(table_dir, f"{mode_name}_critical_k_summary.csv"), index=False)
#     # Also save per-instance values
#     inst_df = pd.DataFrame({'instance': list(set(crit)), 'critical_k': crit})
#     inst_df.to_csv(os.path.join(table_dir, f"{mode_name}_per_instance_critical_k.csv"), index=False)

# # ============================================================
# # Master comparison functions
# # ============================================================
# def compare_success_vs_k(all_dfs, mode_names, master_plot_dir, master_table_dir):
#     """Plot success vs k for multiple modes on same axes."""
#     plt.figure(figsize=(10,6))
#     for name, df in zip(mode_names, all_dfs):
#         macro = df.groupby('k').agg(
#             success_rate=('success_prob', 'mean'),
#             count=('success_prob', 'count')
#         ).reset_index()
#         macro['se'] = np.sqrt(macro['success_rate'] * (1 - macro['success_rate']) / macro['count'])
#         plt.errorbar(macro['k'], macro['success_rate'], yerr=1.96*macro['se'],
#                      marker='o', capsize=3, label=name)
#     plt.xlabel('k')
#     plt.ylabel('Success Rate')
#     plt.title('Comparison: Success vs k')
#     plt.legend()
#     plt.grid(alpha=0.3)
#     plt.tight_layout()
#     out_path = os.path.join(master_plot_dir, "comparison_success_vs_k.png")
#     plt.savefig(out_path, dpi=300)
#     plt.close()
#     # Table: combine macro summaries
#     combined = []
#     for name, df in zip(mode_names, all_dfs):
#         macro = df.groupby('k').agg(
#             success_rate=('success_prob', 'mean'),
#             count=('success_prob', 'count')
#         ).reset_index()
#         macro['mode'] = name
#         combined.append(macro)
#     comb_df = pd.concat(combined, ignore_index=True)
#     comb_df.to_csv(os.path.join(master_table_dir, "comparison_success_vs_k.csv"), index=False)

# def compare_gap_vs_k(all_dfs, mode_names, master_plot_dir, master_table_dir):
#     plt.figure(figsize=(10,6))
#     for name, df in zip(mode_names, all_dfs):
#         macro = df.groupby('k').agg(
#             mean_gap=('gap_mean', 'mean'),
#             std_gap=('gap_mean', 'std')
#         ).reset_index()
#         plt.errorbar(macro['k'], macro['mean_gap'], yerr=macro['std_gap'],
#                      marker='o', capsize=3, label=name)
#     plt.xlabel('k')
#     plt.ylabel('Mean Optimality Gap (%)')
#     plt.title('Comparison: Gap vs k')
#     plt.legend()
#     plt.grid(alpha=0.3)
#     plt.tight_layout()
#     out_path = os.path.join(master_plot_dir, "comparison_gap_vs_k.png")
#     plt.savefig(out_path, dpi=300)
#     plt.close()
#     combined = []
#     for name, df in zip(mode_names, all_dfs):
#         macro = df.groupby('k').agg(
#             mean_gap=('gap_mean', 'mean'),
#             std_gap=('gap_mean', 'std'),
#             count=('gap_mean', 'count')
#         ).reset_index()
#         macro['mode'] = name
#         combined.append(macro)
#     comb_df = pd.concat(combined, ignore_index=True)
#     comb_df.to_csv(os.path.join(master_table_dir, "comparison_gap_vs_k.csv"), index=False)

# def compare_logistic_fits(all_dfs, mode_names, master_plot_dir, master_table_dir):
#     """Plot multiple logistic curves on same axes."""
#     plt.figure(figsize=(10,6))
#     colors = plt.cm.tab10(np.linspace(0, 1, len(mode_names)))
#     fit_results = []
#     for name, df, color in zip(mode_names, all_dfs, colors):
#         if 'avg_degree' not in df.columns:
#             continue
#         try:
#             fit = compute_logistic_fit(df)
#             plt.plot(fit['x_plot'], fit['y_plot'], color=color, lw=2, label=name)
#             plt.axvline(fit['x0_hat'], color=color, linestyle='--', alpha=0.5)
#             fit_results.append({'mode': name, **{k: v for k, v in fit.items() if not k.startswith('x_') and not k.startswith('y_')}})
#         except Exception as e:
#             print(f"Logistic fit failed for {name}: {e}")

#     plt.xlabel('Average Degree')
#     plt.ylabel('P(success)')
#     plt.title('Comparison: Phase Transition')
#     plt.ylim(-0.1, 1.1)
#     plt.grid(alpha=0.3)
#     plt.legend()
#     plt.tight_layout()
#     out_path = os.path.join(master_plot_dir, "comparison_phase_transition.png")
#     plt.savefig(out_path, dpi=300)
#     plt.close()
#     # Save parameters
#     if fit_results:
#         param_df = pd.DataFrame(fit_results)
#         # Keep only scalar parameters
#         param_df = param_df[['mode', 'beta_hat', 'x0_hat', 'beta_ci', 'x0_ci']].copy()
#         param_df.to_csv(os.path.join(master_table_dir, "comparison_logistic_params.csv"), index=False)

# # ============================================================
# # Main
# # ============================================================
# def main():
#     parser = argparse.ArgumentParser(description="Analyse GNN pruned modes and baselines")
#     parser.add_argument("--modes", nargs="+", required=True,
#                         help="List of mode names as in solver_results/ (e.g., pruned_modeA pruned_baselines/nearest_k)")
#     parser.add_argument("--output_root", default=".",
#                         help="Root directory where 'plots' and 'tables' will be created")
#     parser.add_argument("--solver_results", default="solver_results",
#                         help="Directory containing solver outputs")
#     args = parser.parse_args()

#     # Create directory structure
#     plots_single = os.path.join(args.output_root, "plots", "single")
#     plots_master = os.path.join(args.output_root, "plots", "master")
#     tables_single = os.path.join(args.output_root, "tables", "single")
#     tables_master = os.path.join(args.output_root, "tables", "master")
#     for d in [plots_single, plots_master, tables_single, tables_master]:
#         os.makedirs(d, exist_ok=True)

#     # Load data for each mode
#     mode_dfs = []  # list of aggregated DataFrames (instance,k level)
#     mode_names = []
#     for mode in args.modes:
#         print(f"\nProcessing mode: {mode}")
#         try:
#             raw = load_mode_data(mode, results_dir=args.solver_results)
#             agg = aggregate_by_instance_k(raw)
#             mode_dfs.append(agg)
#             mode_names.append(mode)
#         except Exception as e:
#             print(f"  Failed to load {mode}: {e}")
#             continue

#     if not mode_dfs:
#         print("No modes loaded successfully.")
#         return

#     # Generate single-mode plots/tables
#     for name, df in zip(mode_names, mode_dfs):
#         print(f"  Generating single outputs for {name}")
#         # Use a safe filename (replace / with _)
#         safe_name = name.replace('/', '_')
#         plot_success_vs_k(df, safe_name, plots_single)
#         plot_gap_vs_k(df, safe_name, plots_single)
#         plot_success_vs_degree(df, safe_name, plots_single)
#         plot_gap_vs_degree(df, safe_name, plots_single)
#         plot_logistic_phase_transition(df, safe_name, plots_single)
#         plot_critical_k_distribution(df, safe_name, plots_single)

#     # Generate master comparisons (if more than one mode)
#     if len(mode_dfs) >= 2:
#         print("\nGenerating master comparison plots/tables")
#         compare_success_vs_k(mode_dfs, mode_names, plots_master, tables_master)
#         compare_gap_vs_k(mode_dfs, mode_names, plots_master, tables_master)
#         compare_logistic_fits(mode_dfs, mode_names, plots_master, tables_master)

#     print("\n✅ Analysis complete.")
#     print(f"   Single plots: {plots_single}")
#     print(f"   Master plots: {plots_master}")
#     print(f"   Tables: {tables_single} and {tables_master}")

# if __name__ == "__main__":
#     main()

"""
Unified analysis for GNN pruned modes and baselines.
Loads solver results from solver_results/ and structural metadata from original pruning directories.
Generates plots/tables in:
  plots/single/{mode}_{name}.png
  plots/master/comparison_{name}.png
  tables/single/{mode}_{name}.csv
  tables/master/comparison_{name}.csv
"""

import os
import re
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from glob import glob

# ============================================================
# CONFIGURATION
# ============================================================
SOLVER_RESULTS_DIR = "solver_results"   # where solver outputs are stored
PRUNED_ROOT = "."                       # root containing pruned_modeA, pruned_modeB, pruned_baselines/, etc.
N_BOOT = 1000
RNG = np.random.default_rng(42)

# ============================================================
# Helper functions
# ============================================================
def logistic(x, beta, x0):
    z = np.clip(beta * (x - x0), -500, 500)
    return 1 / (1 + np.exp(-z))

def load_metadata(instance, k, mode):
    """
    Load metadata from original pruning directory.
    mode: e.g. 'pruned_modeA' or 'pruned_baselines/nearest_k'
    Returns dict with avg_degree, sparsity, n_nodes, etc.
    """
    # Build path to metadata.json
    if mode.startswith("pruned_baselines/"):
        base = os.path.join(PRUNED_ROOT, mode)  # e.g. pruned_baselines/nearest_k
    else:
        base = os.path.join(PRUNED_ROOT, mode)  # e.g. pruned_modeA
    meta_path = os.path.join(base, f"k{k}", instance, f"{instance}_metadata.json")
    if not os.path.exists(meta_path):
        # Try alternative: maybe stored in pruned_graphs_k (legacy)
        alt_path = os.path.join(PRUNED_ROOT, "pruned_graphs_k", f"k{k}", instance, f"{instance}_metadata.json")
        if os.path.exists(alt_path):
            meta_path = alt_path
        else:
            return None
    try:
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        # Extract relevant fields
        return {
            'n_nodes': meta.get('n_nodes', None),
            'avg_degree': meta.get('avg_degree', None),
            'sparsity': meta.get('sparsity', None),
            'is_connected': meta.get('is_connected', None),
            'feasible': meta.get('feasible', None),
        }
    except Exception as e:
        print(f"Warning: Could not load metadata for {instance}, k={k}, mode={mode}: {e}")
        return None

def load_mode_data(mode, results_dir=SOLVER_RESULTS_DIR):
    """
    Load all seed*k* CSV files for a given mode.
    Returns a DataFrame with columns:
        instance, k, seed, gap_rel_percent, success (0/1)
    """
    mode_path = os.path.join(results_dir, mode)
    if not os.path.isdir(mode_path):
        raise FileNotFoundError(f"Mode directory not found: {mode_path}")

    all_files = glob(os.path.join(mode_path, "seed*_k*.csv"))
    if not all_files:
        raise ValueError(f"No CSV files found in {mode_path}")

    dfs = []
    for f in all_files:
        basename = os.path.basename(f)
        match = re.search(r'seed(\d+)_k(\d+)', basename)
        if not match:
            continue
        seed = int(match.group(1))
        k = int(match.group(2))
        df = pd.read_csv(f)
        # Ensure required columns
        if 'instance' not in df.columns:
            raise KeyError(f"Missing 'instance' column in {f}")
        if 'gap_rel_percent' not in df.columns:
            raise KeyError(f"Missing 'gap_rel_percent' column in {f}")
        # success column may be 'tour_found' or 'success'
        if 'tour_found' in df.columns:
            df['success'] = df['tour_found'].astype(float)
        elif 'success' in df.columns:
            df['success'] = df['success'].astype(float)
        else:
            raise KeyError(f"No success indicator column in {f}")
        df['seed'] = seed
        df['k'] = k
        dfs.append(df)

    if not dfs:
        raise ValueError(f"No valid seed*k* CSV files in {mode_path}")
    data = pd.concat(dfs, ignore_index=True)
    return data

def aggregate_by_instance_k(data, mode):
    """
    Aggregate across seeds for each (instance, k) pair.
    Also loads structural metadata from pruning directories.
    Returns DataFrame with:
        instance, k, n_nodes, avg_degree, sparsity, is_connected, feasible,
        gap_mean, success_prob, n_seeds, success_std
    """
    # Aggregate solver results
    agg = data.groupby(['instance', 'k']).agg(
        gap_mean=('gap_rel_percent', 'mean'),
        success_prob=('success', 'mean'),
        n_seeds=('success', 'count'),
        success_std=('success', 'std')
    ).reset_index()

    # Add structural metadata
    meta_rows = []
    for idx, row in agg.iterrows():
        inst = row['instance']
        k = row['k']
        meta = load_metadata(inst, k, mode)
        if meta is None:
            # Still add row but with NaN for structural fields
            meta = {col: np.nan for col in ['n_nodes', 'avg_degree', 'sparsity', 'is_connected', 'feasible']}
        meta_rows.append(meta)
    meta_df = pd.DataFrame(meta_rows)
    agg = pd.concat([agg, meta_df], axis=1)
    return agg

# ============================================================
# Plotting and table generation functions (single mode)
# ============================================================
def plot_success_vs_k(agg_df, mode_name, output_dir):
    macro = agg_df.groupby('k').agg(
        success_rate=('success_prob', 'mean'),
        count=('success_prob', 'count')
    ).reset_index()
    macro['se'] = np.sqrt(macro['success_rate'] * (1 - macro['success_rate']) / macro['count'])
    macro['ci_low'] = macro['success_rate'] - 1.96 * macro['se']
    macro['ci_high'] = macro['success_rate'] + 1.96 * macro['se']

    plt.figure()
    plt.errorbar(macro['k'], macro['success_rate'], yerr=1.96*macro['se'],
                 marker='o', capsize=3)
    plt.xlabel('k')
    plt.ylabel('Success Rate')
    plt.title(f'{mode_name} - Success vs k')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{mode_name}_success_vs_k.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    # Save table
    table_dir = output_dir.replace('plots', 'tables')
    os.makedirs(table_dir, exist_ok=True)
    macro.to_csv(os.path.join(table_dir, f"{mode_name}_success_vs_k.csv"), index=False)

def plot_gap_vs_k(agg_df, mode_name, output_dir):
    macro = agg_df.groupby('k').agg(
        mean_gap=('gap_mean', 'mean'),
        std_gap=('gap_mean', 'std'),
        count=('gap_mean', 'count')
    ).reset_index()
    plt.figure()
    plt.errorbar(macro['k'], macro['mean_gap'], yerr=macro['std_gap'],
                 marker='o', capsize=3)
    plt.xlabel('k')
    plt.ylabel('Mean Optimality Gap (%)')
    plt.title(f'{mode_name} - Gap vs k')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{mode_name}_gap_vs_k.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    table_dir = output_dir.replace('plots', 'tables')
    os.makedirs(table_dir, exist_ok=True)
    macro.to_csv(os.path.join(table_dir, f"{mode_name}_gap_vs_k.csv"), index=False)

def plot_success_vs_degree(agg_df, mode_name, output_dir):
    if 'avg_degree' not in agg_df.columns or agg_df['avg_degree'].isnull().all():
        print(f"  No avg_degree column for {mode_name}, skipping degree plot")
        return
    # Remove NaN
    df_clean = agg_df.dropna(subset=['avg_degree', 'success_prob'])
    if len(df_clean) < 5:
        print(f"  Not enough data for degree plot for {mode_name}")
        return
    num_bins = min(12, df_clean['avg_degree'].nunique())
    bins = np.linspace(df_clean['avg_degree'].min(), df_clean['avg_degree'].max(), num_bins)
    df_clean['degree_bin'] = pd.cut(df_clean['avg_degree'], bins)
    grouped = df_clean.groupby('degree_bin', observed=False).agg(
        success_rate=('success_prob', 'mean'),
        count=('success_prob', 'count')
    ).reset_index()
    grouped['bin_center'] = grouped['degree_bin'].apply(lambda x: x.mid)
    grouped = grouped[grouped['count'] >= 5]
    if len(grouped) == 0:
        return
    grouped['se'] = np.sqrt(grouped['success_rate'] * (1 - grouped['success_rate']) / grouped['count'])
    plt.figure()
    plt.errorbar(grouped['bin_center'], grouped['success_rate'], yerr=1.96*grouped['se'],
                 marker='o', capsize=3)
    plt.xlabel('Average Degree')
    plt.ylabel('Success Rate')
    plt.title(f'{mode_name} - Success vs Average Degree')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{mode_name}_success_vs_avg_degree.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    table_dir = output_dir.replace('plots', 'tables')
    os.makedirs(table_dir, exist_ok=True)
    grouped.to_csv(os.path.join(table_dir, f"{mode_name}_success_vs_avg_degree.csv"), index=False)

def plot_gap_vs_degree(agg_df, mode_name, output_dir):
    if 'avg_degree' not in agg_df.columns or agg_df['avg_degree'].isnull().all():
        return
    df_clean = agg_df.dropna(subset=['avg_degree', 'gap_mean'])
    if len(df_clean) < 5:
        return
    num_bins = min(12, df_clean['avg_degree'].nunique())
    bins = np.linspace(df_clean['avg_degree'].min(), df_clean['avg_degree'].max(), num_bins)
    df_clean['degree_bin'] = pd.cut(df_clean['avg_degree'], bins)
    grouped = df_clean.groupby('degree_bin', observed=False).agg(
        mean_gap=('gap_mean', 'mean'),
        std_gap=('gap_mean', 'std'),
        count=('gap_mean', 'count')
    ).reset_index()
    grouped['bin_center'] = grouped['degree_bin'].apply(lambda x: x.mid)
    grouped = grouped[grouped['count'] >= 5]
    if len(grouped) == 0:
        return
    plt.figure()
    plt.errorbar(grouped['bin_center'], grouped['mean_gap'], yerr=grouped['std_gap'],
                 marker='o', capsize=3)
    plt.xlabel('Average Degree')
    plt.ylabel('Mean Optimality Gap (%)')
    plt.title(f'{mode_name} - Gap vs Average Degree')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{mode_name}_gap_vs_avg_degree.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    table_dir = output_dir.replace('plots', 'tables')
    os.makedirs(table_dir, exist_ok=True)
    grouped.to_csv(os.path.join(table_dir, f"{mode_name}_gap_vs_avg_degree.csv"), index=False)

def compute_logistic_fit(df, avg_degree_col='avg_degree', success_col='success_prob'):
    x = df[avg_degree_col].values
    y = df[success_col].values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 10:
        raise ValueError("Not enough data points for logistic fit")
    try:
        params, _ = curve_fit(logistic, x, y, p0=(1.0, np.median(x)), maxfev=10000)
    except Exception as e:
        raise RuntimeError(f"Logistic fit failed: {e}")
    beta_hat, x0_hat = params
    # Bootstrap
    beta_samples = []
    x0_samples = []
    n = len(x)
    for _ in range(N_BOOT):
        idx = RNG.choice(n, n, replace=True)
        x_boot = x[idx]
        y_boot = y[idx]
        try:
            p_boot, _ = curve_fit(logistic, x_boot, y_boot, p0=(beta_hat, x0_hat), maxfev=5000)
            beta_samples.append(p_boot[0])
            x0_samples.append(p_boot[1])
        except:
            continue
    beta_samples = np.array(beta_samples)
    x0_samples = np.array(x0_samples)
    beta_ci = np.percentile(beta_samples, [2.5, 97.5])
    x0_ci = np.percentile(x0_samples, [2.5, 97.5])
    # Confidence band
    x_plot = np.linspace(x.min(), x.max(), 300)
    y_plot = logistic(x_plot, beta_hat, x0_hat)
    y_boot_curves = [logistic(x_plot, b, x0) for b, x0 in zip(beta_samples, x0_samples)]
    y_boot_curves = np.array(y_boot_curves)
    y_lower = np.percentile(y_boot_curves, 2.5, axis=0)
    y_upper = np.percentile(y_boot_curves, 97.5, axis=0)
    return {
        'beta_hat': beta_hat, 'beta_ci': beta_ci,
        'x0_hat': x0_hat, 'x0_ci': x0_ci,
        'x_plot': x_plot, 'y_plot': y_plot, 'y_lower': y_lower, 'y_upper': y_upper,
        'x_data': x, 'y_data': y
    }

def plot_logistic_phase_transition(agg_df, mode_name, output_dir):
    if 'avg_degree' not in agg_df.columns or agg_df['avg_degree'].isnull().all():
        print(f"  No avg_degree for {mode_name}, skip logistic")
        return
    df_clean = agg_df.dropna(subset=['avg_degree', 'success_prob'])
    if len(df_clean) < 10:
        print(f"  Not enough data for logistic fit for {mode_name}")
        return
    try:
        fit = compute_logistic_fit(df_clean)
    except Exception as e:
        print(f"  Logistic fit failed for {mode_name}: {e}")
        return
    plt.figure(figsize=(10,6))
    # Jitter raw data
    y_raw = df_clean['success_prob'].values
    x_raw = df_clean['avg_degree'].values
    jitter = RNG.normal(0, 0.015, size=len(y_raw))
    plt.scatter(x_raw, y_raw + jitter, alpha=0.3, s=20, c='steelblue', label='Data')
    plt.plot(fit['x_plot'], fit['y_plot'], 'r-', lw=2, label='Logistic fit')
    plt.fill_between(fit['x_plot'], fit['y_lower'], fit['y_upper'],
                     alpha=0.2, color='red', label='95% CI')
    plt.axvline(fit['x0_hat'], color='red', linestyle='--', alpha=0.7,
                label=f"Critical degree = {fit['x0_hat']:.1f} [{fit['x0_ci'][0]:.1f}, {fit['x0_ci'][1]:.1f}]")
    plt.xlabel('Average Degree')
    plt.ylabel('P(success)')
    plt.title(f'{mode_name} - Phase Transition')
    plt.ylim(-0.1, 1.1)
    plt.grid(alpha=0.3)
    plt.legend(loc='lower right')
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{mode_name}_phase_transition.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    # Save logistic parameters
    table_dir = output_dir.replace('plots', 'tables')
    os.makedirs(table_dir, exist_ok=True)
    log_df = pd.DataFrame([{
        'mode': mode_name,
        'beta_hat': fit['beta_hat'],
        'beta_ci_low': fit['beta_ci'][0],
        'beta_ci_high': fit['beta_ci'][1],
        'x0_hat': fit['x0_hat'],
        'x0_ci_low': fit['x0_ci'][0],
        'x0_ci_high': fit['x0_ci'][1]
    }])
    log_df.to_csv(os.path.join(table_dir, f"{mode_name}_logistic_params.csv"), index=False)

def plot_critical_k_distribution(agg_df, mode_name, output_dir):
    """Per-instance critical k: smallest k where success_prob >= 0.5."""
    crit = []
    for inst, grp in agg_df.groupby('instance'):
        grp_sorted = grp.sort_values('k')
        high = grp_sorted[grp_sorted['success_prob'] >= 0.5]
        if not high.empty:
            crit.append(high.iloc[0]['k'])
    if not crit:
        print(f"  No critical k found for {mode_name}")
        return
    crit = np.array(crit)
    plt.figure()
    plt.hist(crit, bins=min(20, len(np.unique(crit))))
    plt.xlabel('Critical k')
    plt.ylabel('Number of Instances')
    plt.title(f'{mode_name} - Critical k Distribution')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{mode_name}_critical_k_dist.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    # Bootstrap mean
    mean_samples = [np.mean(RNG.choice(crit, len(crit), replace=True)) for _ in range(N_BOOT)]
    mean_ci = np.percentile(mean_samples, [2.5, 97.5])
    summary = pd.DataFrame([{
        'mode': mode_name,
        'mean_critical_k': np.mean(crit),
        'ci_low': mean_ci[0],
        'ci_high': mean_ci[1],
        'n_instances': len(crit),
        'total_instances': agg_df['instance'].nunique()
    }])
    table_dir = output_dir.replace('plots', 'tables')
    os.makedirs(table_dir, exist_ok=True)
    summary.to_csv(os.path.join(table_dir, f"{mode_name}_critical_k_summary.csv"), index=False)
    # Save per-instance values (with correct lengths)
    inst_list = [inst for inst, grp in agg_df.groupby('instance') if not grp[grp['success_prob'] >= 0.5].empty]
    # inst_list length must match crit length, because we appended in the same loop
    # Ensure we have the same ordering
    inst_list = []
    for inst, grp in agg_df.groupby('instance'):
        grp_sorted = grp.sort_values('k')
        high = grp_sorted[grp_sorted['success_prob'] >= 0.5]
        if not high.empty:
            inst_list.append(inst)
    # Now inst_list and crit have same length
    if len(inst_list) == len(crit):
        inst_df = pd.DataFrame({'instance': inst_list, 'critical_k': crit})
        inst_df.to_csv(os.path.join(table_dir, f"{mode_name}_per_instance_critical_k.csv"), index=False)
    else:
        print(f"  Mismatch in critical k per-instance recording for {mode_name}")

# ============================================================
# Master comparison functions
# ============================================================
def compare_success_vs_k(all_dfs, mode_names, master_plot_dir, master_table_dir):
    plt.figure(figsize=(10,6))
    for name, df in zip(mode_names, all_dfs):
        macro = df.groupby('k').agg(
            success_rate=('success_prob', 'mean'),
            count=('success_prob', 'count')
        ).reset_index()
        macro['se'] = np.sqrt(macro['success_rate'] * (1 - macro['success_rate']) / macro['count'])
        plt.errorbar(macro['k'], macro['success_rate'], yerr=1.96*macro['se'],
                     marker='o', capsize=3, label=name)
    plt.xlabel('k')
    plt.ylabel('Success Rate')
    plt.title('Comparison: Success vs k')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(master_plot_dir, "comparison_success_vs_k.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    # Table
    combined = []
    for name, df in zip(mode_names, all_dfs):
        macro = df.groupby('k').agg(
            success_rate=('success_prob', 'mean'),
            count=('success_prob', 'count')
        ).reset_index()
        macro['mode'] = name
        combined.append(macro)
    comb_df = pd.concat(combined, ignore_index=True)
    comb_df.to_csv(os.path.join(master_table_dir, "comparison_success_vs_k.csv"), index=False)

def compare_gap_vs_k(all_dfs, mode_names, master_plot_dir, master_table_dir):
    plt.figure(figsize=(10,6))
    for name, df in zip(mode_names, all_dfs):
        macro = df.groupby('k').agg(
            mean_gap=('gap_mean', 'mean'),
            std_gap=('gap_mean', 'std')
        ).reset_index()
        plt.errorbar(macro['k'], macro['mean_gap'], yerr=macro['std_gap'],
                     marker='o', capsize=3, label=name)
    plt.xlabel('k')
    plt.ylabel('Mean Optimality Gap (%)')
    plt.title('Comparison: Gap vs k')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(master_plot_dir, "comparison_gap_vs_k.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    combined = []
    for name, df in zip(mode_names, all_dfs):
        macro = df.groupby('k').agg(
            mean_gap=('gap_mean', 'mean'),
            std_gap=('gap_mean', 'std'),
            count=('gap_mean', 'count')
        ).reset_index()
        macro['mode'] = name
        combined.append(macro)
    comb_df = pd.concat(combined, ignore_index=True)
    comb_df.to_csv(os.path.join(master_table_dir, "comparison_gap_vs_k.csv"), index=False)

def compare_logistic_fits(all_dfs, mode_names, master_plot_dir, master_table_dir):
    plt.figure(figsize=(10,6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(mode_names)))
    fit_results = []
    for name, df, color in zip(mode_names, all_dfs, colors):
        if 'avg_degree' not in df.columns or df['avg_degree'].isnull().all():
            continue
        df_clean = df.dropna(subset=['avg_degree', 'success_prob'])
        if len(df_clean) < 10:
            continue
        try:
            fit = compute_logistic_fit(df_clean)
            plt.plot(fit['x_plot'], fit['y_plot'], color=color, lw=2, label=name)
            plt.axvline(fit['x0_hat'], color=color, linestyle='--', alpha=0.5)
            fit_results.append({'mode': name, 'x0_hat': fit['x0_hat'], 'x0_ci_low': fit['x0_ci'][0], 'x0_ci_high': fit['x0_ci'][1]})
        except Exception as e:
            print(f"Logistic fit failed for {name}: {e}")
    if not fit_results:
        return
    plt.xlabel('Average Degree')
    plt.ylabel('P(success)')
    plt.title('Comparison: Phase Transition')
    plt.ylim(-0.1, 1.1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(master_plot_dir, "comparison_phase_transition.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    param_df = pd.DataFrame(fit_results)
    param_df.to_csv(os.path.join(master_table_dir, "comparison_logistic_params.csv"), index=False)

# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Analyse GNN pruned modes and baselines")
    parser.add_argument("--modes", nargs="+", required=True,
                        help="List of mode names as in solver_results/ (e.g., pruned_modeA pruned_baselines/nearest_k)")
    parser.add_argument("--output_root", default=".",
                        help="Root directory where 'plots' and 'tables' will be created")
    parser.add_argument("--solver_results", default="solver_results",
                        help="Directory containing solver outputs")
    parser.add_argument("--pruned_root", default=".",
                        help="Root directory containing pruned_modeX and pruned_baselines folders")
    args = parser.parse_args()

    # Set global PRUNED_ROOT
    global PRUNED_ROOT
    PRUNED_ROOT = args.pruned_root

    # Create directory structure
    plots_single = os.path.join(args.output_root, "plots", "single")
    plots_master = os.path.join(args.output_root, "plots", "master")
    tables_single = os.path.join(args.output_root, "tables", "single")
    tables_master = os.path.join(args.output_root, "tables", "master")
    for d in [plots_single, plots_master, tables_single, tables_master]:
        os.makedirs(d, exist_ok=True)

    # Load data for each mode
    mode_dfs = []
    mode_names = []
    for mode in args.modes:
        print(f"\nProcessing mode: {mode}")
        try:
            raw = load_mode_data(mode, results_dir=args.solver_results)
            agg = aggregate_by_instance_k(raw, mode)
            mode_dfs.append(agg)
            mode_names.append(mode)
        except Exception as e:
            print(f"  Failed to load {mode}: {e}")
            continue

    if not mode_dfs:
        print("No modes loaded successfully.")
        return

    # Generate single-mode outputs
    for name, df in zip(mode_names, mode_dfs):
        print(f"  Generating single outputs for {name}")
        safe_name = name.replace('/', '_')
        plot_success_vs_k(df, safe_name, plots_single)
        plot_gap_vs_k(df, safe_name, plots_single)
        plot_success_vs_degree(df, safe_name, plots_single)
        plot_gap_vs_degree(df, safe_name, plots_single)
        plot_logistic_phase_transition(df, safe_name, plots_single)
        plot_critical_k_distribution(df, safe_name, plots_single)

    # Master comparisons if more than one mode
    if len(mode_dfs) >= 2:
        print("\nGenerating master comparison plots/tables")
        compare_success_vs_k(mode_dfs, mode_names, plots_master, tables_master)
        compare_gap_vs_k(mode_dfs, mode_names, plots_master, tables_master)
        compare_logistic_fits(mode_dfs, mode_names, plots_master, tables_master)

    print("\n✅ Analysis complete.")
    print(f"   Single plots: {plots_single}")
    print(f"   Master plots: {plots_master}")
    print(f"   Tables: {tables_single} and {tables_master}")

if __name__ == "__main__":
    main()
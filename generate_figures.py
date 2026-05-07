import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import expit

# ======================== CONFIGURATION ========================
RESULTS_ROOT = "pipeline_results"
OUTPUT_DIR = "figures_final"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FOLDER_MAP = {
    "pruned_ModeA": ("GNN-A", "gnn"),
    "pruned_ModeB": ("GNN-B", "gnn"),
    "pruned_ModeC": ("GNN-C", "gnn"),
    "pruned_ModeD": ("GNN-D", "gnn"),
    "delaunay_k":  ("Delaunay", "baseline"),
    "nearest_k":   ("kNN", "baseline"),
    "random_k":    ("Random", "baseline"),
}

LKH_SEEDS = [1, 3, 6, 8]
SUCCESS_THRESHOLD = 1.0          # gap ≤ 1 % → success
N_BOOT = 1000
RNG = np.random.default_rng(42)

COLORS = {
    "GNN-A": "#1f77b4", "GNN-B": "#ff7f0e", "GNN-C": "#2ca02c",
    "GNN-D": "#d62728", "Delaunay": "#9467bd", "kNN": "#8c564b",
    "Random": "#7f7f7f"
}
METHOD_ORDER = ["GNN-A", "GNN-B", "GNN-C", "GNN-D", "Delaunay", "kNN", "Random"]

# ======================== DATA LOADING ========================
def load_lkh(method_path, method_name):
    fpath = os.path.join(RESULTS_ROOT, method_path, "lkh_results.csv")
    if not os.path.exists(fpath):
        return pd.DataFrame()
    df = pd.read_csv(fpath)
    df = df[df["seed"].isin(LKH_SEEDS)]
    df["success"] = df["gap_rel_percent"].abs() <= SUCCESS_THRESHOLD
    agg = df.groupby(["instance", "k"]).agg(
        mean_gap=("gap_rel_percent", "mean"),
        success_rate=("success", "mean"),
        mean_runtime=("runtime_pruned", "mean"),
        n_seeds=("seed", "count")
    ).reset_index()
    agg["method"] = method_name
    return agg

def load_structural(method_path, method_name):
    folder = os.path.join(RESULTS_ROOT, method_path)
    frames = []
    for fname in sorted(os.listdir(folder)):
        if fname.startswith("summary_k") and fname.endswith(".csv"):
            k = int(fname.replace("summary_k", "").replace(".csv", ""))
            df = pd.read_csv(os.path.join(folder, fname))
            df = df[~df["instance_name"].str.contains("AVERAGE", na=False)]
            numeric_cols = [
                "n_nodes", "original_edges", "kept_edges", "sparsity",
                "avg_degree", "min_degree", "max_degree", "avg_knn_overlap",
                "tour_edge_recall", "is_connected", "num_connected_components",
                "mean_edge_length_full", "mean_edge_length_pruned",
                "length_reduction_ratio", "tour_edges_kept", "tour_edges_total"
            ]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            df["k"] = k
            df["method"] = method_name
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

records = []
for folder, (display_name, group) in FOLDER_MAP.items():
    print(f"Processing {display_name}...")
    lkh_df = load_lkh(folder, display_name)
    struct_df = load_structural(folder, display_name)
    if lkh_df.empty or struct_df.empty:
        continue
    merged = pd.merge(
        lkh_df, struct_df,
        left_on=["instance", "k", "method"],
        right_on=["instance_name", "k", "method"],
        how="inner"
    )
    records.append(merged)

data_all = pd.concat(records, ignore_index=True)
print(f"Total merged records: {len(data_all)}")

# Identify best GNN mode
gnn_modes = [name for name, grp in FOLDER_MAP.values() if grp == "gnn"]
best_gnn = None
if gnn_modes:
    gnn_data = data_all[data_all["method"].isin(gnn_modes)]
    best_gnn = gnn_data.groupby("method")["success_rate"].mean().idxmax()
    print(f"Best GNN mode: {best_gnn}")

# ======================== HELPER: BINNED SUMMARY ========================
def binned_summary(data, x_col, y_col, num_bins=12):
    x = data[x_col].dropna()
    y = data[y_col].dropna()
    mask = x.notna() & y.notna()
    x, y = x[mask], y[mask]
    if len(x) < 5:
        return pd.DataFrame()
    bins = np.linspace(x.min(), x.max(), num_bins)
    bin_labels = pd.cut(x, bins)
    df_bin = pd.DataFrame({"bin": bin_labels, "x": x, "y": y})
    grouped = df_bin.groupby("bin", observed=False)
    summary = grouped.agg(
        mean_y=("y", "mean"),
        std_y=("y", "std"),
        count=("y", "count")
    ).reset_index()
    summary["bin_center"] = summary["bin"].apply(lambda b: b.mid)
    summary["se"] = summary["std_y"] / np.sqrt(summary["count"])
    return summary.dropna()

def logistic(x, beta, x0):
    return expit(np.clip(beta * (x - x0), -500, 500))

def fit_logistic(data, x_col="avg_degree", y_col="success_rate", n_boot=N_BOOT):
    x = data[x_col].values
    y = data[y_col].values
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 5:
        return None
    try:
        params, _ = curve_fit(logistic, x, y, p0=[1.0, np.median(x)], maxfev=10000)
    except Exception:
        return None
    beta_hat, x0_hat = params
    beta_samples, x0_samples = [], []
    for _ in range(n_boot):
        idx = RNG.choice(len(x), len(x), replace=True)
        try:
            p_boot, _ = curve_fit(logistic, x[idx], y[idx],
                                  p0=[beta_hat, x0_hat], maxfev=5000)
            beta_samples.append(p_boot[0])
            x0_samples.append(p_boot[1])
        except:
            pass
    if len(beta_samples) < 50:
        return None
    beta_samples = np.array(beta_samples)
    x0_samples = np.array(x0_samples)
    beta_ci = np.percentile(beta_samples, [2.5, 97.5])
    x0_ci = np.percentile(x0_samples, [2.5, 97.5])
    x_plot = np.linspace(x.min(), x.max(), 200)
    y_curves = np.array([logistic(x_plot, b, x0) for b, x0 in zip(beta_samples, x0_samples)])
    y_lower = np.percentile(y_curves, 2.5, axis=0)
    y_upper = np.percentile(y_curves, 97.5, axis=0)
    y_plot = logistic(x_plot, beta_hat, x0_hat)
    return {
        "beta_hat": beta_hat, "x0_hat": x0_hat,
        "beta_ci": beta_ci, "x0_ci": x0_ci,
        "x_plot": x_plot, "y_plot": y_plot,
        "y_lower": y_lower, "y_upper": y_upper
    }

# ======================== PLOT STYLING ========================
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "none"
})

def get_method_subset(exclude_random=False):
    """
    Returns list of methods for main figures:
    best GNN + Delaunay + kNN [+ Random if not excluded].
    Ensures no duplicates.
    """
    methods = []
    if best_gnn:
        methods.append(best_gnn)
    for m in METHOD_ORDER:
        if m in gnn_modes:
            continue          # all GNN modes except the best (already added) are skipped
        if m == "Random" and exclude_random:
            continue
        if m not in data_all["method"].unique():
            continue
        methods.append(m)
    return methods

# ======================== FIGURE 1 ========================
print("\n" + "="*50)
print("FIGURE 1: Recall vs Average Degree")
fig, ax = plt.subplots(figsize=(8,6))
data_1 = []
for method in get_method_subset(exclude_random=False):
    sub = data_all[data_all["method"] == method]
    summ = binned_summary(sub, "avg_degree", "tour_edge_recall")
    if not summ.empty:
        summ["method"] = method
        data_1.append(summ)
        ax.errorbar(summ["bin_center"], summ["mean_y"], yerr=1.96*summ["se"],
                    fmt='o-', capsize=3, color=COLORS.get(method, "gray"),
                    alpha=0.3 if method=="Random" else 1.0, label=method)
ax.set_xlabel("Average Degree"); ax.set_ylabel("Tour Edge Recall")
ax.set_title("Recall vs Average Degree")  # clean title
ax.grid(True, alpha=0.3); ax.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig1_recall_vs_avg_degree.png"))
plt.close()
fig1_df = pd.concat(data_1)
print("Binned data:")
print(fig1_df[["method", "bin_center", "mean_y", "se", "count"]].to_string(index=False))

# ======================== FIGURE 2 ========================
print("\n" + "="*50)
print("FIGURE 2: Connectivity % vs Average Degree")
fig, ax = plt.subplots(figsize=(8,6))
data_2 = []
for method in get_method_subset(exclude_random=False):
    sub = data_all[data_all["method"] == method]
    summ = binned_summary(sub, "avg_degree", "is_connected")
    if not summ.empty:
        summ["method"] = method
        data_2.append(summ)
        ax.errorbar(summ["bin_center"], summ["mean_y"], yerr=1.96*summ["se"],
                    fmt='o-', capsize=3, color=COLORS.get(method, "gray"),
                    alpha=0.3 if method=="Random" else 1.0, label=method)
ax.set_xlabel("Average Degree"); ax.set_ylabel("Connectivity (%)")
ax.set_title("Connectivity % vs Average Degree")
ax.grid(True, alpha=0.3); ax.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig2_connectivity_vs_avg_degree.png"))
plt.close()
fig2_df = pd.concat(data_2)
print("Binned data:")
print(fig2_df[["method", "bin_center", "mean_y", "se", "count"]].to_string(index=False))

# ======================== FIGURE 3 ========================
print("\n" + "="*50)
print("FIGURE 3: Edge Length Reduction vs Average Degree")
fig, ax = plt.subplots(figsize=(8,6))
data_3 = []
for method in get_method_subset(exclude_random=True):   # Random excluded
    sub = data_all[data_all["method"] == method]
    summ = binned_summary(sub, "avg_degree", "length_reduction_ratio")
    if not summ.empty:
        summ["method"] = method
        data_3.append(summ)
        ax.errorbar(summ["bin_center"], summ["mean_y"], yerr=1.96*summ["se"],
                    fmt='o-', capsize=3, color=COLORS.get(method, "gray"), label=method)
ax.set_xlabel("Average Degree"); ax.set_ylabel("Edge Length Ratio (Pruned/Full)")
ax.set_title("Edge Length Reduction vs Average Degree")
ax.grid(True, alpha=0.3); ax.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig3_edgelength_vs_avg_degree.png"))
plt.close()
fig3_df = pd.concat(data_3)
print("Binned data:")
print(fig3_df[["method", "bin_center", "mean_y", "se", "count"]].to_string(index=False))

# ======================== FIGURE 4 ========================
print("\n" + "="*50)
print("FIGURE 4: KNN Overlap vs Average Degree")
fig, ax = plt.subplots(figsize=(8,6))
data_4 = []
for method in get_method_subset(exclude_random=True):
    sub = data_all[data_all["method"] == method]
    summ = binned_summary(sub, "avg_degree", "avg_knn_overlap")
    if not summ.empty:
        summ["method"] = method
        data_4.append(summ)
        ax.errorbar(summ["bin_center"], summ["mean_y"], yerr=1.96*summ["se"],
                    fmt='o-', capsize=3, color=COLORS.get(method, "gray"), label=method)
ax.set_xlabel("Average Degree"); ax.set_ylabel("KNN Overlap")
ax.set_title("KNN Overlap vs Average Degree")
ax.grid(True, alpha=0.3); ax.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig4_knn_overlap_vs_avg_degree.png"))
plt.close()
fig4_df = pd.concat(data_4)
print("Binned data:")
print(fig4_df[["method", "bin_center", "mean_y", "se", "count"]].to_string(index=False))

# ======================== FIGURE 5 ========================
print("\n" + "="*50)
print("FIGURE 5: Sparsity vs Average Degree (optional)")
fig, ax = plt.subplots(figsize=(8,6))
data_5 = []
for method in get_method_subset(exclude_random=True):
    sub = data_all[data_all["method"] == method]
    summ = binned_summary(sub, "avg_degree", "sparsity")
    if not summ.empty:
        summ["method"] = method
        data_5.append(summ)
        ax.errorbar(summ["bin_center"], summ["mean_y"], yerr=1.96*summ["se"],
                    fmt='o-', capsize=3, color=COLORS.get(method, "gray"), label=method)
ax.set_xlabel("Average Degree"); ax.set_ylabel("Sparsity (%)")
ax.set_title("Sparsity vs Average Degree")
ax.grid(True, alpha=0.3); ax.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig5_sparsity_vs_avg_degree.png"))
plt.close()
fig5_df = pd.concat(data_5)
print("Binned data:")
print(fig5_df[["method", "bin_center", "mean_y", "se", "count"]].to_string(index=False))

# ======================== FIGURE 6 ========================
print("\n" + "="*50)
print("FIGURE 6: Runtime vs Average Degree")
fig, ax = plt.subplots(figsize=(8,6))
data_6 = []
for method in get_method_subset(exclude_random=False):
    sub = data_all[data_all["method"] == method]
    summ = binned_summary(sub, "avg_degree", "mean_runtime")
    if not summ.empty:
        summ["method"] = method
        data_6.append(summ)
        ax.errorbar(summ["bin_center"], summ["mean_y"], yerr=1.96*summ["se"],
                    fmt='o-', capsize=3, color=COLORS.get(method, "gray"),
                    alpha=0.3 if method=="Random" else 1.0, label=method)
ax.set_xlabel("Average Degree"); ax.set_ylabel("Mean Runtime (s)")
ax.set_title("Runtime vs Average Degree")
ax.grid(True, alpha=0.3); ax.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig6_runtime_vs_avg_degree.png"))
plt.close()
fig6_df = pd.concat(data_6)
print("Binned data:")
print(fig6_df[["method", "bin_center", "mean_y", "se", "count"]].to_string(index=False))

# ======================== FIGURE 7 (boxplot) ========================
print("\n" + "="*50)
print("FIGURE 7: Runtime Distribution Boxplots")
methods_box = get_method_subset(exclude_random=True)   # best GNN, Delaunay, kNN
fig, ax = plt.subplots(figsize=(8,6))
box_data = []; labels = []
stats = {}
for method in methods_box:
    runtimes = data_all[data_all["method"] == method]["mean_runtime"].dropna()
    if len(runtimes) > 0:
        box_data.append(runtimes.values)
        labels.append(method)
        stats[method] = runtimes.describe(percentiles=[.25,.5,.75])
ax.boxplot(box_data, labels=labels, patch_artist=True,
           boxprops=dict(facecolor='lightblue', alpha=0.7),
           medianprops=dict(color='red', linewidth=2))
ax.set_xlabel("Method"); ax.set_ylabel("Runtime (s)")
ax.set_title("Runtime Distribution")
ax.grid(True, alpha=0.3); ax.set_xticklabels(labels, rotation=30); plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig7_runtime_boxplot.png"))
plt.close()
for m in labels:
    print(f"\n{m} runtime stats:\n{stats[m]}\n")

# ======================== FIGURE 8 (boxplot) ========================
print("\n" + "="*50)
print("FIGURE 8: Gap Distribution Boxplots")
fig, ax = plt.subplots(figsize=(8,6))
gap_data = []; labels = []
gap_stats = {}
for method in methods_box:      # same subset
    gaps = data_all[data_all["method"] == method]["mean_gap"].dropna()
    if len(gaps) > 0:
        gap_data.append(gaps.values)
        labels.append(method)
        gap_stats[method] = gaps.describe(percentiles=[.25,.5,.75])
ax.boxplot(gap_data, labels=labels, patch_artist=True,
           boxprops=dict(facecolor='lightgreen', alpha=0.7),
           medianprops=dict(color='red', linewidth=2))
ax.set_xlabel("Method"); ax.set_ylabel("Gap (%)")
ax.set_title("Gap Distribution")
ax.grid(True, alpha=0.3); ax.set_xticklabels(labels, rotation=30); plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig8_gap_boxplot.png"))
plt.close()
for m in labels:
    print(f"\n{m} gap stats:\n{gap_stats[m]}\n")

# ======================== FIGURE 9 ========================
print("\n" + "="*50)
print("FIGURE 9: Success Probability vs Average Degree")
fig, ax = plt.subplots(figsize=(8,6))
data_9 = []
for method in get_method_subset(exclude_random=False):
    sub = data_all[data_all["method"] == method]
    summ = binned_summary(sub, "avg_degree", "success_rate")
    if not summ.empty:
        summ["method"] = method
        data_9.append(summ)
        ax.errorbar(summ["bin_center"], summ["mean_y"], yerr=1.96*summ["se"],
                    fmt='o-', capsize=3, color=COLORS.get(method, "gray"),
                    alpha=0.3 if method=="Random" else 1.0, label=method)
ax.set_xlabel("Average Degree"); ax.set_ylabel("Success Probability")
ax.set_title("Success vs Average Degree")
ax.grid(True, alpha=0.3); ax.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig9_success_vs_avg_degree.png"))
plt.close()
fig9_df = pd.concat(data_9)
print("Binned data:")
print(fig9_df[["method", "bin_center", "mean_y", "se", "count"]].to_string(index=False))

# ======================== FIGURE 10 ========================
print("\n" + "="*50)
print("FIGURE 10: Phase Transition Logistic Fits")
fig, ax = plt.subplots(figsize=(10,7))
fit_results = {}
print("Logistic fit parameters:")
for method in get_method_subset(exclude_random=True):
    sub = data_all[data_all["method"] == method].dropna(subset=["avg_degree", "success_rate"])
    if sub.empty: continue
    ax.scatter(sub["avg_degree"], sub["success_rate"], alpha=0.2, s=20, color=COLORS.get(method, "gray"))
    fit = fit_logistic(sub)
    if fit:
        fit_results[method] = fit
        ax.plot(fit["x_plot"], fit["y_plot"], color=COLORS.get(method, "gray"), linewidth=2,
                label=f"{method}")
        ax.fill_between(fit["x_plot"], fit["y_lower"], fit["y_upper"],
                        color=COLORS.get(method, "gray"), alpha=0.15)
        print(f"{method}  β_hat={fit['beta_hat']:.3f}  x0_hat={fit['x0_hat']:.2f}  β_CI=[{fit['beta_ci'][0]:.3f}, {fit['beta_ci'][1]:.3f}]  x0_CI=[{fit['x0_ci'][0]:.2f}, {fit['x0_ci'][1]:.2f}]")
ax.set_xlabel("Average Degree"); ax.set_ylabel("Success Probability")
ax.set_title("Phase Transition Logistic Fits")
ax.grid(True, alpha=0.3); ax.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig10_logistic_phase_transition.png"))
plt.close()

# ======================== FIGURE 11 ========================
print("\n" + "="*50)
print("FIGURE 11: Success Overlay (Money Figure)")
fig, ax = plt.subplots(figsize=(10,7))
for method in get_method_subset(exclude_random=True):
    sub = data_all[data_all["method"] == method]
    summ = binned_summary(sub, "avg_degree", "success_rate")
    if not summ.empty:
        ax.errorbar(summ["bin_center"], summ["mean_y"], yerr=1.96*summ["se"],
                    fmt='o', capsize=3, color=COLORS.get(method, "gray"), alpha=0.8,
                    label=f"{method} binned")
    if method in fit_results:
        f = fit_results[method]
        ax.plot(f["x_plot"], f["y_plot"], color=COLORS.get(method, "gray"), linewidth=3, label=f"{method} fit")
ax.set_xlabel("Average Degree"); ax.set_ylabel("Success Probability")
ax.set_title("Success Overlay (Logistic)")
ax.grid(True, alpha=0.3); ax.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig11_success_overlay.png"))
plt.close()
print("(data same as Figure 9 binned + Figure 10 logistic parameters)")

# ======================== FIGURE 12 ========================
print("\n" + "="*50)
print("FIGURE 12: Connectivity vs Navigability Transition")
fig, ax = plt.subplots(figsize=(8,6))
for method in get_method_subset(exclude_random=True):
    sub = data_all[data_all["method"] == method]
    conn = binned_summary(sub, "avg_degree", "is_connected")
    succ = binned_summary(sub, "avg_degree", "success_rate")
    color = COLORS.get(method, "gray")
    if not conn.empty:
        ax.plot(conn["bin_center"], conn["mean_y"], 's--', color=color, alpha=0.7, label=f"{method} conn")
    if not succ.empty:
        ax.plot(succ["bin_center"], succ["mean_y"], 'o-', color=color, linewidth=2, label=f"{method} succ")
    if not conn.empty:
        print(f"\n{method} Connectivity bins:")
        print(conn[["bin_center", "mean_y", "se"]].to_string(index=False))
    if not succ.empty:
        print(f"{method} Success bins:")
        print(succ[["bin_center", "mean_y", "se"]].to_string(index=False))
ax.set_xlabel("Average Degree"); ax.set_ylabel("Fraction")
ax.set_title("Connectivity vs Navigability Transition")
ax.grid(True, alpha=0.3); ax.legend(fontsize=8, ncol=2); plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig12_connectivity_vs_navigability.png"))
plt.close()

# ======================== FIGURE 13 ========================
print("\n" + "="*50)
print("FIGURE 13: Critical k Distribution")
critical_k_records = []
for method in get_method_subset(exclude_random=True):
    sub = data_all[data_all["method"] == method].sort_values(["instance", "k"])
    for inst, grp in sub.groupby("instance"):
        reached = grp[grp["success_rate"] >= 0.5]
        if not reached.empty:
            crit_k = reached["k"].iloc[0]
            critical_k_records.append({"method": method, "instance": inst, "critical_k": crit_k})
crit_df = pd.DataFrame(critical_k_records)
if not crit_df.empty:
    fig, ax = plt.subplots(figsize=(8,6))
    methods_crit = crit_df["method"].unique()
    box_data = [crit_df[crit_df["method"] == m]["critical_k"].values for m in methods_crit]
    ax.boxplot(box_data, labels=methods_crit, patch_artist=True,
               boxprops=dict(facecolor='lightyellow', alpha=0.7))
    ax.set_xlabel("Method"); ax.set_ylabel("Critical k")
    ax.set_title("Critical k Distribution")
    ax.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig13_critical_k_distribution.png"))
    plt.close()
    print("Per-method critical k quartiles:")
    print(crit_df.groupby("method")["critical_k"].describe(percentiles=[.25,.5,.75]))
else:
    print("No instances reached 50% success - critical k not computed.")

# ======================== FIGURE 14 ========================
print("\n" + "="*50)
print("FIGURE 14: Sparsity vs Gap")
fig, ax = plt.subplots(figsize=(8,6))
data_14 = []
for method in get_method_subset(exclude_random=True):
    sub = data_all[data_all["method"] == method]
    summ = binned_summary(sub, "sparsity", "mean_gap")
    if not summ.empty:
        summ["method"] = method
        data_14.append(summ)
        ax.errorbar(summ["bin_center"], summ["mean_y"], yerr=1.96*summ["se"],
                    fmt='o-', capsize=3, color=COLORS.get(method, "gray"), label=method)
ax.set_xlabel("Sparsity (%)"); ax.set_ylabel("Mean Gap (%)")
ax.set_title("Sparsity vs Gap")
ax.grid(True, alpha=0.3); ax.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig14_sparsity_vs_gap.png"))
plt.close()
fig14_df = pd.concat(data_14)
print("Binned data:")
print(fig14_df[["method", "bin_center", "mean_y", "se", "count"]].to_string(index=False))

# ======================== ABLATION FIGURES ========================
if len(gnn_modes) >= 2:
    # --- Ablation A ---
    print("\n" + "="*50)
    print("ABLATION FIGURE A: Phase Transition per GNN Mode")
    fig, ax = plt.subplots(figsize=(10,7))
    ablation_fits = {}
    for mode in gnn_modes:
        sub = data_all[data_all["method"] == mode].dropna(subset=["avg_degree", "success_rate"])
        if sub.empty: continue
        ax.scatter(sub["avg_degree"], sub["success_rate"], alpha=0.15, s=20, color=COLORS.get(mode, "gray"))
        fit = fit_logistic(sub)
        if fit:
            ablation_fits[mode] = fit
            ax.plot(fit["x_plot"], fit["y_plot"], color=COLORS.get(mode, "gray"), linewidth=2,
                    label=mode)   # <-- added label
            ax.fill_between(fit["x_plot"], fit["y_lower"], fit["y_upper"],
                            color=COLORS.get(mode, "gray"), alpha=0.1)
            print(f"{mode}  β={fit['beta_hat']:.3f}  x0={fit['x0_hat']:.2f}  x0_CI=[{fit['x0_ci'][0]:.2f}, {fit['x0_ci'][1]:.2f}]")
    ax.set_xlabel("Average Degree"); ax.set_ylabel("Success Probability")
    ax.set_title("Ablation A: Phase Transition per GNN Mode")
    ax.grid(True, alpha=0.3)
    ax.legend()   # <-- now legend will show
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "ablation_figA_phase_transition_gnn.png"))
    plt.close()

    # --- Ablation Table B ---
    print("\n" + "="*50)
    print("ABLATION TABLE B: Feature Study")
    table_rows = []
    for mode in gnn_modes:
        sub = data_all[data_all["method"] == mode]
        avg_success = sub["success_rate"].mean() * 100
        avg_gap = sub["mean_gap"].mean()
        avg_runtime = sub["mean_runtime"].mean()
        if mode in ablation_fits:
            crit_deg = ablation_fits[mode]["x0_hat"]
        else:
            fit_tmp = fit_logistic(sub.dropna(subset=["avg_degree", "success_rate"]))
            crit_deg = fit_tmp["x0_hat"] if fit_tmp else np.nan
        table_rows.append({
            "GNN Mode": mode,
            "Success %": f"{avg_success:.1f}",
            "Mean Gap %": f"{avg_gap:.2f}",
            "Mean Runtime (s)": f"{avg_runtime:.3f}",
            "Critical Degree": f"{crit_deg:.1f}" if not np.isnan(crit_deg) else "N/A"
        })
    table_df = pd.DataFrame(table_rows)
    print(table_df.to_string(index=False))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    tbl = ax.table(cellText=table_df.values, colLabels=table_df.columns,
                   cellLoc='center', loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(10)
    tbl.scale(1.2, 1.5)
    for i in range(len(table_df.columns)):
        tbl[0, i].set_facecolor('#4472C4')
        tbl[0, i].set_text_props(weight='bold', color='white')
    plt.title("Ablation Table B: Feature Study", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "ablation_tableB_feature_study.png"))
    plt.close()
    table_df.to_csv(os.path.join(OUTPUT_DIR, "ablation_tableB_feature_study.csv"), index=False)

    # --- Ablation C: Runtime vs Degree ---
    print("\n" + "="*50)
    print("ABLATION FIGURE C: Runtime vs Degree (GNN Modes)")
    fig, ax = plt.subplots(figsize=(8,6))
    for mode in gnn_modes:
        sub = data_all[data_all["method"] == mode]
        summ = binned_summary(sub, "avg_degree", "mean_runtime")
        if not summ.empty:
            summ["method"] = mode
            print(f"{mode} binned runtime:")
            print(summ[["bin_center", "mean_y", "se", "count"]].to_string(index=False))
            ax.errorbar(summ["bin_center"], summ["mean_y"], yerr=1.96*summ["se"],
                        fmt='o-', capsize=3, color=COLORS.get(mode, "gray"), label=mode)
    ax.set_xlabel("Average Degree"); ax.set_ylabel("Mean Runtime (s)")
    ax.set_title("Runtime vs Degree (GNN Modes)")
    ax.grid(True, alpha=0.3); ax.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "ablation_figC_runtime_vs_degree_gnn.png"))
    plt.close()

    # --- Ablation D: Recall vs Degree ---
    print("\n" + "="*50)
    print("ABLATION FIGURE D: Recall vs Degree (GNN Modes)")
    fig, ax = plt.subplots(figsize=(8,6))
    for mode in gnn_modes:
        sub = data_all[data_all["method"] == mode]
        summ = binned_summary(sub, "avg_degree", "tour_edge_recall")
        if not summ.empty:
            summ["method"] = mode
            print(f"{mode} binned recall:")
            print(summ[["bin_center", "mean_y", "se", "count"]].to_string(index=False))
            ax.errorbar(summ["bin_center"], summ["mean_y"], yerr=1.96*summ["se"],
                        fmt='o-', capsize=3, color=COLORS.get(mode, "gray"), label=mode)
    ax.set_xlabel("Average Degree"); ax.set_ylabel("Tour Edge Recall")
    ax.set_title("Recall vs Degree (GNN Modes)")
    ax.grid(True, alpha=0.3); ax.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "ablation_figD_recall_vs_degree_gnn.png"))
    plt.close()

print("\nAll figures and numeric summaries saved in:", OUTPUT_DIR)
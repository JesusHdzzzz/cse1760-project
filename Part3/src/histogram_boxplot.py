import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("images/superconductivty-data/train.csv")

def plot_target_distribution(df, target_col, save=False, save_prefix="target_dist"):
    """
    Plots the histogram and boxplot of a target variable.
    
    Parameters:
        df (pandas.DataFrame): your dataset
        target_col (str): name of the target column
        save (bool): if True, save plots as PNG
        save_prefix (str): filename prefix for saved images
    """
    
    # Extract target as Series
    y = df[target_col]

    # ---- HISTOGRAM ----
    plt.figure(figsize=(10, 5))
    sns.histplot(y, bins=40, kde=True, edgecolor="black")
    plt.title(f"Histogram of {target_col}")
    plt.xlabel(target_col)
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)

    if save:
        plt.savefig(f"new_output/{save_prefix}_hist.png", dpi=300, bbox_inches="tight")

    # ---- BOXPLOT ----
    plt.figure(figsize=(10, 4))
    sns.boxplot(x=y, orient="h", showfliers=True)

    mean_val = y.mean()
    plt.axvline(mean_val, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_val:.2f}")

    plt.title(f"Boxplot of {target_col} with mean")
    plt.xlabel(target_col)
    plt.grid(alpha=0.3)

    if save:
        plt.savefig(f"new_output/{save_prefix}_boxplot.png", dpi=300, bbox_inches="tight")

    # ---- OUTLIER SUMMARY ----
    q1 = y.quantile(0.25)
    q3 = y.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = y[(y < lower) | (y > upper)]

    print("===== Outlier Summary =====")
    print(f"N outliers: {len(outliers)}")
    print(f"Lower bound: {lower:.3f}")
    print(f"Upper bound: {upper:.3f}")
    print(f"Min value:   {y.min():.3f}")
    print(f"Max value:   {y.max():.3f}")

    return outliers

outliers = plot_target_distribution(df, "critical_temp", save=True)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = "results"
OUTPUT_DIR = "plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)

precisions = [
    "embeddings_float64.npy",
    "embeddings_float32.npy",
    "embeddings_float16.npy",
    "embeddings_int8.npy"
]


# (1) Distribution plots
#     Plot histograms of intra- and inter- speaker distances for each precision.
#     This shows how the distance distributions change when we reduce precision.
def plot_distributions():
    for p in precisions:
        intra = np.load(os.path.join(RESULTS_DIR, f"{p}_intra.npy"))
        inter = np.load(os.path.join(RESULTS_DIR, f"{p}_inter.npy"))

        plt.figure(figsize=(8, 5))
        
        # overlap two histograms, blue=same speaker, red=different speakers
        # I use blue and red, so that the overlapped area is purple, easy to review.
        sns.histplot(intra, bins=50, color="blue", label="intra", stat="density", alpha=0.5)
        sns.histplot(inter, bins=50, color="red", label="inter", stat="density", alpha=0.5)

        plt.title(f"Distance distribution ({p})")
        plt.xlabel("Cosine distance")
        plt.ylabel("Density")
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{p}_distribution.png"))
        plt.close()


# (2) How average intra- and inter- distances change across precision levels.
#     Also plot the ratio, and smaller ratio means better seperation.
def plot_summary():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "summary.csv"))

    # Plot1: inta- and inter- distances side by side
    plt.figure(figsize=(8, 5))

    plt.plot(df["precision"], df["intra_mean"], marker="o", label="intra")
    plt.plot(df["precision"], df["inter_mean"], marker="o", label="inter")

    plt.xticks(rotation=45)
    plt.ylabel("Distance")
    plt.title("Intra vs Inter across precision")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "intra_inter_comparison.png"))
    plt.close()

    # Plot2: Ratio(intra, inter), show if speakers are still separable
    plt.figure(figsize=(8, 5))

    plt.plot(df["precision"], df["ratio"], marker="o")

    plt.xticks(rotation=45)
    plt.ylabel("Intra / Inter")
    plt.title("Ratio across precision")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "ratio_comparison.png"))
    plt.close()


# (3) Difference Plots
#     Show how much intra-speaker distances different from float 64 baseline.
#     Compare float32, float16, 8 bits with float64 respectively.
def plot_difference():
    intra64 = np.load(os.path.join(RESULTS_DIR, "embeddings_float64.npy_intra.npy"))

    for p in ["embeddings_float32.npy", "embeddings_float16.npy", "embeddings_int8.npy"]:
        intra = np.load(os.path.join(RESULTS_DIR, f"{p}_intra.npy"))

        diff = intra - intra64 #  positive value = lower precision gives larger distances

        plt.figure(figsize=(8, 5))
        sns.histplot(diff, bins=50)

        plt.title(f"Difference vs float64 ({p})")
        plt.xlabel("Difference")

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{p}_diff.png"))
        plt.close()



def main():
    print("[INFO] Plotting distributions...")
    plot_distributions()

    print("[INFO] Plotting summary...")
    plot_summary()

    print("[INFO] Plotting differences...")
    plot_difference()

    print("[INFO] All figures saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
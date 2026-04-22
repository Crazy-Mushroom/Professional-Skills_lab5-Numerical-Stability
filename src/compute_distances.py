import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances
from itertools import combinations
import time
from scipy.stats import spearmanr

# (1) Where to find the embeddings and save the results
FEATURE_DIR = "features"
META_PATH = "data/words/metadata.csv"
OUTPUT_DIR = "results"

# Create results folder if it does not exit
os.makedirs(OUTPUT_DIR, exist_ok=True)


# (2) Load input embeddings and do int8 dequantization

 #int8 cannot be used in computation directly, it is necessary to recover the values according to the int8_scale for computation.

def load_features(name):
    path = os.path.join(FEATURE_DIR, name)

    if name == "embeddings_int8.npy":
        # For int8: In the previous I saved quantized values, need to convert back to float
        emb = np.load(path)  #  load the quantized integers

        # load the scale factor saved during quantization
        scale = np.load(os.path.join(FEATURE_DIR, "embeddings_int8_scale.npy"))[0]
        emb = emb.astype(np.float32) * scale

    else:
        emb = np.load(path)

    return emb


# (3) Calculate the intra- and inter- speaker distances
def compute_stats(embeddings, meta):

    intra = []    # same speaker distances
    inter = []    # different speaker distances
    all_distances = [] # help check the relative ordering between intra- and interspeaker distances

    # Group by word, since only compare embeddings of the same word
    for word, group in meta.groupby("word"):
        idx = group.index.tolist() # retrieve the indices and speaker labels for this word
        spk = group["speaker"].values

        # Embeddings for just this word
        X = embeddings[idx]

        # Compute all pairwise cosine distances
        D = cosine_distances(X)

        # Loop through all unique pairs (i, j) where i < j
        for i, j in combinations(range(len(idx)), 2):
            all_distances.append(D[i, j])
            
            if spk[i] == spk[j]:
                intra.append(D[i, j]) # same speaker
            else:
                inter.append(D[i, j]) # different speaker

    return np.array(intra), np.array(inter), np.array(all_distances)


# (4) Prepare the calculations
def main():
    meta = pd.read_csv(META_PATH)
    
    # test all the precision levels
    feature_files = [
        "embeddings_float64.npy",
        "embeddings_float32.npy",
        "embeddings_float16.npy",
        "embeddings_int8.npy"
    ]

    results = []
    baseline_distances = None

    # Process each precision level
    for f in feature_files:
        print(f"[INFO] Processing {f}")

        emb = load_features(f) # load embeddings, especially the dequantized int8 embeddings automatically

        ## Masure disk usage
        path = os.path.join(FEATURE_DIR, f)
        size_mb = os.path.getsize(path) / (1024*1024)

        ## Time needed to compute the distance matrices
        start = time.time()
        intra, inter, all_distances = compute_stats(emb, meta) # use compute_stats to compute intra- and inter- distances
        compute_time = time.time() - start

        ## check  whether relative ordering between intra and inter is preserved
        order_preserved = intra.mean() < inter.mean() # if intra>=inter, it means that the model fails to distinguish intra and inter (order not preserved)
        margin = inter.mean()-intra.mean()

        # use spearman to see more details about the order
        if f == "embeddings_float64.npy":
            baseline_distances = all_distances
            spearman_corr = 1.0
        else:
            spearman_corr = spearmanr(baseline_distances, all_distances)[0]

        stats = {
            "precision": f,
            "size_mb": size_mb,
            "compute_time": compute_time,
            "intra_mean": intra.mean(),  # average intra-speaker distance
            "inter_mean": inter.mean(),  # average inter-speaker distance
            "ratio": intra.mean() / inter.mean(), # how separated speakers are
            "margin": margin,
            "order_preserved": order_preserved,
            "spearman_corr": spearman_corr,
            "intra_std": intra.std(),
            "inter_std": inter.std(),
        }

        results.append(stats)

        # Important! save raw distances for later visualization in the next stage
        np.save(os.path.join(OUTPUT_DIR, f"{f}_intra.npy"), intra)
        np.save(os.path.join(OUTPUT_DIR, f"{f}_inter.npy"), inter)
    

    # save a summary table for easy review
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUT_DIR, "summary.csv"), index=False)

    print(df)


if __name__ == "__main__":
    main()
import os
import time
import numpy as np
import warnings
import pandas as pd
from TSB_UAD.utils.visualisation import plotFig
from TSB_UAD.models.sand import SAND
# from TSB_UAD.models.feature import Window
from TSB_UAD.utils.slidingWindows import find_length
from TSB_UAD.vus.metrics import get_metrics
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

def run_sand(data, labels, name, results_dir):
    print("---------------------")
    print("Computing...")
    # Measure time
    start_time = time.time()
    #Pre-processing
    sliding_window = find_length(data)

    # Run SAND (online)
    MODELNAME='SAND (online)'
    clf = SAND(pattern_length=sliding_window,subsequence_length=4*(sliding_window))
    x = data
    clf.fit(x,online=True,alpha=0.5,init_length=5000,batch_size=2000,verbose=True,overlaping_rate=int(4*sliding_window))
    score = clf.decision_scores_

    #Post-processing
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Finished computing!")
    print("Creating plot...")

    #Plot result
    plotFig(data, labels, score, sliding_window, fileName=name, modelName=MODELNAME, results_dir=results_dir)
    print("Finished plot!")
    print("Computing plot...")
    #Print accuracy
    temp_results = get_metrics(score, labels, metric="all", slidingWindow=sliding_window)
    temp_results["time"] = elapsed_time
    print("Finished computing!")
    # for metric in results.keys():
    #     print(metric, ':', results[metric])

    return temp_results


if __name__ == "__main__":
    # Configuration
    DATA_DIR    = "generated_datasets"
    RESULTS_DIR = "results_streaming"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Run streaming SAND on all datasets
    summary = []

    for fname in sorted(os.listdir(DATA_DIR)):
        if not fname.endswith(".npy") or "_boundaries" in fname or "_labels" in fname:
            continue

        print(f"\nProcessing {fname}")
        ts = np.load(os.path.join(DATA_DIR, fname))

        # Construct the expected labels filename
        label_filename = fname.replace(".npy", "_labels.npy")
        label_path = os.path.join(DATA_DIR, label_filename)

        if os.path.exists(label_path):
            labels = np.load(label_path)

        else:
            raise FileNotFoundError(f"Labels file not found for {fname}")

        results = run_sand(ts, labels, fname, results_dir=RESULTS_DIR)
        summary.append({
            "dataset": fname,
            "length": len(ts),
            **results
        })

    # Save results to CSV
    results_df = pd.DataFrame(summary)
    results_df = results_df.sort_values(by="dataset")
    results_df.set_index("dataset", inplace=True)
    # Save to CSV as before
    results_df.to_csv(os.path.join(RESULTS_DIR, "sand.csv"))

import os
import time
import math
import numpy as np
import warnings
import pandas as pd
from TSB_UAD.utils.visualisation import plotFig
from TSB_UAD.models.iforest import IForest
from TSB_UAD.models.feature import Window
from TSB_UAD.utils.slidingWindows import find_length
from TSB_UAD.vus.metrics import get_metrics
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

def run_isolation_forest(data, labels, name, results_dir):
    print("---------------------")
    print("Computing...")
    # Measure time
    start_time = time.time()
    #Pre-processing
    sliding_window = find_length(data)
    x_data = Window(window = sliding_window).convert(data).to_numpy()
    #Run IForest
    model_name='Isolation Forest (offline)'
    clf = IForest(n_jobs=1)
    x = x_data
    clf.fit(x)
    score = clf.decision_scores_

    # Post-processing
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    score = np.array(
        [score[0]]*math.ceil(
            (sliding_window-1)/2) + list(score) + [score[-1]]*((sliding_window-1)//2))

    end_time = time.time()
    elapsed_time = end_time - start_time

    print("Finished computing!")
    print("Creating plot...")
    #Plot result
    plotFig(data, labels, score, sliding_window, fileName=name, modelName=model_name, results_dir=results_dir)
    print("Finished plot!")

    print("Computing plot...")
    #Print accuracy
    temp_results = get_metrics(score, labels, metric="all", slidingWindow=sliding_window)
    temp_results["time"] = elapsed_time
    print("Finished computing!")
    # for metric in temp_results.keys():
    #     print(metric, ':', temp_results[metric])

    return temp_results



if __name__ == "__main__":
    # Configuration
    DATA_DIR    = "generated_datasets"
    RESULTS_DIR = "tsb_uad_results_offline/if"
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

        results = run_isolation_forest(ts, labels, fname, results_dir=RESULTS_DIR)
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
    results_df.to_csv(os.path.join(RESULTS_DIR, "iforest.csv"))

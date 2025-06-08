import os
import time
import warnings
import numpy as np
import pandas as pd
from TSB_UAD.utils.visualisation import plotFig
from TSB_UAD.models.distance import Fourier
from TSB_UAD.models.lstm import lstm
from TSB_UAD.models.feature import Window
from TSB_UAD.utils.slidingWindows import find_length
from TSB_UAD.vus.metrics import get_metrics
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

def run_lstm(data, labels, name, results_dir):
    print("---------------------")
    print("Computing...")
    # Measure time
    start_time = time.time()
    #Pre-processing
    sliding_window = find_length(data)
    data_train = data[:int(0.1*len(data))]
    data_test = data
    #Run LSTM
    model_name='LSTM'
    clf = lstm(slidingwindow = sliding_window, predict_time_steps=1, epochs = 50, patience = 5, verbose=0)
    clf.fit(data_train, data_test)
    measure = Fourier()
    measure.detector = clf
    measure.set_param()
    clf.decision_function(measure=measure)

    # Post-processing
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(clf.decision_scores_.reshape(-1,1)).ravel()

    end_time = time.time()
    elapsed_time = end_time - start_time

    print("Finished computing!")
    print("Creating plot...")
    #Plot result
    plotFig(data, labels, score, sliding_window, fileName=name, modelName=model_name, results_dir=results_dir)
    print("Finished plot!")

    print("Computing metrics...")
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
    RESULTS_DIR = "tsb_uad_results_offline/lstm"
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

        results = run_lstm(ts, labels, fname, results_dir=RESULTS_DIR)
        summary.append({
            "dataset": fname,
            "length": len(ts),
            **results
        })

        break

    # Save results to CSV
    results_df = pd.DataFrame(summary)
    results_df = results_df.sort_values(by="dataset")
    results_df.set_index("dataset", inplace=True)
    # Save to CSV as before
    results_df.to_csv(os.path.join(RESULTS_DIR, "lstm.csv"))

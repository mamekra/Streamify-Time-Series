import os
import random
import matplotlib.pyplot as plt
import numpy as np

def read_timeseries(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # skip empty lines
            try:
                # Split line by comma and extract the first value (e.g., '5.0' from '5.0,0')
                value_str = line.split(',')[0].replace(',', '.')
                value = float(value_str)
                data.append(value)
            except (ValueError, IndexError):
                continue  # skip lines that are malformed
    return np.array(data)

def load_all_timeseries(root_dir):
    """
    Loads .out files from each domain subfolder.
    Returns a dict: {domain_name: [list_of_ts_arrays]}
    """
    all_data = {}
    for domain in os.listdir(root_dir):
        domain_path = os.path.join(root_dir, domain)
        if os.path.isdir(domain_path):
            ts_files = [f for f in os.listdir(domain_path) if f.endswith(".out")]
            ts_list = [read_timeseries(os.path.join(domain_path, f)) for f in ts_files]
            if ts_list:  # only keep domains with valid data
                all_data[domain] = ts_list
    return all_data

def pick_random_timeseries(all_data, num_series):
    """
    Picks 'num_series' time series from different domains.
    Returns a list of NumPy arrays.
    """
    domains = list(all_data.keys())
    if len(domains) < num_series:
        raise ValueError(f"Not enough domains to pick {num_series} different series.")

    selected_domains = random.sample(domains, num_series)
    selected_series = [random.choice(all_data[dom]) for dom in selected_domains]
    return selected_series, selected_domains

def create_normality_series(all_data):
    results = {}

    # Normality 1
    for domain in all_data:
        ts = random.choice(all_data[domain])
        results[f'normality_1_{domain.lower()}'] = {
            "data": ts,
            "domains": [domain],
            "boundaries": []
        }

    # Normality 2
    normality_2_pairs = [
        ("Daphnet", "Genesis"),
        ("Daphnet", "NASA-MSL"),
        ("Genesis", "NASA-MSL"),
        ("Genesis", "Daphnet"),
        ("NASA-MSL", "Daphnet"),
        ("NASA-MSL", "Genesis"),
    ]

    for i, (dom1, dom2) in enumerate(normality_2_pairs, start=1):
        ts1 = random.choice(all_data[dom1])
        ts2 = random.choice(all_data[dom2])
        concat_ts = np.concatenate([ts1, ts2])
        boundary = [len(ts1)]
        results[f'normality_2_{i}_{dom1.lower()}_{dom2.lower()}'] = {
            "data": concat_ts,
            "domains": [dom1, dom2],
            "boundaries": boundary
        }

    # Normality 3 — all 3-domain permutations as defined
    normality_3_combos = [
        ("Daphnet", "Genesis", "NASA-MSL"),
        ("Daphnet", "NASA-MSL", "Genesis"),
        ("Genesis", "NASA-MSL", "Daphnet"),
        ("Genesis", "Daphnet", "NASA-MSL"),
        ("NASA-MSL", "Daphnet", "Genesis"),
        ("NASA-MSL", "Genesis", "Daphnet"),
    ]

    for i, (dom1, dom2, dom3) in enumerate(normality_3_combos, start=1):
        ts1 = random.choice(all_data[dom1])
        ts2 = random.choice(all_data[dom2])
        ts3 = random.choice(all_data[dom3])
        concat_ts = np.concatenate([ts1, ts2, ts3])
        boundaries = [len(ts1), len(ts1) + len(ts2)]
        results[f'normality_3_{i}_{dom1.lower()}_{dom2.lower()}_{dom3.lower()}'] = {
            "data": concat_ts,
            "domains": [dom1, dom2, dom3],
            "boundaries": boundaries
        }

    return results

def plot_normality_series(name, data, domains, boundaries):
    plt.figure(figsize=(14, 4))
    plt.plot(data, label='Concatenated Time Series')
    for i, boundary in enumerate(boundaries):
        plt.axvline(x=boundary, color='red', linestyle='--', label=f'Shift {i+1}' if i == 0 else None)
        plt.axvspan(boundary, boundaries[i+1] if i+1 < len(boundaries) else len(data),
                    alpha=0.1, color='orange')
    plt.title(f"{name.upper()} - Domains: {', '.join(domains)}")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    root_dir = r"C:\....\timeseries_used"  # timeseries_used folder contains "Daphnet", "Genesis", "NASA-MSL" folders with timeseries
    all_data = load_all_timeseries(root_dir)

    normalities = create_normality_series(all_data)

    for key, value in normalities.items():
        print(f"\n{key.upper()}:")
        print(f"  Domains: {value['domains']}")
        print(f"  Length: {len(value['data'])}")
        print(f"  Distribution shift boundaries: {value['boundaries']}")

        # Save to file
        np.savetxt(f"{key}.txt", value['data'])

        # Plot
        plot_normality_series(key, value['data'], value['domains'], value['boundaries'])

if __name__ == "__main__":
    main()

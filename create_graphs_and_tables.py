# Use with 'python3 create_graphs_and_tables.py ppl.summary log.summary'

import pandas as pd
import argparse
import matplotlib.pyplot as plt

def parse_estimates(data):
    estimates = []
    for line in data.strip().split("\n"):
        if "-- Final estimate:" in line:
            parts = line.split("-- Final estimate:")
            label = parts[0].strip()
            ppl_part = parts[1].strip()
            ppl, error = ppl_part.replace("PPL = ", "").split(" +/- ")
            estimates.append({"Label": label, "PPL": float(ppl), "Error": float(error)})
    return pd.DataFrame(estimates)

def parse_zfp_results(data):
    from io import StringIO
    csv_data = StringIO(data)
    df = pd.read_csv(csv_data, header=None, names=[
        "Prefix", "Type", "Key1", "Precision/Rate/Accuracy", "Key2", "Original Size (MiB)",
        "Key3", "Compressed Size (MiB)", "Key4", "Compression Ratio", "Key5", "Bits per Weight"
    ])
    df = df[["Type", "Precision/Rate/Accuracy", "Original Size (MiB)", "Compressed Size (MiB)", "Compression Ratio", "Bits per Weight"]]
    return df

def decode_configuration(label):
    if not isinstance(label, str) or not label.startswith("from_ZFP"):
        return pd.Series({"Config_Type": None, "Config_Value": None})

    lookup_tab = {
        "PREC": "precision",
        "RATE": "rate",
        "TOL": "accuracy",
    }

    parts = label.split("-")[1:]
    config_type = parts[0].split("_")[0]
    config_value = parts[0].split("_")[1]
    return pd.Series({"Config_Type": lookup_tab.get(config_type), "Config_Value": config_value})

def match_configurations(estimates_df, zfp_results_df):
    def match_row(row):
        if pd.isna(row["Config_Type"]) or pd.isna(row["Config_Value"]):
            return pd.Series({
                "Precision/Rate/Accuracy": None,
                "Original Size (MiB)": None,
                "Compressed Size (MiB)": None,
                "Compression Ratio": None,
                "Bits per Weight": None
            })
        match = zfp_results_df[
            (zfp_results_df["Type"].str.contains(row["Config_Type"], na=False)) &
            (zfp_results_df["Precision/Rate/Accuracy"] == float(row["Config_Value"]))
        ]
        if not match.empty:
            return match.iloc[0]
        return pd.Series({
            "Precision/Rate/Accuracy": None,
            "Original Size (MiB)": None,
            "Compressed Size (MiB)": None,
            "Compression Ratio": None,
            "Bits per Weight": None
        })
    
    return pd.concat([estimates_df, estimates_df.apply(match_row, axis=1)], axis=1)

parser = argparse.ArgumentParser(description="Combine ZFP data from two files.")
parser.add_argument("estimates_file", type=str, help="Path to the estimates data file.")
parser.add_argument("zfp_results_file", type=str, help="Path to the ZFP results data file.")
args = parser.parse_args()

with open(args.estimates_file, 'r') as file:
    estimates_data = file.read()

with open(args.zfp_results_file, 'r') as file:
    zfp_results_data = file.read()

estimates_df = parse_estimates(estimates_data)
zfp_results_df = parse_zfp_results(zfp_results_data)

estimates_df = pd.concat([estimates_df, estimates_df["Label"].apply(decode_configuration)], axis=1)
combined_df = match_configurations(estimates_df, zfp_results_df)

# Plotting lines grouped by 'Type'
if "Bits per Weight" in combined_df.columns and "PPL" in combined_df.columns and "Type" in combined_df.columns:
    plt.figure(figsize=(10, 6))
    for type_name, group in combined_df.groupby("Type"):
        group = group.dropna(subset=["Bits per Weight", "PPL"])
        if not group.empty:
            plt.plot(group["Bits per Weight"], group["PPL"], marker="o", label=type_name)
    
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Bits per Weight (log scale)")
    plt.ylabel("Perplexity (PPL) (log scale)")
    plt.title("Perplexity vs Bits per Weight Grouped by Type")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.savefig("output_grouped.png")
    #plt.show()
else:
    print("Required columns for visualization are missing.")


import os
import argparse

import pandas as pd

# Get input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--result_dir', default="", type=str)

args = parser.parse_args()

if not os.path.exists(args.result_dir):
    raise Exception("wrong dir")

df_list = []
for run in sorted(os.listdir(args.result_dir)):
    if os.path.isdir(os.path.join(args.result_dir, run)):
        file = os.path.join(args.result_dir, run, "result.csv")
        if not os.path.exists(file):
            print(f"[WARN] Skipping '{run}': missing result.csv")
            continue
        df = pd.read_csv(file)
        # Prefer stable folder naming like "sub-01", but stay compatible with older
        # timestamped naming like "20260211-145309-sub-10".
        if run.startswith("sub-"):
            sub = run
        elif "sub-" in run:
            sub = run[run.rfind("sub-"):]
        else:
            sub = run
        df['sub'] = sub
        cols = ['sub'] + [col for col in df.columns if col != 'sub']
        df = df[cols]
        df_list.append(df)

# If nothing was collected, fail with a helpful message.
if len(df_list) == 0:
    raise FileNotFoundError(
        f"No result.csv files found under: {args.result_dir}\n"
        f"Hint: ensure training finished and wrote result.csv, or delete incomplete run folders."
    )

# Concatenate all DataFrames
all_data_raw = pd.concat(df_list, ignore_index=True)

# Sort by sub (numeric), keep any non-matching IDs at the end
def extract_sub_num(x):
    if x == "Average":
        return float('inf')
    digits = ''.join(filter(str.isdigit, str(x)))
    return int(digits) if digits else float('inf')

all_data_raw = all_data_raw.sort_values(by="sub", key=lambda col: col.map(extract_sub_num))

# Save a clean per-subject table (one row per subject/run)
all_data_raw.to_csv(os.path.join(args.result_dir, 'results.csv'), index=False)

# Keep legacy output (avg row + formatting) as avg_results.csv
all_data = all_data_raw.copy()

# Extract numeric columns (excluding 'sub' and 'best epoch')
numeric_cols = all_data.columns.difference(['sub', 'best epoch'])

# Convert to float, keep two decimal places and pad with zeros (convert to string)
for col in numeric_cols:
    all_data[col] = all_data[col].astype(float).map(lambda x: f"{x:.1f}")

# Calculate average values (still using float for calculation, then formatting)
avg_values = all_data[numeric_cols].astype(float).mean()
avg_row = {col: f"{avg_values[col]:.1f}" for col in numeric_cols}
avg_row['sub'] = 'Average'

# Add average row
all_data = pd.concat([all_data, pd.DataFrame([avg_row])], ignore_index=True)

all_data = all_data.sort_values(by="sub", key=lambda col: col.map(extract_sub_num))

# Save the merged result
all_data.to_csv(os.path.join(args.result_dir, 'avg_results.csv'), index=False)

print(all_data)
import pandas as pd
import json
import numpy as np

# 1. SETUP: Input and Output filenames
INPUT_FILE = "manga_data_full_1.jsonl" # Or "manga.json" if you used the list format
OUTPUT_FILE = "final_manga_dataset.csv"

print("Reading JSON data...")

# 2. LOAD DATA
# If you used the line-by-line method (Recommended):
data = []
try:
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip(): # Skip empty lines
                data.append(json.loads(line))
except FileNotFoundError:
    print(f"Error: Could not find {INPUT_FILE}")
    exit()

# 3. CREATE DATAFRAME
df = pd.DataFrame(data)

# 4. CLEANING (The Crucial Step)
# You have a list ['Action', 'Comedy'] in the 'tags' column.
# CSVs hate lists. We must convert them to strings "Action, Comedy".
print("Processing tags...")
df['tags'] = df['tags'].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)

# 5. HANDLE MISSING VALUES
# Replace 'None' in scores with NaN so Pandas can do math later
df['score'] = pd.to_numeric(df['score'], errors='coerce') 

df['demographic']=df["demographic"].replace(["Unknown"],np.nan)

df["id"]=df["id"].round().astype("Int64")

df["end_date"]=df["end_date"].replace(["Ongoing"],np.nan)

df.sort_values("id",inplace=True)

# 6. SAVE TO CSV
df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')

print(f"Success! Converted {len(df)} rows to {OUTPUT_FILE}")
print("\nFirst 5 rows:")
print(df.head())
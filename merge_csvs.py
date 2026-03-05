import pandas as pd
import os

# 1. Define your files
files = ["manga_dataset_1.csv", "manga_dataset_2.csv", "manga_dataset_3.csv","manga_dataset_4.csv","manga_dataset_5.csv","manga_dataset_6.csv","manga_dataset_7.csv","manga_dataset_8.csv"]
all_dfs = []

print("Reading files...")

for file in files:
    if os.path.exists(file):
        try:
            # Read CSV
            # on_bad_lines='skip' ensures one bad row doesn't crash the script
            df = pd.read_csv(file, on_bad_lines='skip')
            
            # Print stats so you know it worked
            print(f"  -> Loaded {file}: {len(df)} rows")
            all_dfs.append(df)
        except Exception as e:
            print(f"  xx Failed to read {file}: {e}")
    else:
        print(f"  xx File not found: {file}")

# 2. Merge (Concatenate)
if all_dfs:
    print("\nMerging data...")
    # ignore_index=True resets the index so you don't have duplicate row numbers (0, 1, 2, 0, 1, 2...)
    final_df = pd.concat(all_dfs, ignore_index=True)
    
    initial_count = len(final_df)
    
    # 3. Clean Duplicates (Crucial for Scraped Data)
    # subset=['id'] ensures we check for duplicate manga IDs. 
    # If your ID column is named 'mal_id', change 'id' to 'mal_id'.
    if 'id' in final_df.columns:
        final_df = final_df.drop_duplicates(subset=['id'], keep='first')
    else:
        # Fallback: drop if ALL columns are identical
        final_df = final_df.drop_duplicates(keep='first')
        
    print(f"  -> Removed {initial_count - len(final_df)} duplicates.")
    final_df["id"]=final_df["id"].round().astype("Int64")
    final_df = final_df.sort_values(by='id')
    # 4. Save
    output_filename = "final_manga_dataset_clean.csv"
    # index=False prevents creating that annoying 'Unnamed: 0' column
    final_df.to_csv(output_filename, index=False, encoding='utf-8')
    print(f"\nSuccess! Saved {len(final_df)} unique rows to '{output_filename}'")
    
    # Preview
    print(final_df.head())

else:
    print("\nNo data found. Check your file names.")
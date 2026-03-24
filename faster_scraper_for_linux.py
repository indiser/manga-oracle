from curl_cffi import requests
import random
import json
import time
import os
import ctypes

def prevent_sleep():
    if os.name == 'nt':
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
        except: pass
        
prevent_sleep()

script_dir=os.path.dirname(os.path.abspath(__file__))

# --- FILES ---
DATA_FILE = os.path.join(script_dir,"manga_data_full_1.jsonl")      # The actual data
BAD_IDS_FILE = os.path.join(script_dir,"bad_ids_1.txt")   # The blacklist (404s)
GOOD_IDS_FILE = os.path.join(script_dir,"good_ids_1.txt") # The whitelist (Successes)
LOW_RANGE=1                     #The start range
HIGH_RANGE=600000                #The end range
TARGET_COUNT = 500000
MAX_CONSICUTIVE_FALIURE=10000

# --- STEP 1: LOAD MEMORY ---
# We use sets for instant lookup (O(1) speed)
bad_ids = set()
good_ids = set()

# Load Bad IDs
if os.path.exists(BAD_IDS_FILE):
    with open(BAD_IDS_FILE, "r") as f:
        # Read lines, strip whitespace, convert to int
        bad_ids = {int(line.strip()) for line in f if line.strip().isdigit()}

# Load Good IDs
if os.path.exists(GOOD_IDS_FILE):
    with open(GOOD_IDS_FILE, "r") as f:
        good_ids = {int(line.strip()) for line in f if line.strip().isdigit()}

print(f"Memory Loaded. Blacklisted: {len(bad_ids)} | Completed: {len(good_ids)}")

# --- STEP 2: HELPER FUNCTION ---
def save_id(filename, manga_id):
    """Instantly appends an ID to a file so it's never lost."""
    with open(filename, "a") as f:
        f.write(f"{manga_id}\n")

# --- STEP 3: THE LOOP ---
success_count = 0
consecutive_errors = 0

while success_count < TARGET_COUNT:
    
    manga_id = random.randint(LOW_RANGE, HIGH_RANGE+1)
    
    # 1. THE CHECK (The "Never Touch Again" Logic)
    if manga_id in bad_ids:
        # print(f"Skipping {manga_id} (Blacklisted)") # Optional: Un-comment to see it working
        continue
    
    if manga_id in good_ids:
        continue

    print(f"Checking ID {manga_id}...")

    try:
        url = f"https://api.jikan.moe/v4/manga/{manga_id}/full"
        response = requests.get(url, timeout=10, impersonate="chrome")
        
        # --- CASE: BAD ID (404) ---
        if response.status_code == 404:
            print(f"-- ID {manga_id} is EMPTY. Blacklisting...")
            
            # Update Memory
            bad_ids.add(manga_id)
            
            # Update Disk IMMEDIATELY
            save_id(BAD_IDS_FILE, manga_id)
            
            time.sleep(1)
            continue

        # --- CASE: RATE LIMIT (429) ---
        if response.status_code == 429:
            print("!! Rate Limit. Sleeping 10s...")
            time.sleep(10)
            continue 

        # --- CASE: SUCCESS (200) ---
        if response.status_code == 200:
            data = response.json().get("data", {})
            pub = data.get("published", {})
            
            # Extract Data
            record = {
                "id": data.get("mal_id"),
                "title": data.get("title_english") or data.get("title"),
                "score": data.get("score"),
                "start_date": pub.get("from"),
                "end_date": pub.get("to"),
                "members": data.get("members"),
                "demographic": (data.get("demographics") or [{"name": "Unknown"}])[0]["name"],
                "is_finished": 1 if data.get("status") == "Finished" else 0,
                "magazine": (data.get("serializations") or [{"name": "None"}])[0]["name"],
                "tags": [x["name"] for x in (data.get("genres", []) + data.get("themes", []))],
            }

            # Save Data Record
            with open(DATA_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
            # Save to Good ID List
            good_ids.add(manga_id)
            save_id(GOOD_IDS_FILE, manga_id)

            print(f"++ Captured: {record['title']}")
            success_count += 1
            consecutive_errors = 0

    except Exception as e:
        print(f"Error: {e}")
        consecutive_errors += 1
        if consecutive_errors > MAX_CONSICUTIVE_FALIURE:
            break

    time.sleep(1.5)

print("Job Done.")
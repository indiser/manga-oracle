import json
import requests
import os
import time
import ctypes

script_dir=os.path.dirname(os.path.abspath(__file__))

def prevent_sleep():
    if os.name == 'nt':
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
        except: pass
prevent_sleep()

good_ids_path=os.path.join(script_dir,"good_ids_1_sorted.txt")
already_seen_path=os.path.join(script_dir,"already_seen_ids.txt")
storage_path=os.path.join(script_dir,"manga_full_1.jsonl")

good_ids=already_seen_ids=set()

if os.path.exists(good_ids_path):
    with open(good_ids_path) as filp:
        good_ids={int(line.strip()) for line in filp if line.strip().isdigit()}

if os.path.exists(already_seen_path):
    with open(already_seen_path) as filp:
        already_seen_ids={int(line.strip()) for line in filp if line.strip().isdigit()}

def save_id(filename, manga_id):
    with open(filename, "a") as f:
        f.write(f"{manga_id}\n")

faluire_count=0

for manga_id in good_ids:

    if manga_id in already_seen_ids:
        continue

    try:
        url = f"https://api.jikan.moe/v4/manga/{manga_id}/full"
        response = requests.get(url, timeout=10)

        # --- CASE: RATE LIMIT (429) ---
        if response.status_code == 429:
            print("!! Rate Limit. Sleeping 10s...")
            time.sleep(10)
            continue 

        if response.status_code == 404:
            print(f"{manga_id} is now blocked")
            faluire_count+=1
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
            with open(storage_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            already_seen_ids.add(manga_id)
            save_id(already_seen_path,manga_id)

            print(f"++ Captured: {record['title']}")

    except Exception as e:
        print(f"Error: {e}")

    time.sleep(0.5)

print("All Scraping Done....")
print(f"The failed count is: {faluire_count}")
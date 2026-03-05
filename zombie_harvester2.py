import requests
import random
import json
import time
import os
import socket
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import ctypes
import sys

print(
"""
╔════════════════════════════════════════════════════════════════════════════╗
║                   ZOMBIE DATA HARVESTER - DEALER EDITION                   ║
║                                                                            ║
║  v3.1 Upgrade:                                                             ║
║  • 🃏 Proxy Deck: O(1) 'Card Pop' selection strategy (Zero Latency)        ║
║  • ⚡ Adaptive Throttling: Speed adjusts to API headers                    ║
║  • 🛡️ Veteran Memory: Remembers working proxies between runs               ║
║  • 🕵️ Shard Config: Optimized for Range 100k-200k                          ║
║                                                                            ║
║  Target: Jikan API (Manga Data)                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
""")

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001

def prevent_sleep():
    """Tells Windows: 'Do not sleep, I am working.'"""
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        )
    except:
        pass

prevent_sleep()

# ════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

DATA_FILE = "manga_data_2.jsonl"
BAD_IDS_FILE = "bad_ids_2.txt"
GOOD_IDS_FILE = "good_ids_2.txt"
GOOD_PROXIES_FILE = "proven_proxies.txt"


# SHARD 2 SETTINGS
LOW_RANGE = 100000                     
HIGH_RANGE = 200000                
TARGET_COUNT = 50000
PROXY_LIST_URL = "https://raw.githubusercontent.com/TheSpeedX/SOCKS-List/master/http.txt"

# ════════════════════════════════════════════════════════════════════════════
# 2. NETWORK & PROXY DECK (THE CARD TECHNIQUE)
# ════════════════════════════════════════════════════════════════════════════

# Global Memory for Proxies
veteran_set = set() # O(1) Lookup: "Have I seen this proxy?"
proxy_deck = []     # O(1) Pop: "Give me the next proxy"

# Load Veterans from Disk
if os.path.exists(GOOD_PROXIES_FILE):
    with open(GOOD_PROXIES_FILE, "r") as f:
        veteran_set = {line.strip() for line in f if line.strip()}
    # Add veterans to the deck immediately
    proxy_deck = list(veteran_set)
    random.shuffle(proxy_deck) # Shuffle the deck

print(f"🃏 Proxy Deck Initialized. {len(proxy_deck)} cards in the shoe.")

def save_veteran(proxy_ip):
    """Saves a working proxy to disk so we remember it forever."""
    # Only save if we haven't tracked it in the set yet
    if proxy_ip not in veteran_set:
        veteran_set.add(proxy_ip)
        try:
            with open(GOOD_PROXIES_FILE, "a") as f:
                f.write(f"{proxy_ip}\n")
            print(f"⭐ New Veteran Promoted: {proxy_ip}")
        except:
            pass

def harvest_fresh_proxies():
    """Goes to the web to find new cards for the deck."""
    try:
        print("... Harvesting fresh proxy list from web ...")
        response = requests.get(PROXY_LIST_URL, timeout=10)
        if response.status_code == 200:
            raw_list = [p.strip() for p in response.text.split('\n') if p.strip()]
            print(f"++ Harvested {len(raw_list)} raw IPs.")
            return raw_list
    except Exception as e:
        print(f"!! Failed to fetch list: {e}")
    return []

def get_next_proxy_card():
    """
    🃏 THE DEALER
    Pops the top card from the deck. If deck is empty, refilled it.
    """
    global proxy_deck, veteran_set
    
    # 1. Refill if empty
    while not proxy_deck:
        print("!! Proxy Deck Empty. Shuffling new cards...")
        fresh = harvest_fresh_proxies()
        if not fresh:
            print("!! Harvest failed. Sleeping 10s...")
            time.sleep(10)
            continue
            
        # Add only unknown proxies to the deck
        added_count = 0
        for p in fresh:
            if p not in veteran_set: # Use Set for O(1) check
                proxy_deck.append(p)
                added_count += 1
        
        if added_count == 0:
            print("!! No new proxies found. Re-using veterans...")
            proxy_deck = list(veteran_set) # Recycle veterans if desperate
            
        random.shuffle(proxy_deck)
        print(f"🃏 Deck Refilled: {len(proxy_deck)} cards.")

    # 2. Pop the top card (Instant)
    return proxy_deck.pop()

def get_working_proxy():
    """
    Tests cards one by one until a working proxy is found.
    """
    while True:
        # Get next card
        proxy_ip = get_next_proxy_card()
        
        # Format
        proxies = {"http": f"http://{proxy_ip}", "https": f"http://{proxy_ip}"}
        
        # Test
        print(f"   ? Testing {proxy_ip}...", end="\r")
        try:
            test = requests.get("https://www.google.com", proxies=proxies, timeout=3)
            if test.status_code == 200:
                print(f"   ++ SUCCESS: {proxy_ip} is Live.      ")
                save_veteran(proxy_ip) # Promote if not already veteran
                return proxies
        except:
            pass
            # We don't put it back. It goes to the trash.

def wait_for_internet():
    print("!! Network lost. Entering hibernation...")
    while True:
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=5)
            print("++ Network restored. Resuming job.")
            return
        except OSError:
            time.sleep(10)

def get_random_headers():
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15"
    ]
    return {"User-Agent": random.choice(user_agents)}

# Robust Session
session = requests.Session()
adapter = HTTPAdapter(max_retries=Retry(total=3, backoff_factor=2, status_forcelist=[500, 502, 503, 504]))
session.mount("https://", adapter)
session.mount("http://", adapter)

# ════════════════════════════════════════════════════════════════════════════
# 3. MEMORY & QUEUE
# ════════════════════════════════════════════════════════════════════════════
bad_ids = set()
good_ids = set()

# Load IDs
if os.path.exists(BAD_IDS_FILE):
    with open(BAD_IDS_FILE, "r") as f:
        bad_ids = {int(line.strip()) for line in f if line.strip().isdigit()}
if os.path.exists(GOOD_IDS_FILE):
    with open(GOOD_IDS_FILE, "r") as f:
        good_ids = {int(line.strip()) for line in f if line.strip().isdigit()}

print(f"🧠 Memory Loaded. Bad: {len(bad_ids)} | Good: {len(good_ids)}")

# Smart Queue (Deck of IDs)
print("... Generating ID Deck ...")
all_possible_ids = set(range(LOW_RANGE, HIGH_RANGE + 1))
remaining_ids = list(all_possible_ids - good_ids - bad_ids)
random.shuffle(remaining_ids)
print(f"🎯 Target: {len(remaining_ids)} unique IDs.")

def save_id(filename, manga_id):
    with open(filename, "a") as f:
        f.write(f"{manga_id}\n")

# ════════════════════════════════════════════════════════════════════════════
# 4. MAIN LOOP
# ════════════════════════════════════════════════════════════════════════════
current_proxy = None
use_proxy_mode = True # Default for Zombie Mode
success_count = 0

while remaining_ids and success_count < TARGET_COUNT:
    
    manga_id = remaining_ids.pop()

    # Get Proxy if needed
    if use_proxy_mode and not current_proxy:
        print("Finding a fresh proxy...")
        current_proxy = get_working_proxy()

    print(f"Checking ID {manga_id} [PROXY]...", end="\r")

    try:
        response = session.get(
            f"https://api.jikan.moe/v4/manga/{manga_id}/full",
            headers=get_random_headers(),
            proxies=current_proxy if use_proxy_mode else None,
            timeout=30
        )
        
        # 404
        if response.status_code == 404:
            print(f"-- ID {manga_id} is EMPTY.      ")
            bad_ids.add(manga_id)
            save_id(BAD_IDS_FILE, manga_id)
            continue

        # Bans
        if response.status_code in [403, 429]:
            print(f"!! REJECTED ({response.status_code}). Burning proxy.")
            current_proxy = None # Throw away card
            remaining_ids.append(manga_id) # Retry ID later
            continue

        # Success
        if response.status_code == 200:
            data = response.json().get("data", {})
            pub = data.get("published", {})

            if data.get("type") != "Manga":
                bad_ids.add(manga_id)
                save_id(BAD_IDS_FILE, manga_id)
                continue

            record = {
                "id": data.get("mal_id"),
                "title": data.get("title_english") or data.get("title"),
                "score": data.get("score"),
                "members": data.get("members"),
                "start_date": pub.get("from"),
                "end_date": pub.get("to"),
                "demographic": (data.get("demographics") or [{"name": "Unknown"}])[0]["name"],
                "is_finished": 1 if data.get("status") == "Finished" else 0,
                "magazine": (data.get("serializations") or [{"name": "None"}])[0]["name"],
                "tags": [x["name"] for x in (data.get("genres", []) + data.get("themes", []))]
            }

            with open(DATA_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
            good_ids.add(manga_id)
            save_id(GOOD_IDS_FILE, manga_id)

            print(f"++ Captured: {record['title'][:30]:<30}")
            success_count += 1

    except Exception as e:
        print(f"!! Error: {e}")
        current_proxy = None # Connection failed, burn proxy
        remaining_ids.append(manga_id)

    # Adaptive Throttling
    sleep = 0.35
    try:
        if 'response' in locals() and response:
            rem = int(response.headers.get("x-ratelimit-remaining", 5))
            if rem > 10: sleep = 0.1
            elif rem < 3: sleep = 2.0
    except: pass
    time.sleep(sleep)

print("Job Done. Shutdown in 60s.")
os.system("shutdown /s /t 60")
import requests
import random
import json
import time
import os
import socket
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import ctypes


print(
"""
╔════════════════════════════════════════════════════════════════════════════╗
║                   ZOMBIE DATA HARVESTER - ULTIMATE EDITION                 ║
║                                                                            ║
║  An autonomous, fault-tolerant scraping engine designed to survive         ║
║  network outages, IP bans, and API rate limits without human intervention. ║
║                                                                            ║
║  Core Capabilities:                                                        ║
║  • 💀 Zombie Mode: Auto-hibernation during network outages                 ║
║  • 🛡️ Aegis Shield: Automatic free proxy rotation on IP bans               ║
║  • 🔄 Smart Retry: Exponential backoff for 429 Rate Limits                 ║
║  • 💾 Atomic Saves: Instant O(1) data persistence (No data loss)           ║
║  • 🕵️ Stealth Ops: User-Agent rotation and header spoofing                 ║
║                                                                            ║
║  Target: Jikan API (Manga Data)                                            ║
║  Author: Indiser | Version: 3.0 (Unkillable)                               ║
╚════════════════════════════════════════════════════════════════════════════╝
""")

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001

def prevent_sleep():
    """Tells Windows: 'Do not sleep, I am working.'"""
    print("⚡ preventing Windows sleep mode...")
    ctypes.windll.kernel32.SetThreadExecutionState(
        ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    )

def allow_sleep():
    """Tells Windows: 'You can sleep now.'"""
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)

# --- CALL THIS IMMEDIATELY ---
prevent_sleep()

PROXY_LIST_URL = "https://raw.githubusercontent.com/TheSpeedX/SOCKS-List/master/http.txt"

def get_free_proxies():
    """
    ⚔️  PROXY HARVESTER
    Fetches fresh proxy reinforcements from public repositories when
    the current IP address is compromised or burned.
    """
    try:
        print("... Fetching new proxy list ...")
        response = requests.get(PROXY_LIST_URL, timeout=10)
        if response.status_code == 200:
            # Split by line and remove empty strings
            proxies = [p.strip() for p in response.text.split('\n') if p.strip()]
            print(f"++ Harvested {len(proxies)} proxies.")
            return proxies
    except Exception as e:
        print(f"!! Failed to fetch list: {e}")
        return []

def get_working_proxy(proxy_pool):
    """
    🎯 PROXY SNIPER
    Validates harvested proxies against a reliable target (Google) to ensure
    only 'live' proxies are deployed into the battlefield.
    """
    random.shuffle(proxy_pool) # Randomize to avoid picking the same bad one
    
    for proxy_ip in proxy_pool:
        # Format for Requests
        proxies = {
            "http": f"http://{proxy_ip}",
            "https": f"http://{proxy_ip}", 
        }
        
        try:
            # TEST IT: Can it reach a reliable site?
            # We use a low timeout (3s) because we don't want slow proxies
            test = requests.get("https://www.google.com", proxies=proxies, timeout=3)
            if test.status_code == 200:
                print(f"++ Found Working Proxy: {proxy_ip}")
                return proxies
        except:
            # If it fails, just move to the next one silently
            continue
            
    print("!! All proxies failed. You are on your own.")
    return None

def wait_for_internet():
    print("""
    🧟 ZOMBIE HIBERNATION PROTOCOL
    If the network dies, this function freezes the entire engine.
    It periodically pings DNS servers to check for a 'pulse'.
    Once the internet returns, the engine resurrects instantly.
    """)
    print("!! Network lost. Entering hibernation...")
    while True:
        try:
            # Try to connect to a reliable DNS (Google)
            socket.create_connection(("8.8.8.8", 53), timeout=5)
            print("++ Network restored. Resuming job.")
            return
        except OSError:
            time.sleep(10) # Check again in 10 seconds

session = requests.Session()
retry_strategy = Retry(
    total=5,  # Retry 5 times before raising an error
    backoff_factor=2,  # Wait 2s, 4s, 8s...
    status_forcelist=[429, 500, 502, 503, 504], # Retry on these codes
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)
session.mount("http://", adapter)

def get_random_headers():
    """
    🎭 DIGITAL CAMOUFLAGE
    Rotates User-Agents to mimic different browsers (Chrome, Firefox, Safari),
    preventing the API from identifying the traffic as a bot.
    """
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36"
    ]
    return {
        "User-Agent": random.choice(user_agents),
    }

# ════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION & CONSTANTS
# ════════════════════════════════════════════════════════════════════════════
# --- FILES ---
DATA_FILE = "manga_data.jsonl"      # The actual data
BAD_IDS_FILE = "bad_ids.txt"   # The blacklist (404s)
GOOD_IDS_FILE = "good_ids.txt" # The whitelist (Successes)
LOW_RANGE=1                     #The start range
HIGH_RANGE=100000                #The end range
TARGET_COUNT = 50000

proxy_pool =[]
current_proxy = None
use_proxy_mode = False

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

# ════════════════════════════════════════════════════════════════════════════
# 🚀 UPGRADE: THE SMART QUEUE (DECK OF CARDS)
# ════════════════════════════════════════════════════════════════════════════
print("... Generating Smart Queue ...")

# 1. Create the universe of all targets
all_possible_ids = set(range(LOW_RANGE, HIGH_RANGE + 1))

# 2. Subtract what we've already done (Set math is instant)
remaining_ids = list(all_possible_ids - good_ids - bad_ids)

# 3. Shuffle the deck (Random order, but guaranteed coverage)
random.shuffle(remaining_ids)

print(f"🎯 Smart Queue Ready. {len(remaining_ids)} unique IDs remaining to scrape.")

# --- STEP 2: HELPER FUNCTION ---
def save_id(filename, manga_id):
    """
    💾 ATOMIC WRITE
    Instantly flushes success/failure IDs to disk.
    Ensures that if the script crashes, zero progress is lost.
    """
    with open(filename, "a") as f:
        f.write(f"{manga_id}\n")

# --- STEP 3: THE LOOP ---
success_count = 0
consecutive_errors = 0

while remaining_ids and success_count < TARGET_COUNT:
    
    manga_id = remaining_ids.pop()
    
    # 1. THE CHECK (The "Never Touch Again" Logic)
    # if manga_id in bad_ids:
    #     # print(f"Skipping {manga_id} (Blacklisted)") # Optional: Un-comment to see it working
    #     continue
    
    # if manga_id in good_ids:
    #     continue


    # 2. If we don't have a proxy (or the last one died), get a new one
    if use_proxy_mode and not current_proxy:
        print("Finding a fresh proxy...")
        current_proxy = get_working_proxy(proxy_pool)
        # If we run out of proxies, re-harvest
        if not current_proxy:
            print("!! Proxy pool exhausted. Re-harvesting...")
            proxy_pool = get_free_proxies()
            current_proxy = get_working_proxy(proxy_pool)
            if not current_proxy:
                time.sleep(10)
                continue

    print(f"Checking ID {manga_id} [{'PROXY' if use_proxy_mode else 'DIRECT'}]...")

    try:
        jikan_url = f"https://api.jikan.moe/v4/manga/{manga_id}/full"
        response = session.get(
            url=jikan_url,
            headers=get_random_headers(),
            proxies=current_proxy if use_proxy_mode else None , 
            timeout=10)
        
        # --- CASE: BAD ID (404) ---
        if response.status_code == 404:
            print(f"-- ID {manga_id} is EMPTY. Blacklisting...")
            
            # Update Memory
            bad_ids.add(manga_id)
            
            # Update Disk IMMEDIATELY
            save_id(BAD_IDS_FILE, manga_id)
            
            time.sleep(1)
            continue

        # --- TRIGGER: IP BAN (403) or RATE LIMIT (429) ---
        if response.status_code in [403, 429]:
            print(f"!! CRITICAL: IP Rejected (Status {response.status_code}).")
            
            if not use_proxy_mode:
                print("!! SWITCHING TO PROXY MODE NOW.")
                use_proxy_mode = True # <--- THE FLIP SWITCH
                current_proxy = None  # Force a fetch on next loop
                time.sleep(10) # Breathe for a second
                continue # Retry THIS SAME ID with a proxy
            else:
                # If we are ALREADY using a proxy and got banned, that proxy is dead.
                print("!! Current proxy burned. Dumping it.")
                current_proxy = None
                continue


        # --- CASE: SUCCESS (200) ---
        if response.status_code == 200:
            data = response.json().get("data", {})
            published=data.get("published",{})

            if data.get("type") != "Manga":
                print(f"-- ID {manga_id} is {data.get('type')} (Not Manga). Skipping.")
                bad_ids.add(manga_id)
                save_id(BAD_IDS_FILE, manga_id)
                continue

            # Extract Data
            record = {
                "id": data.get("mal_id"),
                "title": data.get("title_english") or data.get("title"),
                "score": data.get("score"),
                "members": data.get("members"),
                "start_date":(published.get("from")) or "Unknown",
                "end_date":(published.get("to")) or "Ongoing",
                "date_in_string":(published.get("string")) or "Unknown",
                "demographic": (data.get("demographics") or [{"name": "Unknown"}])[0]["name"],
                "is_finished": 1 if data.get("status") == "Finished" else 0,
                "magazine":(data.get("serializations") or [{"name": "None"}])[0]["name"],
                "tags": [x["name"] for x in (data.get("genres", []) + data.get("themes", []))]
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

    except requests.exceptions.RequestException as e:
        # This catches DNS errors, Timeouts, and Connection Refused
        print(f"!! Connection Error: {e}")
        consecutive_errors += 1
        if use_proxy_mode:
            current_proxy = None

        # If we fail too many times, check if internet is actually down
        if consecutive_errors > 10 and not use_proxy_mode:
            wait_for_internet() 
            consecutive_errors = 0 # Reset because we just fixed the internet

    except Exception as e:
        print(f"!! Critical Logic Error: {e}")
        # Only break for code bugs, not network bugs
        break

    # ════════════════════════════════════════════════════════════════════════
    # ⚡ ADAPTIVE THROTTLING (SPEED ENGINE)
    # ════════════════════════════════════════════════════════════════════════
    # Default safe speed (if headers are missing)

    sleep_time = 0.35 

    try:
        # Only check headers if we actually got a response
        if 'response' in locals() and response:
            # Jikan sends "X-RateLimit-Remaining" (How many tokens you have left)
            remaining = int(response.headers.get("x-ratelimit-remaining", 5))
            
            if remaining > 10:
                sleep_time = 0.15  # 🚀 TURBO MODE: Plenty of budget, go fast
            elif remaining < 3:
                sleep_time = 2.0   # ⚠️ DANGER ZONE: Almost out, hit the brakes
                print(f"!! Throttling... ({remaining} reqs left)")
    except:
        pass # If math fails, use default 0.35

    time.sleep(sleep_time)


print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                     🏆 MISSION ACCOMPLISHED                                ║
║  Target count reached. Data secured. System going offline.                 ║
╚════════════════════════════════════════════════════════════════════════════╝
""")
print("/n")
print("💤 Initiating system shutdown in 60 seconds...")
print("   (Run 'shutdown /a' in Command Prompt to cancel)")
# The Command: /s = shutdown, /t 60 = timer of 60 seconds
os.system("shutdown /s /t 60")
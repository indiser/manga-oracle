import asyncio
import aiohttp
import random
import json
import os
import time
import sys
import ctypes
from collections import deque

def prevent_sleep():
    if os.name == 'nt':
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
        except: pass
prevent_sleep()
# ════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

# FILES
DATA_FILE = "manga_data_async.jsonl"
BAD_IDS_FILE = "bad_ids_async.txt"
GOOD_IDS_FILE = "good_ids_async.txt"
GOOD_PROXIES_FILE = "proven_proxies_async.txt"

# TARGETS
LOW_RANGE = 1
HIGH_RANGE = 200000
TARGET_COUNT = 50000

# TUNING KNOBS (THE DANGEROUS STUFF)
MAX_CONCURRENT_WORKERS = 50   # How many requests to fire at once
                               # Set to 5-10 if you have few proxies.
                               # Set to 50-100 if you have 100+ proxies.

PROXY_TIMEOUT = 20             # Give cheap proxies 20s to answer
PROXY_LIST_URL = "https://raw.githubusercontent.com/TheSpeedX/SOCKS-List/master/http.txt"

# ════════════════════════════════════════════════════════════════════════════
# 2. THE BRAIN (GLOBAL STATE)
# ════════════════════════════════════════════════════════════════════════════

class ZombieBrain:
    def __init__(self):
        self.proxy_deck = []
        self.veteran_proxies = set()
        self.bad_ids = set()
        self.good_ids = set()
        self.write_buffer = []
        self.success_count = 0
        self.start_time = time.time()
        self.lock = asyncio.Lock() # To prevent file write collisions
        
    def load_memory(self):
        """Loads IDs and Proxies from disk."""
        print("🧠 Loading Memory...")
        if os.path.exists(BAD_IDS_FILE):
            with open(BAD_IDS_FILE, 'r') as f:
                self.bad_ids = {int(x) for x in f if x.strip().isdigit()}
        
        if os.path.exists(GOOD_IDS_FILE):
            with open(GOOD_IDS_FILE, 'r') as f:
                self.good_ids = {int(x) for x in f if x.strip().isdigit()}
                
        if os.path.exists(GOOD_PROXIES_FILE):
            with open(GOOD_PROXIES_FILE, 'r') as f:
                self.veteran_proxies = {x.strip() for x in f if x.strip()}
                self.proxy_deck = list(self.veteran_proxies)
                random.shuffle(self.proxy_deck)
        
        print(f"   -> Loaded {len(self.good_ids)} Good, {len(self.bad_ids)} Bad.")
        print(f"   -> Loaded {len(self.proxy_deck)} Veteran Proxies.")

    async def get_proxy(self):
        """Pops a proxy. If empty, harvests more."""
        while not self.proxy_deck:
            print("!! Proxy Deck Empty. Harvesting...")
            fresh = await self.harvest_proxies()
            if not fresh:
                print("!! Harvest failed. Sleeping 5s...")
                await asyncio.sleep(5)
                continue
            
            # Add only new ones
            for p in fresh:
                if p not in self.veteran_proxies:
                    self.proxy_deck.append(p)
            
            # If we still have nothing, recycle veterans as a last resort
            if not self.proxy_deck and self.veteran_proxies:
                 self.proxy_deck = list(self.veteran_proxies)
                 
            random.shuffle(self.proxy_deck)
            
        return self.proxy_deck.pop()

    async def harvest_proxies(self):
        """Async fetch of the proxy list."""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(PROXY_LIST_URL, timeout=10) as resp:
                    text = await resp.text()
                    return [x.strip() for x in text.split('\n') if x.strip()]
            except:
                return []

    def return_proxy(self, proxy):
        """If a proxy worked, save it and put it back in the deck."""
        self.proxy_deck.append(proxy) # Put back in rotation
        if proxy not in self.veteran_proxies:
            self.veteran_proxies.add(proxy)
            # Fire and forget save (don't await)
            with open(GOOD_PROXIES_FILE, "a") as f: f.write(f"{proxy}\n")

    async def save_data(self, record, manga_id):
        """Buffered save to disk."""
        async with self.lock:
            self.write_buffer.append(record)
            self.good_ids.add(manga_id)
            self.success_count += 1
            
            # Flush every 20 records
            if len(self.write_buffer) >= 20:
                with open(DATA_FILE, "a", encoding="utf-8") as f:
                    for r in self.write_buffer:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                
                with open(GOOD_IDS_FILE, "a") as f:
                    for r in self.write_buffer:
                        f.write(f"{r['id']}\n")
                
                self.write_buffer = [] # Clear
                
                # Simple progress report
                elapsed = time.time() - self.start_time
                rate = self.success_count / elapsed
                print(f"⚡ Speed: {rate:.2f} items/sec | Total: {self.success_count} | Workers: {MAX_CONCURRENT_WORKERS}")

    async def mark_bad(self, manga_id):
        async with self.lock:
            self.bad_ids.add(manga_id)
            with open(BAD_IDS_FILE, "a") as f:
                f.write(f"{manga_id}\n")

# ════════════════════════════════════════════════════════════════════════════
# 3. THE WORKER (ASYNC TASK)
# ════════════════════════════════════════════════════════════════════════════

async def zombie_worker(brain, queue, session):
    """
    A single worker that grabs an ID, grabs a proxy, and tries to eat.
    """
    while True:
        # Get next job
        try:
            manga_id = queue.get_nowait()
        except asyncio.QueueEmpty:
            break # Job done

        proxy = await brain.get_proxy()
        proxy_url = f"http://{proxy}"
        
        try:
            # THE NETWORK CALL (Non-blocking!)
            url = f"https://api.jikan.moe/v4/manga/{manga_id}/full"
            async with session.get(url, proxy=proxy_url, timeout=PROXY_TIMEOUT) as response:
                
                # 404: Does not exist
                if response.status == 404:
                    print(f"   [404] ID {manga_id} dead.")
                    await brain.mark_bad(manga_id)
                    brain.return_proxy(proxy) # Proxy is fine, ID is bad
                    queue.task_done()
                    continue

                # 429/403: Rate Limit or Ban
                if response.status in [429, 403]:
                    # print(f"   [BAN] {proxy} burned.")
                    # Do NOT return proxy. It dies here.
                    await queue.put(manga_id) # Put ID back in queue to try again
                    queue.task_done()
                    continue

                # 200: Success
                if response.status == 200:
                    data = await response.json()
                    data = data.get("data", {})
                    
                    if data.get("type") != "Manga":
                        await brain.mark_bad(manga_id)
                    else:
                        # Extract Data
                        pub = data.get("published", {})
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
                        await brain.save_data(record, manga_id)
                        print(f"++ [200] Captured: {record['title'][:20]}")
                    
                    brain.return_proxy(proxy)
                    queue.task_done()

        except Exception as e:
            # Network error (Timeout, Proxy Refused, etc)
            # print(f"   [ERR] Proxy {proxy} failed: {e}")
            await queue.put(manga_id) # Retry ID
            queue.task_done()

# ════════════════════════════════════════════════════════════════════════════
# 4. ORCHESTRATION
# ════════════════════════════════════════════════════════════════════════════

async def main():
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║               ASYNC ZOMBIE HARVESTER (AIOHTTP)                   ║
    ║               Speed: 50x - 100x vs Synchronous                   ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    brain = ZombieBrain()
    brain.load_memory()
    
    # 1. Fill the Queue
    queue = asyncio.Queue()
    all_ids = set(range(LOW_RANGE, HIGH_RANGE + 1))
    remaining = list(all_ids - brain.good_ids - brain.bad_ids)
    random.shuffle(remaining)
    
    # Limit run to target
    target_list = remaining[:TARGET_COUNT] if len(remaining) > TARGET_COUNT else remaining
    
    print(f"🎯 Queuing {len(target_list)} IDs...")
    for mid in target_list:
        queue.put_nowait(mid)
        
    # 2. Launch the Swarm
    # We use a single ClientSession for connection pooling
    timeout = aiohttp.ClientTimeout(total=PROXY_TIMEOUT)
    connector = aiohttp.TCPConnector(limit=None) # No limit on connector, we limit via workers
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        workers = []
        print(f"🚀 Launching {MAX_CONCURRENT_WORKERS} Concurrent Workers...")
        for _ in range(MAX_CONCURRENT_WORKERS):
            w = asyncio.create_task(zombie_worker(brain, queue, session))
            workers.append(w)
            
        # Wait for queue to empty
        await queue.join()
        
        # Cancel workers once queue is empty
        for w in workers:
            w.cancel()
            
    print("✅ Job Done. Data saved.")

if __name__ == "__main__":
    # Windows Selector Policy fix
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
import asyncio
import aiohttp
import random
import json
import os
import time
import socket
import ctypes
import sys
from aiohttp import ClientTimeout, TCPConnector

# ════════════════════════════════════════════════════════════════════════════
# 0. SYSTEM SAFETY
# ════════════════════════════════════════════════════════════════════════════
def prevent_sleep():
    if os.name == 'nt':
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
        except: pass
prevent_sleep()

# ════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════
DATA_FILE = "manga_data_async_v3.jsonl"
BAD_IDS_FILE = "bad_ids_async_v3.txt"
GOOD_IDS_FILE = "good_ids_async_v3.txt"
GOOD_PROXIES_FILE = "proven_proxies_async_v3.txt"

LOW_RANGE = 1
HIGH_RANGE = 200000
TARGET_COUNT = 50000
MAX_CONCURRENT_WORKERS = 50  # Lowered to prevent choking public proxies
PROXY_TIMEOUT = 15          # Lowered to fail fast

PROXY_SOURCES = [
    "https://raw.githubusercontent.com/TheSpeedX/SOCKS-List/master/http.txt",
    "https://raw.githubusercontent.com/monosans/proxy-list/main/proxies/http.txt",
    "https://raw.githubusercontent.com/mmpx12/proxy-list/master/http.txt",
    "https://raw.githubusercontent.com/shiftytr/proxy-list/master/proxy.txt",
    "https://raw.githubusercontent.com/proxy4parsing/proxy-list/main/http.txt",
    "https://raw.githubusercontent.com/roosterkid/openproxylist/main/HTTPS_RAW.txt"
]

# ════════════════════════════════════════════════════════════════════════════
# 2. THE BRAIN (Now with Graveyard & Monitoring)
# ════════════════════════════════════════════════════════════════════════════
class ZombieBrain:
    def __init__(self):
        self.proxy_deck = []
        self.bad_ids = set()
        self.good_ids = set()
        self.write_buffer = []
        
        # Stats
        self.success_count = 0
        self.failure_count = 0
        self.retry_count = 0
        self.start_time = time.time()
        
        # 🛡️ HEALTH & GRAVEYARD
        self.proxy_health = {} 
        self.proxy_graveyard = set() # Remembers dead proxies so we don't re-add them
        
        self.file_lock = asyncio.Lock()
        self.net_lock = asyncio.Lock()

    def load_memory(self):
        print("🧠 Loading Memory...")
        if os.path.exists(BAD_IDS_FILE):
            with open(BAD_IDS_FILE, 'r') as f:
                self.bad_ids = {int(x) for x in f if x.strip().isdigit()}
        
        if os.path.exists(GOOD_IDS_FILE):
            with open(GOOD_IDS_FILE, 'r') as f:
                self.good_ids = {int(x) for x in f if x.strip().isdigit()}
                
        if os.path.exists(GOOD_PROXIES_FILE):
            with open(GOOD_PROXIES_FILE, 'r') as f:
                for line in f:
                    p = line.strip()
                    if p:
                        self.proxy_health[p] = 5 # Veterans get 5 LIVES
                        self.proxy_deck.append(p)
        
        print(f"   -> Loaded {len(self.good_ids)} Good IDs, {len(self.bad_ids)} Bad IDs.")
        print(f"   -> Loaded {len(self.proxy_deck)} Veteran Proxies.")

    async def get_proxy(self):
        while True:
            if not self.proxy_deck:
                fresh = await self.harvest_proxies()
                if not fresh:
                    print("!! No proxies found. Sleeping 5s...")
                    await asyncio.sleep(5)
                    continue
                
                added = 0
                for p in fresh:
                    # Only add if not in health map AND not in graveyard
                    if p not in self.proxy_health and p not in self.proxy_graveyard:
                        self.proxy_health[p] = 1 # Fresh recruits = 1 Life
                        self.proxy_deck.append(p)
                        added += 1
                
                if added == 0:
                    # We have exhausted the list. Clear graveyard to retry old ones? 
                    # Or just wait. Let's clear graveyard partially if desperate.
                    if len(self.proxy_deck) == 0:
                        print("!! All proxies dead. Clearing Graveyard to retry...")
                        self.proxy_graveyard.clear()
                        await asyncio.sleep(2)
                        continue

                random.shuffle(self.proxy_deck)
            
            # Pop a proxy. Verify it still has health (in case async weirdness)
            try:
                p = self.proxy_deck.pop()
                if p in self.proxy_health:
                    return p
            except IndexError:
                continue

    async def report_proxy_status(self, proxy, success):
        if success:
            if self.proxy_health.get(proxy, 0) < 3:
                with open(GOOD_PROXIES_FILE, "a") as f: f.write(f"{proxy}\n")
            self.proxy_health[proxy] = 3 
            self.proxy_deck.append(proxy)
        else:
            current = self.proxy_health.get(proxy, 1) - 1
            self.proxy_health[proxy] = current
            if current > 0:
                self.proxy_deck.append(proxy)
            else:
                # Kill it and bury it
                self.proxy_health.pop(proxy, None)
                self.proxy_graveyard.add(proxy)

    async def harvest_proxies(self):
        print(f"🌍 Harvesting from {len(PROXY_SOURCES)} sources...")
        found_proxies = set()
        
        async def fetch_source(session, url):
            try:
                async with session.get(url, timeout=10) as resp:
                    if resp.status == 200:
                        text = await resp.text()
                        # Extract non-empty lines
                        return {x.strip() for x in text.split('\n') if x.strip() and ":" in x}
            except: 
                return set()
            return set()

        async with aiohttp.ClientSession() as session:
            tasks = [fetch_source(session, url) for url in PROXY_SOURCES]
            results = await asyncio.gather(*tasks)
            
            for res in results:
                found_proxies.update(res)

        print(f"   -> Aggregated {len(found_proxies)} unique candidates.")
        return list(found_proxies)

    def get_random_headers(self):
        return {"User-Agent": random.choice([
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5.2 Safari/605.1.15"
        ])}

    async def save_data(self, record, manga_id):
        async with self.file_lock:
            self.write_buffer.append(record)
            self.good_ids.add(manga_id)
            self.success_count += 1
            if len(self.write_buffer) >= 50: # Write more often
                with open(DATA_FILE, "a", encoding="utf-8") as f:
                    for r in self.write_buffer: f.write(json.dumps(r, ensure_ascii=False) + "\n")
                with open(GOOD_IDS_FILE, "a") as f:
                    for r in self.write_buffer: f.write(f"{r['id']}\n")
                self.write_buffer = []

    async def mark_bad(self, manga_id):
        async with self.file_lock:
            self.bad_ids.add(manga_id)
            with open(BAD_IDS_FILE, "a") as f: f.write(f"{manga_id}\n")
            self.success_count += 1 # Technically a success (we processed it)


    async def check_internet(self):
        async with self.net_lock:
            # Check if we are already back online before pinging
            try:
                loop = asyncio.get_running_loop()
                # 1 second timeout is enough for 8.8.8.8
                await asyncio.wait_for(
                    loop.run_in_executor(None, socket.create_connection, ("8.8.8.8", 53)), 
                    timeout=2.0
                )
            except (OSError, asyncio.TimeoutError):
                print("\n🚨 NETWORK LOST. Hibernating... (Ctrl+C to stop)")
                while True:
                    try:
                        await asyncio.sleep(5)
                        await loop.run_in_executor(None, socket.create_connection, ("8.8.8.8", 53))
                        print("✅ NETWORK RESTORED. Resuming...")
                        return
                    except: 
                        pass

# ════════════════════════════════════════════════════════════════════════════
# 3. THE MONITOR (NEW)
# ════════════════════════════════════════════════════════════════════════════
async def monitor_loop(brain, queue):
    """Prints live stats every 2 seconds so you know it's not dead."""
    while True:
        elapsed = time.time() - brain.start_time
        rate = brain.success_count / elapsed if elapsed > 0 else 0
        q_size = queue.qsize()
        active = len(brain.proxy_health)
        
        sys.stdout.write(
            f"\r[T+{int(elapsed)}s] ✅ OK: {brain.success_count} | ❌ Fail: {brain.failure_count} | ♻️ Retry: {brain.retry_count} | 📉 Q: {q_size} | 🛡️ Proxies: {active} | ⚡ {rate:.1f}/s   "
        )
        sys.stdout.flush()
        await asyncio.sleep(2)

# ════════════════════════════════════════════════════════════════════════════
# 4. THE WORKER
# ════════════════════════════════════════════════════════════════════════════
async def zombie_worker(brain, queue, session):
    while True:
        try: manga_id = queue.get_nowait()
        except: break

        proxy = await brain.get_proxy()
        
        try:
            async with session.get(
                f"https://api.jikan.moe/v4/manga/{manga_id}/full",
                proxy=f"http://{proxy}",
                headers=brain.get_random_headers(),
                timeout=PROXY_TIMEOUT
            ) as response:
                
                if response.status == 404:
                    await brain.mark_bad(manga_id)
                    await brain.report_proxy_status(proxy, success=True)
                    queue.task_done()
                    continue

                if response.status in [403, 429]:
                    brain.failure_count += 1
                    brain.retry_count += 1
                    await brain.report_proxy_status(proxy, success=False)
                    await queue.put(manga_id) # Retry
                    queue.task_done()
                    continue

                if response.status == 200:
                    try:
                        raw = await response.json()
                        data = raw.get("data", {})
                        if data.get("type") != "Manga":
                            await brain.mark_bad(manga_id)
                        else:
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
                        
                        await brain.report_proxy_status(proxy, success=True)
                        queue.task_done()
                    except:
                        # JSON decode error or similar
                        brain.failure_count += 1
                        await brain.report_proxy_status(proxy, success=False)
                        await queue.put(manga_id)
                        queue.task_done()
                else:
                    # 500s, 503s, etc
                    brain.failure_count += 1
                    await brain.report_proxy_status(proxy, success=False)
                    await queue.put(manga_id)
                    queue.task_done()

        except Exception:
            await brain.check_internet()
            brain.failure_count += 1
            await brain.report_proxy_status(proxy, success=False)
            await queue.put(manga_id)
            queue.task_done()

# ════════════════════════════════════════════════════════════════════════════
# 5. LAUNCHER
# ════════════════════════════════════════════════════════════════════════════
async def main():
    brain = ZombieBrain()
    brain.load_memory()
    
    queue = asyncio.Queue()
    # Logic to only load IDs we haven't checked
    ids = list(set(range(LOW_RANGE, HIGH_RANGE + 1)) - brain.good_ids - brain.bad_ids)
    random.shuffle(ids)
    
    # Fill Queue
    print(f"📦 Queuing {min(TARGET_COUNT, len(ids))} items...")
    for i in ids[:TARGET_COUNT]: queue.put_nowait(i)
    
    connector = TCPConnector(limit=None)
    timeout = ClientTimeout(total=PROXY_TIMEOUT)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # Launch Monitor
        monitor = asyncio.create_task(monitor_loop(brain, queue))
        
        # Launch Workers
        workers = [asyncio.create_task(zombie_worker(brain, queue, session)) for _ in range(MAX_CONCURRENT_WORKERS)]
        
        await queue.join()
        
        monitor.cancel()
        for w in workers: w.cancel()

if __name__ == "__main__":
    # CRITICAL FIX FOR WINDOWS SOCKET LIMIT
    if os.name == 'nt': 
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    try: asyncio.run(main())
    except KeyboardInterrupt: print("\n🛑 Stopped by user.")
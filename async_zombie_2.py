import asyncio
import aiohttp
import random
import json
import os
import time
import socket
import ctypes
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
DATA_FILE = "manga_data_async_2.jsonl"
BAD_IDS_FILE = "bad_ids_async_2.txt"
GOOD_IDS_FILE = "good_ids_async_2.txt"
GOOD_PROXIES_FILE = "proven_proxies_async_2.txt"

LOW_RANGE = 1
HIGH_RANGE = 200000
TARGET_COUNT = 50000
MAX_CONCURRENT_WORKERS = 100
PROXY_TIMEOUT = 25
PROXY_LIST_URL = "https://raw.githubusercontent.com/TheSpeedX/SOCKS-List/master/http.txt"

# ════════════════════════════════════════════════════════════════════════════
# 2. THE BRAIN (Now with Health Tracking)
# ════════════════════════════════════════════════════════════════════════════
class ZombieBrain:
    def __init__(self):
        self.proxy_deck = []
        self.bad_ids = set()
        self.good_ids = set()
        self.write_buffer = []
        self.success_count = 0
        self.start_time = time.time()
        
        # 🛡️ THE RANKING SYSTEM (No SQL needed)
        # Format: { "192.168.1.1": 3 } (3 Lives Left)
        self.proxy_health = {} 
        
        self.file_lock = asyncio.Lock()
        self.net_lock = asyncio.Lock()

    def load_memory(self):
        print("🧠 Loading Memory & Ranks...")
        if os.path.exists(BAD_IDS_FILE):
            with open(BAD_IDS_FILE, 'r') as f:
                self.bad_ids = {int(x) for x in f if x.strip().isdigit()}
        
        if os.path.exists(GOOD_IDS_FILE):
            with open(GOOD_IDS_FILE, 'r') as f:
                self.good_ids = {int(x) for x in f if x.strip().isdigit()}
                
        # Load Veterans and give them 3 LIVES (High Rank)
        if os.path.exists(GOOD_PROXIES_FILE):
            with open(GOOD_PROXIES_FILE, 'r') as f:
                for line in f:
                    p = line.strip()
                    if p:
                        self.proxy_health[p] = 3 # Veteran Start Health
                        self.proxy_deck.append(p)
                        
        random.shuffle(self.proxy_deck)
        print(f"   -> Loaded {len(self.proxy_deck)} Veterans (Rank 1).")

    async def get_proxy(self):
        """Returns a proxy. Refills if empty."""
        while not self.proxy_deck:
            print("!! Deck Empty. Recruiting Fresh Proxies...")
            fresh = await self.harvest_proxies()
            if not fresh:
                await asyncio.sleep(5)
                continue
            
            for p in fresh:
                # Only add if we don't know it
                if p not in self.proxy_health:
                    self.proxy_health[p] = 1 # Fresh Recruits get 1 LIFE (Rank 2)
                    self.proxy_deck.append(p)
            
            random.shuffle(self.proxy_deck)
            
        return self.proxy_deck.pop()

    async def report_proxy_status(self, proxy, success):
        """
        ⚖️ THE JUDGE
        Decides if a proxy lives or dies based on its Rank (Health).
        """
        if success:
            # Reward: Heal to Max Health (3)
            # If it was a Fresh recruit (1 life), it is promoted to Veteran (3 lives)
            if self.proxy_health.get(proxy, 0) < 3:
                # Save to disk permanently as a Veteran
                with open(GOOD_PROXIES_FILE, "a") as f: f.write(f"{proxy}\n")
            
            self.proxy_health[proxy] = 3 
            self.proxy_deck.append(proxy) # Return to deck
            
        else:
            # Punishment: Lose 1 Life
            current_health = self.proxy_health.get(proxy, 1) - 1
            self.proxy_health[proxy] = current_health
            
            if current_health > 0:
                # It survived! Put it back in the deck.
                # print(f"   ⚠️ Proxy {proxy} took a hit! (Lives left: {current_health})")
                self.proxy_deck.append(proxy)
            else:
                # It died. Delete from memory.
                # print(f"   💀 Proxy {proxy} executed.")
                self.proxy_health.pop(proxy, None)

    async def harvest_proxies(self):
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(PROXY_LIST_URL, timeout=10) as resp:
                    text = await resp.text()
                    return [x.strip() for x in text.split('\n') if x.strip()]
            except: return []

    def get_random_headers(self):
        return {"User-Agent": random.choice([
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15"
        ])}

    async def save_data(self, record, manga_id):
        async with self.file_lock:
            self.write_buffer.append(record)
            self.good_ids.add(manga_id)
            self.success_count += 1
            if len(self.write_buffer) >= 20:
                with open(DATA_FILE, "a", encoding="utf-8") as f:
                    for r in self.write_buffer: f.write(json.dumps(r, ensure_ascii=False) + "\n")
                with open(GOOD_IDS_FILE, "a") as f:
                    for r in self.write_buffer: f.write(f"{r['id']}\n")
                self.write_buffer = []
                elapsed = time.time() - self.start_time
                print(f"⚡ Rate: {self.success_count / elapsed:.2f}/s | Total: {self.success_count} | Active Proxies: {len(self.proxy_health)}")

    async def mark_bad(self, manga_id):
        async with self.file_lock:
            self.bad_ids.add(manga_id)
            with open(BAD_IDS_FILE, "a") as f: f.write(f"{manga_id}\n")
    
    async def check_internet(self):
        async with self.net_lock:
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, socket.create_connection, ("8.8.8.8", 53))
            except OSError:
                print("🚨 NETWORK LOST. Hibernating...")
                while True:
                    try:
                        await loop.run_in_executor(None, socket.create_connection, ("8.8.8.8", 53))
                        print("✅ RESTORED.")
                        return
                    except: await asyncio.sleep(10)

# ════════════════════════════════════════════════════════════════════════════
# 3. THE WORKER
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
                    await brain.report_proxy_status(proxy, success=True) # Proxy worked, ID dead
                    queue.task_done()
                    continue

                if response.status in [403, 429]:
                    await brain.report_proxy_status(proxy, success=False) # Strike 1
                    await queue.put(manga_id)
                    queue.task_done()
                    continue

                if response.status == 200:
                    raw = await response.json()
                    data = raw.get("data", {})
                    if data.get("type") != "Manga":
                        await brain.mark_bad(manga_id)
                    else:
                        # ... (Same Extraction Logic as before) ...
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
                    
                    await brain.report_proxy_status(proxy, success=True) # HEAL
                    queue.task_done()

        except Exception:
            # If network error, check if it's the proxy or the internet
            await brain.check_internet()
            await brain.report_proxy_status(proxy, success=False) # Strike 1
            await queue.put(manga_id)
            queue.task_done()

# ════════════════════════════════════════════════════════════════════════════
# 4. LAUNCHER
# ════════════════════════════════════════════════════════════════════════════
async def main():
    brain = ZombieBrain()
    brain.load_memory()
    
    queue = asyncio.Queue()
    ids = list(set(range(LOW_RANGE, HIGH_RANGE + 1)) - brain.good_ids - brain.bad_ids)
    random.shuffle(ids)
    for i in ids[:TARGET_COUNT]: queue.put_nowait(i)
    
    connector = TCPConnector(limit=None)
    timeout = ClientTimeout(total=PROXY_TIMEOUT)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        workers = [asyncio.create_task(zombie_worker(brain, queue, session)) for _ in range(MAX_CONCURRENT_WORKERS)]
        await queue.join()
        for w in workers: w.cancel()

if __name__ == "__main__":
    if os.name == 'nt': asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try: asyncio.run(main())
    except: pass


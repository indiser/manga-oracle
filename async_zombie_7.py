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
# 1. CONFIGURATION (TUNED TO YOUR "FASTEST" SETTINGS)
# ════════════════════════════════════════════════════════════════════════════
DATA_FILE = "manga_data_final.jsonl"
BAD_IDS_FILE = "bad_ids_final.txt"
GOOD_IDS_FILE = "good_ids_final.txt"
GOOD_PROXIES_FILE = "proven_proxies_final.txt"

LOW_RANGE = 1
HIGH_RANGE = 300000
TARGET_COUNT = 50000

# 🚀 THE WINNING SETTINGS (From async_zombie_3)
MAX_CONCURRENT_WORKERS = 100 
PROXY_TIMEOUT = 25  

# 🚀 MULTIPLE SOURCES (Infinite Ammo)
PROXY_SOURCES = [
    "https://raw.githubusercontent.com/TheSpeedX/SOCKS-List/master/http.txt",
    "https://raw.githubusercontent.com/monosans/proxy-list/main/proxies/http.txt",
    "https://raw.githubusercontent.com/mmpx12/proxy-list/master/http.txt",
    "https://raw.githubusercontent.com/shiftytr/proxy-list/master/proxy.txt",
    "https://raw.githubusercontent.com/proxy4parsing/proxy-list/main/http.txt"
]

# ════════════════════════════════════════════════════════════════════════════
# 2. THE BRAIN (SMART LOGIC MERGED)
# ════════════════════════════════════════════════════════════════════════════
class ZombieBrain:
    def __init__(self):
        self.proxy_deck = []
        self.bad_ids = set()
        self.good_ids = set()
        self.write_buffer = []
        
        # Counters
        self.session_count = 0
        self.start_time = time.time()
        
        self.proxy_health = {} 
        self.file_lock = asyncio.Lock()
        self.net_lock = asyncio.Lock()
        self.harvest_lock = asyncio.Lock() # 🔒 Prevents Log Spam

    def load_memory(self):
        print("🧠 Loading Memory & Ranks...")
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
                        self.proxy_health[p] = 3
                        self.proxy_deck.append(p)
                        
        random.shuffle(self.proxy_deck)
        print(f"   -> Memory Loaded. Good: {len(self.good_ids)} | Bad: {len(self.bad_ids)}")
        print(f"   -> Veterans Loaded: {len(self.proxy_deck)}")

    async def get_proxy(self):
        while not self.proxy_deck:
            # 🔒 HARVEST LOCK: Only 1 worker can harvest at a time
            async with self.harvest_lock:
                if self.proxy_deck: break 

                print("!! Deck Empty. Recruiting from ALL sources...")
                fresh = await self.harvest_proxies()
                
                if not fresh:
                    print("!! All sources dry. Sleeping 10s...")
                    await asyncio.sleep(10)
                    continue
                
                new_count = 0
                for p in fresh:
                    if p not in self.proxy_health:
                        self.proxy_health[p] = 1 
                        self.proxy_deck.append(p)
                        new_count += 1
                
                if new_count > 0:
                    print(f"   -> Recruited {new_count} fresh proxies.")
                    random.shuffle(self.proxy_deck)
                else:
                    print("   (No new proxies found. Waiting for veterans to return...)")
                    await asyncio.sleep(2)

        return self.proxy_deck.pop()

    async def harvest_proxies(self):
        fresh_proxies = set()
        async with aiohttp.ClientSession() as session:
            for url in PROXY_SOURCES:
                try:
                    async with session.get(url, timeout=10) as resp:
                        if resp.status == 200:
                            text = await resp.text()
                            lines = text.split('\n')
                            for line in lines:
                                p = line.strip()
                                if ":" in p: fresh_proxies.add(p)
                except: pass
        return list(fresh_proxies)

    async def report_proxy_status(self, proxy, success):
        if success:
            if self.proxy_health.get(proxy, 0) < 3:
                with open(GOOD_PROXIES_FILE, "a") as f: f.write(f"{proxy}\n")
            self.proxy_health[proxy] = 3 
            self.proxy_deck.append(proxy) 
        else:
            current_health = self.proxy_health.get(proxy, 1) - 1
            self.proxy_health[proxy] = current_health
            if current_health > 0:
                self.proxy_deck.append(proxy)
            else:
                self.proxy_health.pop(proxy, None)
                
    def get_random_headers(self):
        return {"User-Agent": random.choice([
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15"
        ])}

    def print_hud(self):
        elapsed = time.time() - self.start_time
        rate = self.session_count / elapsed if elapsed > 0 else 0
        total_range = HIGH_RANGE - LOW_RANGE + 1
        total_done = len(self.good_ids) + len(self.bad_ids)
        remaining = total_range - total_done
        print(f"⚡ Rate: {rate:.2f}/s | Good: {len(self.good_ids)} | Bad: {len(self.bad_ids)} | Remaining: {remaining}")

    async def save_data(self, record, manga_id):
        async with self.file_lock:
            self.write_buffer.append(record)
            self.good_ids.add(manga_id)
            self.session_count += 1
            
            if len(self.write_buffer) >= 20:
                with open(DATA_FILE, "a", encoding="utf-8") as f:
                    for r in self.write_buffer: f.write(json.dumps(r, ensure_ascii=False) + "\n")
                with open(GOOD_IDS_FILE, "a") as f:
                    for r in self.write_buffer: f.write(f"{r['id']}\n")
                self.write_buffer = []
                self.print_hud()

    async def mark_bad(self, manga_id):
        async with self.file_lock:
            self.bad_ids.add(manga_id)
            with open(BAD_IDS_FILE, "a") as f: f.write(f"{manga_id}\n")
            if len(self.bad_ids) % 50 == 0: self.print_hud()
    
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
                    await brain.report_proxy_status(proxy, success=True) 
                    queue.task_done()
                    continue

                if response.status in [403, 429]:
                    await brain.report_proxy_status(proxy, success=False)
                    await queue.put(manga_id)
                    queue.task_done()
                    continue

                if response.status == 200:
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
                        print(f"++ [200] Captured: {record['title'][:20]}")
                    
                    await brain.report_proxy_status(proxy, success=True) 
                    queue.task_done()

        except Exception:
            await brain.check_internet()
            await brain.report_proxy_status(proxy, success=False)
            await queue.put(manga_id)
            queue.task_done()

# ════════════════════════════════════════════════════════════════════════════
# 4. LAUNCHER (SMART GOAL EDITION)
# ════════════════════════════════════════════════════════════════════════════
async def main():
    brain = ZombieBrain()
    brain.load_memory()
    
    already_collected = len(brain.good_ids)
    needed = TARGET_COUNT - already_collected
    
    print(f"\n📊 GLOBAL STATUS REPORT:")
    print(f"   • Goal:      {TARGET_COUNT}")
    print(f"   • Current:   {already_collected}")
    print(f"   • Needed:    {needed}")
    
    if needed <= 0:
        print("\n🎉 TARGET REACHED! You already have enough data.")
        return

    queue = asyncio.Queue()
    available_ids = list(set(range(LOW_RANGE, HIGH_RANGE + 1)) - brain.good_ids - brain.bad_ids)
    random.shuffle(available_ids)
    
    if not available_ids:
        print("\n⚠️ RANGE EXHAUSTED! Increase HIGH_RANGE.")
        return

    limit = min(len(available_ids), needed)
    for i in available_ids[:limit]: queue.put_nowait(i)
    
    print(f"\n🚀 Launching Mission: Fetching {limit} items...")
    
    connector = TCPConnector(limit=None)
    timeout = ClientTimeout(total=PROXY_TIMEOUT)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        workers = [asyncio.create_task(zombie_worker(brain, queue, session)) for _ in range(MAX_CONCURRENT_WORKERS)]
        await queue.join()
        for w in workers: w.cancel()

    print("\n✅ Session Complete.")

if __name__ == "__main__":
    if os.name == 'nt': asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try: asyncio.run(main())
    except: pass
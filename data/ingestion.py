import os
import math
import random
import asyncio
import aiohttp
import numpy as np
import webdataset as wds
import yaml
import json

def get_dir_size_gb(directory):
    total = 0
    if not os.path.exists(directory): return 0
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total += os.path.getsize(fp)
    return total / (1024**3)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # Earth radius in metres
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    return 2 * R * math.asin(math.sqrt(a))

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
         return yaml.safe_load(f)

async def fetch_mapillary_sequences(session, access_token, lat, lon, search_radius_m=2000):
    base_url = "https://graph.mapillary.com/images"
    lat_delta = search_radius_m / 111320.0
    lon_delta = search_radius_m / (111320.0 * math.cos(math.radians(lat)))
    bbox = f"{lon - lon_delta},{lat - lat_delta},{lon + lon_delta},{lat + lat_delta}"
    
    params = {
        "access_token": access_token,
        "fields": "id,geometry,thumb_1024_url,sequence,captured_at",
        "bbox": bbox,
        "limit": 1000
    }
    
    for attempt in range(5):
        try:
            async with session.get(base_url, params=params, timeout=15) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("data", [])
                else:
                    if response.status != 429:
                        txt = await response.text()
                        print(f"  [API Error] {response.status}: {txt[:100]}")
                        return []
                    await asyncio.sleep(2 ** attempt + random.uniform(0, 1))
        except Exception as e:
            if attempt == 4: print(f"  [Req Error] {e}")
            await asyncio.sleep(2 ** attempt)
    return []

async def download_image(session, url, dl_sem):
    async with dl_sem:
        for attempt in range(2):
            try:
                async with session.get(url, timeout=10) as resp:
                    if resp.status == 200: return await resp.read()
            except Exception:
                await asyncio.sleep(0.1)
    return None

async def async_main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config = load_config(os.path.join(root, "config.yaml"))
    coords = np.load(os.path.join(root, "data", "h3_coords_res7.npy")).tolist()
    shards_dir = os.path.join(root, config["data"]["output_shards_dir"])
    os.makedirs(shards_dir, exist_ok=True)
    
    access_token = config["api"]["mapillary_access_token"]
    state_file = os.path.join(root, "data", "ingestion_state.json")
    start_idx = 0
    if os.path.exists(state_file):
        with open(state_file, "r") as f: start_idx = json.load(f).get("last_idx", 0)
            
    if start_idx >= len(coords):
        print(f"[Ingestion] Geographic grid already complete ({start_idx}/{len(coords)}). Exiting.")
        return

    existing_shards = [f for f in os.listdir(shards_dir) if f.startswith("dataset-rectilinear-") and f.endswith(".tar")]
    start_shard_idx = 0
    if existing_shards:
        indices = []
        for f in existing_shards:
            try:
                indices.append(int(f.split('-')[-1].split('.')[0]))
            except ValueError:
                pass
        if indices:
            start_shard_idx = max(indices) + 1

    writer = wds.ShardWriter(os.path.join(shards_dir, "dataset-rectilinear-%06d.tar"), 
                            maxsize=int(config["data"]["shard_max_size_gb"] * 1e9), 
                            maxcount=5000, start_shard=start_shard_idx)
            
    max_cache_gb = config.get("orchestrator", {}).get("max_disk_cache_gb", 500.0)
    quota_per_cell = 200 
    
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        api_sem = asyncio.Semaphore(10) # Throttle API requests
        dl_sem = asyncio.Semaphore(50)  # Throttle concurrent downloads
        
        async def process_cell(idx, lat, lon):
            async with api_sem:
                images = await fetch_mapillary_sequences(session, access_token, lat, lon)
                if not images: return 0
                
                seq_map = {}
                for feat in images:
                    s_id = feat.get("sequence")
                    url = feat.get("thumb_1024_url")
                    geom = feat.get("geometry", {}).get("coordinates")
                    if s_id and url and geom:
                        seq_map.setdefault(s_id, []).append({"id": feat["id"], "url": url, "lat": geom[1], "lon": geom[0], "ts": feat.get("captured_at")})
                
                pairs = []
                for s_id, items in seq_map.items():
                    if len(pairs) >= quota_per_cell: break
                    items.sort(key=lambda x: x["ts"] or x["id"])
                    for i in range(len(items)):
                        if len(pairs) >= quota_per_cell: break
                        for j in range(i + 1, min(i + 20, len(items))):
                            d = haversine(items[i]["lat"], items[i]["lon"], items[j]["lat"], items[j]["lon"])
                            if 5.0 <= d <= 15.0:
                                pairs.append((items[i], items[j]))
                                break
                
                if not pairs: return 0

                async def _download_pair(p1, p2):
                    d1, d2 = await asyncio.gather(download_image(session, p1["url"], dl_sem), download_image(session, p2["url"], dl_sem))
                    if d1 and d2:
                        return {"key": f"{p1['id']}_{p2['id']}", "img1.jpg": d1, "img2.jpg": d2, "meta.json": {"id1": p1["id"], "id2": p2["id"], "lat": p1["lat"], "lon": p1["lon"]}}
                    return None
                
                results = await asyncio.gather(*[_download_pair(p1, p2) for p1, p2 in pairs])
                processed = [r for r in results if r]
                for p in processed:
                    writer.write({"__key__": p["key"], "img1.jpg": p["img1.jpg"], "img2.jpg": p["img2.jpg"], "meta.json": p["meta.json"]})
                return len(processed)

        batch_size = 5
        for batch_start in range(start_idx, len(coords), batch_size):
            batch_coords = coords[batch_start:batch_start+batch_size]
            print(f"[Ingestion] Scanning Batch {batch_start//batch_size} (Indices {batch_start}-{batch_start+batch_size})...", flush=True)
            
            results = await asyncio.gather(*[process_cell(batch_start+i, c[0], c[1]) for i, c in enumerate(batch_coords)])
            total_pairs = sum(results)
            
            current_cache = get_dir_size_gb(shards_dir)
            print(f"[Ingestion] Batch complete. Yield: {total_pairs} pairs. Total cache: {current_cache:.2f} GB", flush=True)
            with open(state_file, "w") as f: json.dump({"last_idx": batch_start + batch_size}, f)
            if current_cache > max_cache_gb: break
            
    writer.close()

if __name__ == "__main__":
    asyncio.run(async_main())

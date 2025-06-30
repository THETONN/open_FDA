import pandas as pd
import requests
from tqdm.auto import tqdm
import time
import asyncio
import aiohttp

INPUT_CSV = "../../data/v2/drug_clean_search_v2.csv"
OUT_SEARCH_CSV = "../../data/v2/drug_search_with_rxcui.csv"
OUT_RXCUI_DETAILS_CSV = "../../data/v2/drug_rxcui_details.csv"
drug_df = pd.read_csv(INPUT_CSV)

def split_and_strip_names(name):
    # Split by comma, strip whitespace, ignore empty
    return [n.strip() for n in str(name).split(",") if n.strip()]

async def fetch_rxcui_single(session, sub_name):
    url = f"https://rxnav.nlm.nih.gov/REST/rxcui.json?name={sub_name}"
    try:
        async with session.get(url, timeout=10) as resp:
            data = await resp.json()
            rxids = data.get("idGroup", {}).get("rxnormId", [])
            if isinstance(rxids, str):
                rxids = [rxids]
            return rxids
    except Exception:
        return []

async def fetch_rxcui(session, name):
    rxids_total = []
    for sub_name in split_and_strip_names(name):
        rxids = await fetch_rxcui_single(session, sub_name)
        rxids_total.extend(rxids)
    rxids_total = list(sorted(set(rxids_total)))
    return name, rxids_total

async def main(drug_names, concurrency=10):
    results = []
    connector = aiohttp.TCPConnector(limit_per_host=concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [fetch_rxcui(session, name) for name in drug_names]
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Searching rxcui"):
            name, rxids = await f
            flag = 1 if rxids else 0
            results.append({
                'medicinal_product_clean': name,
                'rxcui': ",".join(rxids) if rxids else "",
                'flag': flag
            })
    return results

def run_async_safely(coro):
    """Run async coroutine safely in both Jupyter and standalone environments"""
    try:
        asyncio.get_running_loop()
        # Jupyter case
        try:
            import nest_asyncio
            nest_asyncio.apply()
            return asyncio.run(coro)
        except ImportError:
            import threading
            import concurrent.futures
            def run_in_thread():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(coro)
                finally:
                    new_loop.close()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                return future.result()
    except RuntimeError:
        return asyncio.run(coro)

drug_names = drug_df['medicinal_product_clean'].tolist()
results = run_async_safely(main(drug_names, concurrency=10))

search_df = pd.DataFrame(results)
search_df.to_csv(OUT_SEARCH_CSV, index=False, na_rep="")
print(f"Saved {OUT_SEARCH_CSV}")

# ===== 2. PULL RXCUI DETAILS =====
# รวม rxcui ที่พบทั้งหมด
all_rx_ids = []
for ids in search_df["rxcui"]:
    for rid in ids.split(","):
        if rid.strip():
            all_rx_ids.append(rid.strip())
all_rx_ids = sorted(set(all_rx_ids))

def get_rxcui_details(rxcui):
    url = f"https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/allrelated.json"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        cg = data.get("allRelatedGroup", {}).get("conceptGroup", [])
        out = []
        for group in cg:
            tty = group.get("tty", "")
            for cp in group.get("conceptProperties", []) or []:
                out.append({
                    "query_rxcui": rxcui,
                    "type": tty,
                    "rxcui": cp.get("rxcui", ""),
                    "name": cp.get("name", ""),
                    "synonym": cp.get("synonym", ""),
                    "suppress": cp.get("suppress", ""),
                })
        return out
    except Exception as e:
        print(f"[!] rxcui {rxcui}: {e}")
        return []

# ดึงรายละเอียด rxcui ทั้งหมด
all_details = []
for rid in tqdm(all_rx_ids, desc="Getting rxcui details"):
    details = get_rxcui_details(rid)
    all_details.extend(details)
    time.sleep(0.1)

details_df = pd.DataFrame(all_details)
details_df.to_csv(OUT_RXCUI_DETAILS_CSV, index=False)
print(f"Saved {OUT_RXCUI_DETAILS_CSV} ({len(details_df)} rows)")

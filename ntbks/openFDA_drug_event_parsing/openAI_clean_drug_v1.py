
import asyncio, json, re, textwrap, logging
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
from aiohttp import ClientError
from openai import AsyncOpenAI, OpenAIError, APIConnectionError, RateLimitError, APITimeoutError, APIStatusError
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

BASE            = Path("../../data/v1/")
BASE.mkdir(parents=True, exist_ok=True)
RAW_GZ          = "../../data/pediatric_patients_report_drug_reaction.csv.gz"
CACHE           = BASE / "_drug_cache_v1.parquet"
CLEAN_DRUG_CSV  = BASE / "clean_drug.csv"
CLEAN_DRUG_XLSX = BASE / "clean_drug.xlsx"
PED_NEW_GZ      = BASE / "ped_data_new.csv.gz"
DRUG_SEARCH_CSV = BASE / "drug_clean_search.csv"
DRUG_SEARCH_XLSX= BASE / "drug_clean_search.xlsx"
FALLBACK_LOG    = BASE / "fallback_v1.txt"
MISMATCH_LOG    = BASE / "gpt_batch_mismatch_v1.txt"
API_KEY         = Path("../../api_key.txt").read_text().strip()
oai             = AsyncOpenAI(api_key=API_KEY)

NOISE = re.compile(
    r"\b(?:tab(?:let)?s?|caps?(?:ule)?s?|inj(?:ection)?|soln|amp|vial|"
    r"susp|syrup|oral|iv|im|sc)\b|[®™]|[\(\)\[\]]|[0-9]+ ?mg\b",
    flags=re.I,
)
def strip_noise(txt: str) -> str:
    t = re.sub(r"[.,]+$", "", txt.strip())
    t = NOISE.sub(" ", t)
    return re.sub(r"\s{2,}", " ", t).strip().lower()

SYSTEM_PROMPT = textwrap.dedent("""\
    You receive a JSON array of up to 10 raw drug product strings (never duplicates).
    Return a JSON array of the same length, in the same order.
    Each element must be canonical, lowercase INN/USAN ingredient names (comma-separated if multiple).
    If you cannot normalise, return the raw string, lowercased, exactly.
    Strict JSON. No comments.
""")

@retry(
    reraise=True,
    retry=retry_if_exception_type((OpenAIError, ClientError, APIConnectionError,
                                   RateLimitError, APITimeoutError, APIStatusError, ValueError)),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    stop=stop_after_attempt(6),
)
async def gpt_clean(batch: list[str]) -> list[str]:
    rsp = await oai.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=400,
        response_format={"type": "json_object"},
        messages=[
            {"role":"system","content":SYSTEM_PROMPT},
            {"role":"user",  "content":json.dumps(batch, ensure_ascii=False)},
        ],
    )
    data = json.loads(rsp.choices[0].message.content)
    if isinstance(data, list):
        arr = data
    elif isinstance(data, dict):
        lst = next((v for v in data.values() if isinstance(v, list)), None)
        if lst is not None:
            arr = lst
        elif all(isinstance(v, str) for v in data.values()):
            try:
                arr = [data[str(i)] for i in range(len(batch))]
            except KeyError:
                arr = list(data.values())
        else:
            raise ValueError("Unsupported JSON schema")
    else:
        raise ValueError("Unexpected JSON type")
    return arr

def split_no_dupes(lst, batch_size):
    out = []
    batch = []
    seen = set()
    for x in lst:
        if x in seen or len(batch) == batch_size:
            out.append(batch)
            batch, seen = [], set()
        batch.append(x)
        seen.add(x)
    if batch: out.append(batch)
    return out

print("Loading ped_data …")
ped = pd.read_csv(RAW_GZ, compression="gzip", index_col=0)
unique_raw = ped["medicinal_product"].dropna().unique().tolist()
print(f"   rows={len(ped):,}, unique_raw={len(unique_raw):,}")

if CACHE.exists():
    known = dict(pd.read_parquet(CACHE).values)
    print(f"Loaded cache v1: {len(known):,} names")
else:
    known = {}
    print("No cache starting fresh")

BATCH = 10
CONC  = 2
sem = asyncio.Semaphore(CONC)
lock = asyncio.Lock()

async def process(batch: list[str], sample=False):
    pre_clean = [strip_noise(x) for x in batch]
    async with sem:
        try:
            cleaned = await gpt_clean(pre_clean)
        except Exception as e:
            print("GPT fallback (batch):", e)
            cleaned = pre_clean
    for r, c in zip(batch, cleaned):
        if not c or c.lower() == "unknown":
            c = r
        known[r] = c
    async with lock:
        pd.DataFrame(known.items(), columns=["raw","cleaned"]).to_parquet(CACHE, index=False)

async def main():
    todo = [r for r in unique_raw if r not in known]
    batches = split_no_dupes(todo, BATCH)
    with tqdm(total=len(todo), unit="drug") as bar:
        async def runner(b):
            await process(b)
            bar.update(len(b))
        await asyncio.gather(*(runner(b) for b in batches))

def run_async():
    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            import nest_asyncio; nest_asyncio.apply()
            return loop.run_until_complete(main())
    except RuntimeError:
        pass
    asyncio.run(main())

if __name__ == "__main__":
    run_async()
    mapping = pd.DataFrame([
        {"medicinal_product": r, "medicinal_product_clean": known.get(r, r)}
        for r in unique_raw
    ])
    ped_new = ped.merge(mapping, on="medicinal_product", how="left")
    ped_new.to_csv(PED_NEW_GZ, compression="gzip", index=False)
    clean_drug = ped_new[["safetyreportid","medicinal_product","medicinal_product_clean"]]
    clean_drug.to_csv(CLEAN_DRUG_CSV, index=False)
    clean_drug.to_excel(CLEAN_DRUG_XLSX, index=False)
    drug_search = mapping.drop_duplicates()
    drug_search.to_csv(DRUG_SEARCH_CSV, index=False)
    drug_search.to_excel(DRUG_SEARCH_XLSX, index=False)
    print("v1: FINAL complete! (raw unique: %d, clean unique: %d)" % (
        len(unique_raw), len(drug_search["medicinal_product_clean"].unique())
    ))
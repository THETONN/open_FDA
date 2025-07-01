import asyncio, json, re, textwrap
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
from aiohttp import ClientError
from openai import AsyncOpenAI, OpenAIError, APIConnectionError, RateLimitError, APITimeoutError, APIStatusError
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

BASE            = Path("../../data/v2/")
BASE.mkdir(parents=True, exist_ok=True)
RAW_GZ          = "../../data/pediatric_patients_report_drug_reaction.csv.gz"
CACHE           = BASE / "_drug_cache_v2.parquet"
CLEAN_DRUG_CSV  = BASE / "clean_drug_v2.csv"
CLEAN_DRUG_XLSX = BASE / "clean_drug_v2.xlsx"
PED_NEW_GZ      = BASE / "ped_data_new_v2.csv.gz"
DRUG_SEARCH_CSV = BASE / "drug_clean_search_v2.csv"
DRUG_SEARCH_XLSX= BASE / "drug_clean_search_v2.xlsx"
DROP_LOG        = BASE / "dropped_v2.txt"
API_KEY         = Path("../../api_key.txt").read_text().strip()
oai             = AsyncOpenAI(api_key=API_KEY)

NOISE = re.compile(
    r"\b(?:tab(?:let)?s?|caps?(?:ule)?s?|inj(?:ection)?|soln|amp|vial|susp|syrup|oral|iv|im|sc)\b"
    r"|[®™]|[\(\)\[\]]|[0-9]+ ?mg\b",
    flags=re.I
)
def strip_noise(txt: str) -> str:
    t = re.sub(r"[.,]+$", "", txt.strip())
    t = NOISE.sub(" ", t)
    return re.sub(r"\s{2,}", " ", t).strip().lower()

SYSTEM_PROMPT = textwrap.dedent("""\
    You receive a JSON array of up to 10 unique drug product names.
    For each, return ONLY the canonical, lowercase INN/USAN name if it is a small molecule drug.
    If it is not a small molecule (e.g. protein, peptide, vaccine, antibody, biological, supplement, vitamin, device, or unknown), return an empty string "".
    The output must be a JSON array in the same order as input.
    Strict JSON only. No commentary.
""")



@retry(
    reraise=True,
    retry=retry_if_exception_type(
        (OpenAIError, ClientError, APIConnectionError,
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
            {"role":"user",  "content":json.dumps([strip_noise(x) for x in batch], ensure_ascii=False)},
        ],
    )
    data = json.loads(rsp.choices[0].message.content)
    if isinstance(data, list):
        arr = data
    elif isinstance(data, dict):
        lst_val = next((v for v in data.values() if isinstance(v, list)), None)
        if lst_val is not None:
            arr = lst_val
        elif all(isinstance(v, str) for v in data.values()):
            try:
                arr = [data[str(i)] for i in range(len(batch))]
            except KeyError:
                arr = list(data.values())
        else:
            raise ValueError("Unsupported JSON schema")
    else:
        raise ValueError("Unexpected JSON type")
    arr = (arr + [""] * len(batch))[: len(batch)]
    if len(arr) != len(batch):
        raise ValueError(f"Row mismatch {len(batch)} vs {len(arr)}")
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

print("loading ped_data …")
ped = pd.read_csv(RAW_GZ, compression="gzip", index_col=0)
unique_raw = ped["medicinal_product"].dropna().unique().tolist()
print(f"   rows={len(ped):,}, unique_raw={len(unique_raw):,}")

if CACHE.exists():
    known = dict(pd.read_parquet(CACHE).values)
    print(f"Loaded cache v2:", len(known))
else:
    known = {}
    print("No cache v2 – fresh start")

BATCH = 10
CONC  = 2
sem = asyncio.Semaphore(CONC)
lock = asyncio.Lock()

async def process(batch, sample=False):
    pre_clean = [strip_noise(x) for x in batch]
    async with sem:
        try:
            cleaned = await gpt_clean(batch)
        except Exception as e:
            print("GPT error (batch) → dropped:", e)
            cleaned = ["" for _ in batch]
    for r, c in zip(batch, cleaned):
        if not c or c.lower() in ("unknown", "."):
            c = ""
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
    # build mapping
    mapping = pd.DataFrame([
        {"medicinal_product": r, "medicinal_product_clean": known.get(r, "")}
        for r in unique_raw
    ])
    ped_new = ped.merge(mapping, on="medicinal_product", how="left")
    ped_new.to_csv(PED_NEW_GZ, compression="gzip", index=False)
    clean_drug = ped_new[["safetyreportid","medicinal_product","medicinal_product_clean"]]
    clean_drug.to_csv(CLEAN_DRUG_CSV, index=False)
    clean_drug.to_excel(CLEAN_DRUG_XLSX, index=False)
    drug_search = mapping[mapping["medicinal_product_clean"].str.strip() != ""].drop_duplicates()
    drug_search.to_csv(DRUG_SEARCH_CSV, index=False)
    drug_search.to_excel(DRUG_SEARCH_XLSX, index=False)
    print("v2: FINAL complete! (raw unique: %d, clean unique: %d)" % (
        len(unique_raw), len(drug_search["medicinal_product_clean"].unique())
    ))
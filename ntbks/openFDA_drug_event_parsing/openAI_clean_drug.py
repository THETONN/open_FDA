import pandas as pd
from openai import OpenAI
import re
import time


ped_data = pd.read_csv('../../data/pediatric_patients_report_drug_reaction.csv.gz', compression='gzip')
unique_drugs = ped_data['medicinal_product'].dropna().unique()

drug_df = pd.DataFrame({'medicinal_product': unique_drugs})
if 'medicinal_product_clean' not in drug_df.columns:
    drug_df['medicinal_product_clean'] = None


with open("../../api_key.txt", "r") as f:
    api_key = f.read().strip()

client = OpenAI(api_key=api_key)

def get_clean_drug_name(drug_name):
    prompt = f"""
แยกชื่อยาหลักจากข้อความต่อไปนี้ เพื่อใช้ค้นหาในฐานข้อมูลยา เช่น RxNorm:
ชื่อยา: {drug_name}
โปรดตอบกลับเฉพาะชื่อยาหลักเท่านั้น เช่น: Acetaminophen
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=50
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error: {e}")
        return None

def extract_drug_name_from_gpt(raw_response: str) -> str:
    if not raw_response:
        return None
    match = re.search(r'"([^"]+)"', raw_response)
    if match:
        return match.group(1).strip()
    match = re.search(r'คือ\s+([A-Za-z\- ]+)', raw_response)
    if match:
        return match.group(1).strip()
    return raw_response.strip()


batch_size = 1000
pause_time = 120  
total = len(drug_df)

for start in range(0, total, batch_size):
    end = min(start + batch_size, total)
    print(f"\n🌀 Processing batch {start} to {end - 1}...\n")

    for i in range(start, end):
        if pd.notna(drug_df.loc[i, 'medicinal_product_clean']):
            continue
        
        raw_name = drug_df.loc[i, 'medicinal_product']
        gpt_response = get_clean_drug_name(raw_name)
        clean_name = extract_drug_name_from_gpt(gpt_response)
        print(f"{i+1}/{total} | Raw: {raw_name} → Clean: {clean_name}")
        drug_df.loc[i, 'medicinal_product_clean'] = clean_name

        if i % 50 == 0:
            
            drug_df.to_csv("../../data/intermediate_cleaned.csv", index=False)

        time.sleep(1.0)  

    # save full batch
    drug_df.to_csv(f"../../data/batch_cleaned_{start}_{end}.csv", index=False)

    print(f"✅ Batch {start}-{end-1} complete. Pausing {pause_time} sec...\n")
    time.sleep(pause_time)

drug_df.to_csv("../../data/drug_cleaned_final.csv", index=False)

merged_df = ped_data.merge(drug_df, on='medicinal_product', how='left')
merged_df.to_csv("../../data/full_dataset_with_clean_drug.csv", index=False)

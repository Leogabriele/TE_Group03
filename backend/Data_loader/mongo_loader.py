from motor.motor_asyncio import AsyncIOMotorClient
from datasets import Dataset
import logging
import pandas as pd
import os
from dotenv import load_dotenv  
load_dotenv()

class MongoDatasetLoader:
    def __init__(self, uri, db_name, collection):
        self.client = AsyncIOMotorClient(uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection]
    
    async def get_training_data(self, verdict_filter="JAILBROKEN", limit=1000):
        logging.info(f"📡 Extracting {verdict_filter} logs...")
        cursor = self.collection.find({"verdict": verdict_filter})
        docs = await cursor.to_list(length=limit)
        
        # Standard Alpaca format for SFTTrainer [web:87]
        formatted = []
        for d in docs:
            full_prompt = f"Goal: {d['forbidden_goal']}\n\n{d['generated_prompt']}"
            
            if verdict_filter == "JAILBROKEN":
                # Fixed refusal for failures
                refusal = self._get_refusal_template(d.get('harm_category', 'generic'))
                formatted.append({
                    "instruction": full_prompt,
                    "output": refusal
                })
            else:  # REFUSED
                formatted.append({
                    "instruction": full_prompt,
                    "output": d['response_text']
                })
        return formatted
    
    def _get_refusal_template(self, category):
        templates = {
            "violence": "I cannot assist with requests promoting violence or harm.",
            "hate": "I won't create content that promotes discrimination.",
            "generic": "I cannot assist with harmful or unsafe requests."
        }
        return templates.get(category, templates['generic'])
def load_benign_csv(csv_path, limit=30000):
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logging.error(f"Error loading benign CSV: {e}")
        return []
    # Assume CSV has 'instruction', 'output' columns
    return [{"instruction": row['Goal'], "output": row['Target']} for _, row in df.head(limit).iterrows()]
async def create_training_dataset():
    loader = MongoDatasetLoader(os.getenv("MONGODB_URI"), "llm_security_auditor", "all_results")
    
    # 10% failures, 30% refusals, 60% benign
    failures = await loader.get_training_data("JAILBROKEN", 2000)
    refusals = await loader.get_training_data("REFUSED", 6000)
    benign = load_benign_csv("data/benign_data.csv", 30000)
    
    full_data = benign + refusals + failures
    dataset = Dataset.from_list(full_data)
    
    # Save as Parquet (fastest for training) [web:86]
    dataset.save_to_disk("data/jailbreak_defense_dataset")
    
    logging.info(f"✅ Dataset ready: {len(full_data)} examples")
    return dataset
if __name__ == "__main__":
    import asyncio

    async def test_loader():
        dataset = await create_training_dataset()
        print(f"Loaded {len(dataset)} training examples.")
    asyncio.run(test_loader())
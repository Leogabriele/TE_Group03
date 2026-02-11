import sys
import os

# Adds the parent directory of 'pipeline' (the root) to the search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import asyncio
from backend.Data_loader.mongo_loader import create_training_dataset
from backend.training.trainer import UnslothTrainer
from backend.export.gguf_converter import GGUFExporter

async def run_retraining(model_name:str):
    # 1. Load Data
    dataset = await create_training_dataset()

    # 2. Train Model
    engine = UnslothTrainer(model_name=f'unsloth/{model_name}')
    engine.apply_lora()
    engine.train(dataset)

    # 3. Export
    GGUFExporter.export(engine.model, engine.tokenizer)

if __name__ == "__main__":
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description="Model Retraining Module")
    parser.add_argument("--model", type=str, required=True, help="Name of the model to retrain")
    
    args = parser.parse_args()
    
    # Execute the retraining function with the passed argument
    asyncio.run(run_retraining(args.model))
    #asyncio.run(run_retraining("Llama-3.2-3B-Instruct-bnb-4bit"))
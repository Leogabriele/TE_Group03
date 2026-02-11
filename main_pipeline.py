import asyncio
from typing import List, Dict, Any
from loguru import logger
import subprocess
import os

# Paths to your virtual environment executables
# Windows example: "envs/training_env/Scripts/python.exe"
STRESS_ENV_PYTHON = os.path.abspath(".venv/Scripts/python.exe")
RETRAIN_PYTHON = os.path.abspath("training_venv/Scripts/python.exe")
class LLM_Audit_And_Retrain():
    def __init__(self,model_name:str):
        self.model_name=model_name
    def run_stress_test(self):
        logger.info("🕵️ Starting Stress Testing...")
        # This runs the stress_testing.py script
        try:
            subprocess.run([STRESS_ENV_PYTHON, "stress_testing.py","--model_test",self.model_name], check=True)
            logger.success("✅ Stress testing complete.")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Stress testing failed: {e}")
    def trigger_retraining(self):
            logger.info(f"🚀 Triggering retraining for {self.model_name}...")
            
            # We call the script and pass the model name as a CLI argument
            try:
                subprocess.run(
                    [RETRAIN_PYTHON, "pipeline/retraining.py", "--model", self.model_name],
                    check=True
                )
                logger.success("✅ Retraining process finished successfully.")
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ Retraining failed: {e}")

if __name__ == "__main__":
    # Example: This is where you'd call it after your stress test
    target_model = "Llama-3.2-3B-Instruct-bnb-4bit"
    audit_and_retrain = LLM_Audit_And_Retrain(model_name=target_model)
    audit_and_retrain.run_stress_test()
    audit_and_retrain.trigger_retraining()
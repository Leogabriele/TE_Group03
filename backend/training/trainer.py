from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
import torch
# gpu check
if torch.device("cuda").type == "cuda":
    print("GPU is available")
class UnslothTrainer:
    def __init__(self, model_name, max_seq=2048):
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq,
            load_in_4bit=True
        )

    def apply_lora(self):
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none"
        )

    def train(self, dataset, output_dir="./outputs"):
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            formatting_func=formatting,
            dataset_text_field="instruction", # Or custom formatting function
            args=TrainingArguments(
                per_device_train_batch_size=2,
                max_steps=60,
                learning_rate=2e-4,
                output_dir=output_dir,
                logging_steps=1
            )
        )
        trainer.train()
import torch
from unsloth import FastLanguageModel
print("✅ Unsloth loaded successfully")

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0)) 
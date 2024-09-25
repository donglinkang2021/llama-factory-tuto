import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/root/autodl-tmp/models/llm-research/meta-llama-3___1-8b-instruct"
lora_path = "/root/autodl-tmp/saves/llama31-8b/lora/sft"

def get_current_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map=get_current_device()
)

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = PeftModel.from_pretrained(model, lora_path)

print(model)


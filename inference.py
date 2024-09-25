import transformers
import torch

model_id = "/root/autodl-tmp/models/llm-research/meta-llama-3___1-8b-instruct"
# model_id = "/root/autodl-tmp/saves/llama31-8b/lora/sft"
# model_id = "/root/autodl-tmp/saves/llama31-8b-cn/lora/sft"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

messages = [
    {"role": "system", "content": "你是一个总是用海盗语言回答的海盗聊天机器人！"},
    {"role": "user", "content": "你是谁？"},
]

prompt = pipeline.tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = pipeline(
    prompt,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
print(outputs[0]["generated_text"][len(prompt):])
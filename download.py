# pip install modelscope
from modelscope import snapshot_download

model_dir = snapshot_download(
    model_id = 'llm-research/meta-llama-3.1-8b-instruct', 
    cache_dir = '/root/autodl-tmp/models',
)
print(model_dir)
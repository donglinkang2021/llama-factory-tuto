# LLaMA Factory Tuto

```shell
git clone --depth 1 git@github.com:hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
conda create -n llama python=3.9 -y
conda activate llama
pip install -e ".[torch,metrics]"
```

```shell
# show the parameters
llamafactory-cli train -h
```

```shell
cd LLaMA-Factory
# sft finetune need just 16G, I use 24G RTX4090 
# dataset: identity,alpaca_en_demo,alpaca_zh_demo
# cost 20min
llamafactory-cli train examples/train_lora/llama3_1_lora_sft.yaml 

# mmlu eval need 60G, I use 80G A800
# cost 18min
llamafactory-cli eval examples/train_lora/llama3_1_lora_eval.yaml 
```

- Average: 66.64
- STEM: 56.99
- Social Sciences: 76.73
- Humanities: 62.44 
- Other: 72.12
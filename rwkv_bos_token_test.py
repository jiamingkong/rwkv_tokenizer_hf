import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rwkv_6_tokenizer.hf_rwkv_tokenizer import Rwkv6Tokenizer


tokenizer = AutoTokenizer.from_pretrained("./rwkv_6_tokenizer", trust_remote_code=True, add_bos_token=True)
tokenizer2 = AutoTokenizer.from_pretrained("./rwkv_6_tokenizer", trust_remote_code=True, add_bos_token=False)
text = """I was so happy to see him that I almost sobbed his name. Eli stiffened and let out a hiss. “Mohiri!” Fear crept into his voice, and my dazed mind wondered what on earth scared a vampire. Nikolas chuckled, and I felt a tremor run through my captor. “I see there is no need for"""
prompt = text

inputs = tokenizer(prompt, return_tensors="pt")
print(inputs)

inputs2 = tokenizer2(prompt, return_tensors="pt")
print(inputs2)
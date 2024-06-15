import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from hf_rwkv_tokenizer import Rwkv6Tokenizer

def generate_prompt(instruction, input=""):
    instruction = instruction.strip().replace('\r\n','\n').replace('\n\n','\n')
    input = input.strip().replace('\r\n','\n').replace('\n\n','\n')
    if input:
        return f"""Instruction: {instruction}

Input: {input}

Response:"""
    else:
        return f"""User: hi

Assistant: Hi. I am your assistant and I will provide expert full response in full details. Please feel free to ask any question and I will always answer it.

User: {instruction}

Assistant:"""


model = AutoModelForCausalLM.from_pretrained("RWKV/rwkv-6-world-1b6", trust_remote_code=True).to(torch.float32)


# tokenizer = AutoTokenizer.from_pretrained("RWKV/rwkv-6-world-7b", trust_remote_code=True, padding_side='left', pad_token="<s>")
# NOTE: Rwkv6Tokenizer is a custom tokenizer class that extends the AutoTokenizer class
tokenizer = Rwkv6Tokenizer("./asset/rwkv_vocab_v20230424.txt")


text = "请介绍北京的旅游景点"
prompt = generate_prompt(text)

inputs = tokenizer(prompt, return_tensors="pt")
output = model.generate(inputs["input_ids"], max_new_tokens=333, do_sample=True, temperature=1.0, top_p=0.3, top_k=0, )
print(tokenizer.decode(output[0].tolist(), skip_special_tokens=True))

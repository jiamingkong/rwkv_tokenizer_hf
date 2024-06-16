import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from hf_rwkv_tokenizer import Rwkv6Tokenizer


def generate_prompt(instruction, input=""):
    instruction = instruction.strip().replace("\r\n", "\n").replace("\n\n", "\n")
    input = input.strip().replace("\r\n", "\n").replace("\n\n", "\n")
    if input:
        return f"""Instruction: {instruction}

Input: {input}

Response:"""
    else:
        return f"""User: hi

Assistant: Hi. I am your assistant and I will provide expert full response in full details. Please feel free to ask any question and I will always answer it.

User: {instruction}

Assistant:"""


model = AutoModelForCausalLM.from_pretrained(
    "RWKV/rwkv-6-world-1b6", trust_remote_code=True
).to(torch.float32)


# NOTE: Rwkv6Tokenizer is a custom tokenizer class that extends the AutoTokenizer class
# You can use one of the two lines below to initialize the tokenizer, they are equivalent
# tokenizer = Rwkv6Tokenizer("rwkv-6-1.6b/rwkv_vocab_v20230424.txt")
tokenizer = AutoTokenizer.from_pretrained("./rwkv-6-tokenizer", trust_remote_code=True)

text = """I was so happy to see him that I almost sobbed his name. Eli stiffened and let out a hiss. “Mohiri!” Fear crept into his voice, and my dazed mind wondered what on earth scared a vampire. Nikolas chuckled, and I felt a tremor run through my captor. “I see there is no need for"""
prompt = text

inputs = tokenizer(prompt, return_tensors="pt")
output = model.generate(
    inputs["input_ids"],
    max_new_tokens=4,
    do_sample=False,
    # temperature=1.0,
    # top_p=0.3,
    # top_k=0,
)
print(tokenizer.decode(output[0].tolist(), skip_special_tokens=True))

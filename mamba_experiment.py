from transformers import MambaConfig, MambaForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
model = MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf")
# input_ids = tokenizer("Hey how are you doing?", return_tensors= "pt")["input_ids"]
# prompt = "Hey how are you doing?"
prompt = "What is the result of 2+2?"
input_ids = tokenizer(prompt, return_tensors= "pt")["input_ids"]
out = model.generate(input_ids, max_new_tokens=10)
print(tokenizer.batch_decode(out))
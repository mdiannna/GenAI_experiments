from transformers import AutoModelForCausalLM




from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "meta-llama/Meta-Llama-Guard-2-8B"
device = "cuda"
dtype = torch.bfloat16

# tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained("SweatyCrayfish/llama-3-8b-quantized")
# model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=device)
model = AutoModelForCausalLM.from_pretrained("SweatyCrayfish/llama-3-8b-quantized", device_map="auto", load_in_4bit=True)

def moderate(chat):
    input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt").to(device)
    output = model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
    prompt_len = input_ids.shape[-1]
    return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)

answer = moderate([
    {"role": "user", "content": "I forgot how to kill a process in Linux, can you help?"},
    {"role": "assistant", "content": "Sure! To kill a process in Linux, you can use the kill command followed by the process ID (PID) of the process you want to terminate."},
])

print("answer:")
print(answer)
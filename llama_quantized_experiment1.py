from transformers import AutoModelForCausalLM

model_4bit = AutoModelForCausalLM.from_pretrained("SweatyCrayfish/llama-3-8b-quantized", device_map="auto", load_in_4bit=True)


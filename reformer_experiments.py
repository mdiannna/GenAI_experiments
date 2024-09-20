# from transformers import AutoTokenizer, ReformerModel
# import torch

# tokenizer = AutoTokenizer.from_pretrained("google/reformer-crime-and-punishment")
# model = ReformerModel.from_pretrained("google/reformer-crime-and-punishment")

# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# outputs = model(**inputs)

# last_hidden_states = outputs.last_hidden_state

# print("outputs:", outputs)
########
# from transformers import AutoTokenizer, ReformerModelWithLMHead
# import torch

# # Load the tokenizer and model with language modeling head
# tokenizer = AutoTokenizer.from_pretrained("google/reformer-crime-and-punishment")
# model = ReformerModelWithLMHead.from_pretrained("google/reformer-crime-and-punishment")

# # Tokenize input
# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

# # Get model outputs (logits)
# outputs = model(**inputs)

# # Generate predictions by taking the argmax of the logits
# predicted_token_ids = torch.argmax(outputs.logits, dim=-1)

# # Decode the token IDs into text
# generated_text = tokenizer.decode(predicted_token_ids[0], skip_special_tokens=True)

# print("Generated text:", generated_text)

# reformer for question answering:
from transformers import AutoTokenizer, ReformerForQuestionAnswering
import torch

tokenizer = AutoTokenizer.from_pretrained("google/reformer-crime-and-punishment")
model = ReformerForQuestionAnswering.from_pretrained("google/reformer-crime-and-punishment")

question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"

inputs = tokenizer(question, text, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

answer_start_index = outputs.start_logits.argmax()
answer_end_index = outputs.end_logits.argmax()

predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]

# target is "nice puppet"
target_start_index = torch.tensor([14])
target_end_index = torch.tensor([15])

outputs = model(**inputs, start_positions=target_start_index, end_positions=target_end_index)
loss = outputs.loss
# Install transformers from source - only needed for versions <= v4.34
# pip install git+https://github.com/huggingface/transformers.git
# pip install accelerate

import torch
from transformers import pipeline

# 1.89 GB
# quantize model"
pipe = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-alpha", torch_dtype=torch.bfloat16, device_map="auto")
# pipe = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-alpha")




user_reviews = [
    "Emily S: I absolutely love this dress! The fit is perfect, and the fabric is so comfortable. The color is true to the picture, and it's versatile enough to wear to both casual and more formal events. I've received so many compliments whenever I wear it. Highly recommend!",
    
    "Alex B: The dress is cute and well-made. The only reason I didn't give it a 5-star rating is that the sizing is a bit tricky. I had to exchange it for a size up, so I recommend checking the size chart carefully before ordering. Other than that, it's a great addition to my wardrobe.",
    
    "Sophie M: The dress is beautiful, but the material is a little thinner than I expected. It's great for warm weather, but I was hoping for something a bit more substantial. The sizing was accurate for me, though, and the design is lovely.",
    
    "Daniel H: My wife looks stunning in this dress! I bought it for her birthday, and she was thrilled with it. The quality is excellent, and it arrived right on time. I'm very happy with this purchase.",
    
    "Lindsay T: The dress is pretty, but the zipper is a bit finicky. It tends to get stuck, and it's frustrating when you're in a hurry. Otherwise, the style and color are lovely, but the zipper issue is a downside for me.",
    
    "Ryan K: This dress is perfect for a summer day out. The fit is comfortable, and the length is just right. The material is breathable, and I appreciate the attention to detail in the stitching. Overall, a good buy!",
    
    "Megan R: I adore this dress! It's elegant and feminine, and I feel confident every time I wear it. The fabric is of high quality, and the dress has held up well after multiple washes. I'm considering buying it in another color."
]

reviews_joined = "\n-".join(user_reviews)


# We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
messages = [
    {
        "role": "system",
        "content": "You are a user reviews summarizer",
    },
    {"role": "user", "content": "Summarize the following reviews:" + reviews_joined},
]
prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=30, top_p=0.95)

print("------------summarized reviews:------------")
print(outputs[0]["generated_text"])
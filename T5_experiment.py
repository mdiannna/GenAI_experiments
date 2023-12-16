# Install the Hugging Face Transformers library
# pip install transformers

from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer

# Load a pre-trained T5 model and tokenizer
# t5-small - 242 MB
model_name = "t5-small"  # You can use different variants like "t5-base" for better performance
# 892 MB
# model_name = "t5-base"  # You can use different variants like "t5-base" for better performance
# flan-t5 was fine-tuned for more tasks:
# 990 MB
# model_name = "google/flan-t5-base"
# 4.69 gb:
# model_name = "google/flan-ul2" # should work for prompting

# 11.4 GB
# Too big for my GPU
# model_name = "t5-3B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

user_reviews = [
    "Emily S: I absolutely love this dress! The fit is perfect, and the fabric is so comfortable. The color is true to the picture, and it's versatile enough to wear to both casual and more formal events. I've received so many compliments whenever I wear it. Highly recommend!",
    
    "Alex B: The dress is cute and well-made. The only reason I didn't give it a 5-star rating is that the sizing is a bit tricky. I had to exchange it for a size up, so I recommend checking the size chart carefully before ordering. Other than that, it's a great addition to my wardrobe.",
    
    "Sophie M: The dress is beautiful, but the material is a little thinner than I expected. It's great for warm weather, but I was hoping for something a bit more substantial. The sizing was accurate for me, though, and the design is lovely.",
    
    "Daniel H: My wife looks stunning in this dress! I bought it for her birthday, and she was thrilled with it. The quality is excellent, and it arrived right on time. I'm very happy with this purchase.",
    
    "Lindsay T: The dress is pretty, but the zipper is a bit finicky. It tends to get stuck, and it's frustrating when you're in a hurry. Otherwise, the style and color are lovely, but the zipper issue is a downside for me.",
    
    "Ryan K: This dress is perfect for a summer day out. The fit is comfortable, and the length is just right. The material is breathable, and I appreciate the attention to detail in the stitching. Overall, a good buy!",
    
    "Megan R: I adore this dress! It's elegant and feminine, and I feel confident every time I wear it. The fabric is of high quality, and the dress has held up well after multiple washes. I'm considering buying it in another color."
]


# Combine user data into a single text
user_text = "\n-".join(user_reviews)

input_text = "Summarize the following user reviews about the dress:"
input_text += user_text
print("input_text: ", input_text)
print(len(input_text))
# input_text = "translate to german: Carbon emissions can be estimated using the Machine Learning Impact calculator "
input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=2048, truncation=True, padding="max_length")

generated_ids = model.generate(input_ids, max_length=300, num_beams=4, no_repeat_ngram_size=2, early_stopping=True)

# Decode and print the generated user profile
summarized_reviews = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print()
print("--- generated text:----")

print()
print(summarized_reviews)

print("-------------------------------")
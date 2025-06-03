This model was trained on the TinyStories dataset - https://arxiv.org/abs/2305.07759

In order to use the model, put the files config.json and pytorch_model.pt in any folder, and then run the following code:

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

model = AutoModelForCausalLM.from_pretrained('path to folder here') # REPLACE BY PATH

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

prompt = "Once upon a time there was"

input_ids = tokenizer.encode(prompt, return_tensors="pt")

# Generate completion
output = model.generate(input_ids, max_length = 1000, num_beams=1, generation_config = generation_config)

# Decode the completion
output_text = tokenizer.decode(output[0], skip_special_tokens=True)

# Print the generated text
print(output_text)
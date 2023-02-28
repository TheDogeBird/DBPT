import os
import openai


# Set up OpenAI API client
openai.api_key = "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"

# Define model and parameters
model = "text-davinci-002"
temperature = 0.5
max_tokens = 100
top_p = 0.5

# Define prompt to use
prompt = "Hello, how are you today?"

# Generate response using OpenAI API
response = openai.Completion.create(
    engine=model,
    prompt=prompt,
    temperature=temperature,
    max_tokens=max_tokens,
    top_p=top_p
)

# Save model to local folder
model_folder = "E:/projects/Icadoa/trainingmodel"
if not os.path.exists(model_folder):
    os.makedirs(model_folder)
model_file = f"{model}-t{temperature}-mt{max_tokens}-tp{top_p}.txt"
with open(os.path.join(model_folder, model_file), "w") as f:
    f.write(response.choices[0].text)

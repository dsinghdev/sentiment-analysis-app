import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import json

# Load environment variables
load_dotenv(override=True)
hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    raise ValueError("Hugging Face API token not found. Please set the HF_TOKEN environment variable.")

client = InferenceClient("Qwen/Qwen2.5-72B-Instruct", token=hf_token)

def load_prompt(file_path):
    """Load prompt from a text file."""
    with open(file_path, 'r') as file:
        return file.read().strip()  

def create_prompt(review):

    base_prompt = load_prompt('sentiment_prompt_v1.txt')
    return base_prompt.format(SENTENCE=review)

def predict_sentiment(review):
    """Predict sentiment using the Hugging Face model."""
    try:
        prompt = create_prompt(review)
        response = client.text_generation(
            prompt,
            model="Qwen/Qwen2.5-72B-Instruct",
            temperature=0,
            max_new_tokens=500,
        )

        print("Model Response:", response)

        try:
            sentiment = json.loads(response)
        except json.JSONDecodeError:
            sentiment = {"Sentiment": response.strip()}

        return sentiment

    except Exception as e:
        print(f"Error during prediction: {e}")
        return {"error": str(e)}

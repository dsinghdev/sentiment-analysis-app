import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import json

# Load environment variables
load_dotenv(override=True)
hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    raise ValueError("Hugging Face API token not found. Please set the HF_TOKEN environment variable.")

client = InferenceClient(api_key=hf_token)

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
        messages = [{"role": "user", "content": prompt}]
        
        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-72B-Instruct",
            messages=messages,
            temperature=0,
            max_tokens=500,
        )

        # Print full response for debugging
        print("Model Response:", response)
        try:
            sentiment_str = response.choices[0].message["content"]
            sentiment = json.loads(sentiment_str)
        except json.JSONDecodeError:
            sentiment = {"Sentiment": sentiment_str.strip()}

        return sentiment

    except Exception as e:
        print(f"Error during prediction: {e}")
        return {"error": str(e)}
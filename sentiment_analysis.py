import os
from dotenv import load_dotenv
from groq import Groq
import json

# Load environment variables
load_dotenv(override=True)
groq_token = os.getenv("GROQ_API_KEY")

if not groq_token:
    raise ValueError("Groq API token not found. Please set the GROQ_API_KEY environment variable.")

# Initialize Groq client
client = Groq(api_key=groq_token)

def load_prompt(file_path):
    """Load prompt from a text file."""
    with open(file_path, 'r') as file:
        return file.read().strip()  

def create_prompt(review):
    base_prompt = load_prompt('sentiment_prompt_v1.txt')
    return base_prompt.format(SENTENCE=review)

def predict_sentiment(review):
    """Predict sentiment using the Groq model."""
    try:
        prompt = create_prompt(review)
        messages = [{"role": "user", "content": prompt}]
        
        response = client.chat.completions.create(
            model="llama3-70b-8192",   # Groq supports LLaMA 3 models, not Qwen
            messages=messages,
            temperature=0,
            max_tokens=500,
        )

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

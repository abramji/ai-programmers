import os
from dotenv import load_dotenv

import openai
import json

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
# Initialize OpenAI client
client = openai.OpenAI()

def analyze_sentiment(review):
    """
    Analyze the sentiment of a movie review using structured output.
    Returns a dictionary with 'thought' and 'sentiment' keys.
    """
    prompt = """
    You are an expert linguistic analyst used to evaluate text sentiment. 
    Analyze the sentiment of the following movie review {review} and provide your analysis in a structured format.
    the Output should be in a json format, removing the spaces before the keys, between the keys and values, and after the values. The format should be compatible with python's json parser. 
    remove the json from the output, and only return the json object. The json should have the following keys:
    thought: [Your analysis of the review]
    sentiment: [positive/negative]
    """
    prompt = prompt.format(review=review)

    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    content = response.choices[0].message.content
    content = json.loads(content)  # Parse the JSON response
    result = {
        "thought": content["thought"],  
        "sentiment": content["sentiment"] 
    }
    
    return result

def main():
    # Test cases
    reviews = [
        "This film shouldn't work at all. It doesn't have much of a story and the whole dial up internet thing is incredibly dated. However Hanks and Ryan sell it beautifully.",
        "The movie was terrible. The acting was wooden, the plot made no sense, and I want my two hours back.",
        "An absolute masterpiece! The cinematography was stunning, the acting was superb, and the story kept me engaged from start to finish."
    ]

    # Test each review
    for i, review in enumerate(reviews, 1):
        print(f"\nAnalyzing Review {i}: {review}")
        result = analyze_sentiment(review)
        print(f"\nReview {i}:")
        print(f"Thought: {result['thought']}")
        print(f"Sentiment: {result['sentiment']}")

if __name__ == "__main__":
    main() 
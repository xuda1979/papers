"""
LLM Client using OpenAI API
"""
import os
from openai import OpenAI

# Initialize the client
# Set your API key via environment variable: OPENAI_API_KEY
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def generate_response(prompt: str, model: str = "gpt-4o", max_tokens: int = 4096) -> str:
    """
    Generate a response using OpenAI API.
    
    Args:
        prompt: The user prompt/question
        model: The model to use (default: gpt-4o)
        max_tokens: Maximum tokens in response
        
    Returns:
        The generated response text
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens
    )
    return response.choices[0].message.content


def generate_with_system(system_prompt: str, user_prompt: str, 
                         model: str = "gpt-4o", max_tokens: int = 4096) -> str:
    """
    Generate a response with a system prompt.
    
    Args:
        system_prompt: The system instructions
        user_prompt: The user prompt/question
        model: The model to use
        max_tokens: Maximum tokens in response
        
    Returns:
        The generated response text
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=max_tokens
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    # Example usage
    result = generate_response("What is the Yang-Mills mass gap problem?")
    print(result)

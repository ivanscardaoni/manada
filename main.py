from openai import OpenAI

# Initialize the client pointing to local Ollama instance
client = OpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1"
)

def call_ollama(prompt: str, model: str = "qwen3:8b", temperature: float = 0.7) -> str:
    """Call Ollama model via OpenAI-compatible API."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=temperature
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    # Test prompt
    test_prompt = "Explica qué es un Large Language Model en una frase"
    result = call_ollama(test_prompt)
    print(f"Prompt: {test_prompt}")
    print(f"Response:\n{result}")

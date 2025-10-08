import openai
import os

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_answer(context, question):
    """Generate an answer using OpenAI GPT-4 based on retrieved context."""
    
    if not openai.api_key:
        return "⚠️ OpenAI API key not found. Please set OPENAI_API_KEY environment variable."
    
    prompt = f"""You are Cognitron, an intelligent enterprise assistant. Answer the question using ONLY the provided context.

Context:
{context}

Question: {question}

Instructions:
- Provide a clear, accurate answer based on the context
- If the context doesn't contain relevant information, say so
- Be concise but comprehensive
"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are Cognitron, a helpful enterprise AI assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400,
            temperature=0.3
        )
        return response['choices'][0]['message']['content']
    
    except Exception as e:
        return f"Error generating answer: {str(e)}"

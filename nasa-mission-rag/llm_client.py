import os
from typing import Dict, List
from openai import OpenAI


def generate_response(openai_key: str, user_message: str, context: str,
                     conversation_history: List[Dict], model: str = "gpt-3.5-turbo") -> str:
    """Generate response using OpenAI with context.

    Args:
        openai_key: OpenAI API key.
        user_message: The user's current question.
        context: Retrieved document context from the RAG system (may be empty).
        conversation_history: Prior turns as a list of {"role", "content"} dicts.
        model: The chat model to use (default: gpt-3.5-turbo).

    Returns:
        The assistant's response text, or an error message on failure.
    """

    # System prompt: NASA mission expert that answers from retrieved context
    # and admits when information is missing.
    system_prompt = """You are a NASA mission intelligence assistant, an expert on \
historic NASA missions including Apollo 11, Apollo 13, and the Challenger (STS-51L) mission.

Your role is to help users understand these missions using retrieved documents such as
mission transcripts, flight plans, and technical reports.

Guidelines:
- Answer questions accurately using the provided context.
- When the context contains the answer, ground your response in it and cite relevant details.
- If the context does not contain enough information, say so clearly instead of guessing.
- Be clear, concise, and factual. Use a professional, informative tone.
- When helpful, explain technical terms so a general audience can follow."""

    # Start with the system prompt, then inject retrieved context (if any)
    # as an additional system message the model can reference.
    messages: List[Dict] = [{"role": "system", "content": system_prompt}]

    if context:
        messages.append({
            "role": "system",
            "content": f"Use the following retrieved context to answer the "
                       f"user's question:\n\n{context}"
        })

    # Include prior conversation turns so the assistant has memory of the dialogue.
    if conversation_history:
        messages.extend(conversation_history)

    # Add the user's current message as the final turn.
    messages.append({"role": "user", "content": user_message})

    # OPENAI_BASE_URL is optional; unset uses the public OpenAI API.
    client = OpenAI(
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=openai_key
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,   # Balanced creativity and consistency
            max_tokens=500     # Reasonable length for informative answers
        )

        return response.choices[0].message.content

    except Exception as e:
        print(f"Error generating response: {e}")
        return ("I apologize, but I'm having trouble generating a response right now. "
                "Please try again in a moment.")

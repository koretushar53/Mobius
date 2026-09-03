# services/llm_service.py
import os
import re
from groq import Groq
import dotenv

dotenv.load_dotenv(override=True)

class DocumentAgent:
    """Groq agent constrained to answering from retrieved document context."""

    def __init__(self):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is missing. Please set the environment variable.")

        self.client = Groq(api_key=api_key)
        self.model = os.environ.get("GROQ_MODEL") or "qwen/qwen3.6-27b"

    def answer(self, question, context_chunks):
        context = "\n\n".join(context_chunks)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are Mobius, a document question-answering agent. "
                        "Answer using only the provided document context. If the "
                        "answer is not present, say that the document does not "
                        "contain enough information. Do not invent facts."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Document context:\n{context}\n\n"
                        f"Question: {question}\n"
                        "Answer clearly and concisely:"
                    ),
                },
            ],
        )
        answer = response.choices[0].message.content or ""
        return re.sub(r"<think>.*?</think>\s*", "", answer, flags=re.DOTALL).strip()


def generate_answer(question, context_chunks):
    return DocumentAgent().answer(question, context_chunks)
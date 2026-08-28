from .config import (
    MODEL_NAME,
    GROQ_API_URL,
    TEMPERATURE,
    REQUEST_TIMEOUT,
    MAX_RETRIES,
    MIN_REQUEST_INTERVAL,
    PLAIN_LLM_API_KEY
)

from .groq_client import (
    GroqRateLimitClient
)


class PlainLLM:

    def __init__(self):

        self.client = (
            GroqRateLimitClient(

                api_key=PLAIN_LLM_API_KEY,

                model=MODEL_NAME,

                api_url=GROQ_API_URL,

                timeout=REQUEST_TIMEOUT,

                max_retries=MAX_RETRIES,

                min_request_interval=(
                    MIN_REQUEST_INTERVAL
                )
            )
        )

    def generate(
        self,
        question
    ):

        system_prompt = """
You are answering questions about academic
and university-related topics.

Answer using your existing knowledge.

Do not claim to have accessed documents,
websites, databases, or university systems
unless such information is explicitly provided
to you.

If you do not know the answer, say so.
Do not invent specific facts, names, email
addresses, URLs, or policies.
""".strip()

        messages = [

            {
                "role": "system",
                "content": system_prompt
            },

            {
                "role": "user",
                "content": question
            }
        ]

        result = self.client.chat(
            messages=messages,
            temperature=TEMPERATURE
        )

        data = result["data"]

        answer = (
            data["choices"][0]
            ["message"]
            ["content"]
            .strip()
        )

        return {

            "answer": answer,

            "latency_seconds":
                result["latency_seconds"],

            "prompt_tokens":
                result["usage"]["prompt_tokens"],

            "completion_tokens":
                result["usage"][
                    "completion_tokens"
                ],

            "total_tokens":
                result["usage"]["total_tokens"],

            "rate_limits":
                result["rate_limits"]
        }
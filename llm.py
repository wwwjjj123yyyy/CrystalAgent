"""
LLM Interface
Author: CrystalAgent

统一封装 DeepSeek API
以后如果更换 GPT/Qwen，只需要修改这里
"""

from openai import OpenAI


class DeepSeekLLM:

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        temperature: float = 0.8,
        max_tokens: int = 4096,
    ):

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt):

        response = self.client.chat.completions.create(

            model=self.model,

            messages=[
                {
                    "role": "system",
                    "content":
                    "You are an expert computational materials scientist."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=self.temperature,

            max_tokens=self.max_tokens,
        )

        return response.choices[0].message.content

from openai import Client


class gptClient:
    def __init__(self):
        self.client = Client()

    def ask_gpt4(self, query, code_snippets):
        try:
            prompt = f"""
            You are an expert coding assistant. The user has asked: "{query}"

            Below are relevant code snippets from the codebase:

            {'\n\n'.join(code_snippets)}

            Provide a concise, accurate answer based on these snippets. Include code examples where applicable.
            """
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a coding expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: Could not get response from GPT-4 ({e})"
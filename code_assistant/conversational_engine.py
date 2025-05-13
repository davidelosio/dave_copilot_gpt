from openai import Client


class gptClient:
    def __init__(self):
        self.client = Client()

    def ask_gpt4(self, query, code_snippets):
        """
        Send a query with relevant code snippets to GPT-4 and retrieve the response.

        Args:
            query (str): The user's question or request about the codebase.
            code_snippets (list): A list of code snippets relevant to the query.

        Returns:
            str: The GPT-4 response, or an error message if the request fails.
        """
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
                temperature=0.5,  # Control randomness of the response
                max_tokens=500   # Limit response length to 500 tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: Could not get response from GPT-4 ({e})"
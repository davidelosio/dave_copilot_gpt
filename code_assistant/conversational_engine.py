from openai import Client


class gptClient():

    def __init__(self):
        self.client = Client()

    def ask_gpt4(self, query, code_snippets):
        """Send query along with retrieved code snippets to GPT-4."""
        # Construct the conversation prompt
        prompt = f"""
            You are an AI assistant helping with a codebase. 
            Here is the user's question: {query}

            Here are relevant code snippets from the project:
            {'\n\n'.join(code_snippets)}

            Please answer the user's question concisely and helpfully.
            """
        # Call the updated ChatCompletion API
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for a coding project."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
        )
        # Extract the assistant's response
        return response.choices[0].message.content

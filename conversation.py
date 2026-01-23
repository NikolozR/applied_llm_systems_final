from typing import Literal, Optional, Any
import time
from constants import openai_client, gemini_client
from google import genai

class CustomConversation:
    def __init__(self, api_provider: Literal['Gemini', 'OpenAI'], model: str):
        self.api_provider = api_provider
        self.model = model
        self.conversation: Any = None
        self._start_conversation()

    def _start_conversation(self):
        if self.api_provider == 'Gemini':
            self.conversation = gemini_client.chats.create(model=self.model)
        else:
            self.conversation = openai_client.conversations.create()

    def send_message(self, message: str, structured_output: Optional[Any] = None) -> Any:
        import random
        retries = 0
        max_retries = 15  # Increased from 5 to 15
        base_delay = 4    # Increased base delay

        while retries < max_retries:
            try:
                if self.api_provider == 'Gemini':
                    if structured_output:
                        response = self.conversation.send_message(
                            message,
                            config=genai.types.GenerateContentConfig(
                                response_mime_type="application/json",
                                response_schema=structured_output
                            )
                        )
                        return structured_output.model_validate_json(response.text)
                    else:
                        response = self.conversation.send_message(message)
                    return response
                else:
                    if structured_output:
                        response = openai_client.responses.parse(
                            model=self.model,
                            input=[{"role": "user", "content": message}],
                            conversation=self.conversation.id,
                            text_format=structured_output
                        )
                        return response.output_parsed
                    else:
                        response = openai_client.responses.create(
                            model=self.model,
                            input=[{"role": "user", "content": message}],
                            conversation=self.conversation.id
                        )
                    return response

            except Exception as e:
                # Check for rate limit errors in message
                error_msg = str(e).lower()
                if "rate limit" in error_msg or "429" in error_msg:
                    # Exponential backoff with jitter
                    wait_time = (base_delay * (2 ** retries)) + random.uniform(0, 1)
                    print(f"Rate limit hit for {self.model}. Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    retries += 1
                else:
                    raise e
        
        raise Exception(f"Max retries exceeded for {self.model} after {max_retries} attempts.")
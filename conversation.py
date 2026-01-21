from typing import Literal, Optional, Any
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
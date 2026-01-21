from utils import *
from message import *
from schemas import *

convo_4o = CustomConversation('OpenAI', 'gpt-4o')
convo_5 = CustomConversation('OpenAI', 'gpt-5.2-2025-12-11')

response_4o = convo_4o.send_message(ROLE_SELECTION_PROMPT, RolePreference)
response_5 = convo_5.send_message(ROLE_SELECTION_PROMPT, RolePreference)



model_confidences = [
    {
        "model": "gpt-4o",
        "confidences": [entry.model_dump() for entry in response_4o.output_parsed.confidence_by_role]
    },
    {
        "model": "gpt-5.2-2025-12-11",
        "confidences": [entry.model_dump() for entry in response_5.output_parsed.confidence_by_role]
    }
] 

print(model_confidences)
print("==============================================================================")


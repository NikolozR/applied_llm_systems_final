from conversation import *
from message import *
from schemas import *
from utils import distribute_roles
from concurrent.futures import ThreadPoolExecutor

models = [
    ('OpenAI', 'gpt-4o'),
    ('OpenAI', 'gpt-5.2-2025-12-11'),
    ('Gemini', 'gemini-3-pro-preview'),
    ('Gemini', 'gemini-2.0-flash-lite-001')
]

def get_model_response(provider, model_name):
    convo = CustomConversation(provider, model_name)
    response = convo.send_message(ROLE_SELECTION_PROMPT, RolePreference)
    return {
        "model": model_name,
        "confidences": [entry.model_dump() for entry in response.confidence_by_role]
    }

# with ThreadPoolExecutor() as executor:
#     futures = [
#         executor.submit(get_model_response, provider, model) 
#         for provider, model in models
#     ]
#     model_confidences = [future.result() for future in futures]

model_confidences_by_hand = [
    {
        'model': 'gpt-4o', 
        'confidences': [
            {'role': 'Solver', 'score': 0.9}, 
            {'role': 'Judge', 'score': 0.8}
        ]
    }, 
    {
        'model': 'gpt-5.2-2025-12-11', 
        'confidences': [
            {'role': 'Solver', 'score': 0.72}, 
            {'role': 'Judge', 'score': 0.66}
        ]
    }, 
    {
        'model': 'gemini-3-pro-preview', 
        'confidences': [
            {'role': 'Solver', 'score': 0.95}, 
            {'role': 'Judge', 'score': 0.9}
        ]
    }, 
    {
        'model': 'gemini-2.0-flash-lite-001', 
        'confidences': [
            {'role': 'Judge', 'score': 0.85}, 
            {'role': 'Solver', 'score': 0.7}
        ]
    }   
]


print(model_confidences_by_hand)

assignments = distribute_roles(model_confidences_by_hand)
print("\nRole Assignments:")
print(assignments)


from conversation import *
from message import *
from schemas import *
from utils import distribute_roles
from concurrent.futures import ThreadPoolExecutor

question = "A small bird native to Australian forests, the wattlebird honeyeater, is on the verge of extinction. The situation is made more difficult by the fact that these birds have trouble forming pairs. To help them, ornithologists capture the males and keep them in cages for a certain period of time. During this time, scientists effectively become \"X\" for the birds.\nQuestion: Name X in two words."

models = [
    ('gpt-4o', CustomConversation('OpenAI', 'gpt-4o')),
    ('gpt-5.2-2025-12-11', CustomConversation('OpenAI', 'gpt-5.2-2025-12-11')),
    ('gemini-3-pro-preview', CustomConversation('Gemini', 'gemini-3-pro-preview')),
    ('gemini-2.0-flash-lite-001', CustomConversation('Gemini', 'gemini-2.0-flash-lite-001'))
]

def get_model_response(model_name, convo):
    response = convo.send_message(ROLE_SELECTION_PROMPT, RolePreference)
    return {
        "model": model_name,
        "confidences": [entry.model_dump() for entry in response.confidence_by_role]
    }

with ThreadPoolExecutor() as executor:
    futures = [
        executor.submit(get_model_response, model, convo) 
        for model, convo in models
    ]
    model_confidences = [future.result() for future in futures]

# model_confidences_by_hand = [
#     {
#         'model': 'gpt-4o', 
#         'confidences': [
#             {'role': 'Solver', 'score': 0.9}, 
#             {'role': 'Judge', 'score': 0.8}
#         ]
#     }, 
#     {
#         'model': 'gpt-5.2-2025-12-11', 
#         'confidences': [
#             {'role': 'Solver', 'score': 0.72}, 
#             {'role': 'Judge', 'score': 0.66}
#         ]
#     }, 
#     {
#         'model': 'gemini-3-pro-preview', 
#         'confidences': [
#             {'role': 'Solver', 'score': 0.95}, 
#             {'role': 'Judge', 'score': 0.9}
#         ]
#     }, 
#     {
#         'model': 'gemini-2.0-flash-lite-001', 
#         'confidences': [
#             {'role': 'Judge', 'score': 0.85}, 
#             {'role': 'Solver', 'score': 0.7}
#         ]
#     }   
# ]

assignments = distribute_roles(model_confidences, models)

# Separate Judge and Solvers
judge_assignment = None
solvers_assignments = []

for model_name, data in assignments.items():
    if data['role'] == 'Judge':
        judge_assignment = {'model': model_name, 'conversation': data['conversation']}
    else:
        solvers_assignments.append({'model': model_name, 'conversation': data['conversation']})

print(f"\nJudge: {judge_assignment['model']}")
print(f"Solvers: {[s['model'] for s in solvers_assignments]}")


def get_solver_answer(model_name, convo):
    response = convo.send_message(get_solver_prompt(question), SolverResponse)
    return {
        "model": model_name,
        "result": response
    }

with ThreadPoolExecutor() as executor:
    futures = [
        executor.submit(
            lambda m, c: {
                "model": m, 
                "response": c.send_message(get_solver_prompt(question), SolverResponse)
            }, 
            solver['model'], 
            solver['conversation']
        ) 
        for solver in solvers_assignments
    ]
    solver_answers = [future.result() for future in futures]

print("\nSolver Answers:")
for ans in solver_answers:
    print(f"Model: {ans['model']}")
    print(f"Response: {ans['response']}")
    print("-" * 20)

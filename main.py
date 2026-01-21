from conversation import *
from message import *
from schemas import *
from utils import *
from concurrent.futures import ThreadPoolExecutor

question = "A small bird native to Australian forests, the wattlebird honeyeater, is on the verge of extinction. The situation is made more difficult by the fact that these birds have trouble forming pairs. To help them, ornithologists capture the males and keep them in cages for a certain period of time. During this time, scientists effectively become \"X\" for the birds.\nQuestion: Name X in two words."

models = [
    ('gpt-4o', CustomConversation('OpenAI', 'gpt-4o')),
    ('gpt-5.2-2025-12-11', CustomConversation('OpenAI', 'gpt-5.2-2025-12-11')),
    ('gemini-3-pro-preview', CustomConversation('Gemini', 'gemini-3-pro-preview')),
    ('gemini-2.0-flash-lite-001', CustomConversation('Gemini', 'gemini-2.0-flash-lite-001'))
]

with ThreadPoolExecutor() as executor:
    futures = [
        executor.submit(get_role_decision, model, convo) 
        for model, convo in models
    ]
    model_confidences = [future.result() for future in futures]

assignments = distribute_roles(model_confidences, models)

judge_assignment = None
solvers_assignments = []

solver_id_counter = 1
for model_name, data in assignments.items():
    if data['role'] == 'Judge':
        judge_assignment = {'model': model_name, 'conversation': data['conversation']}
    else:
        solvers_assignments.append({
            'solver_id': f'solver_{solver_id_counter}',
            'model': model_name, 
            'conversation': data['conversation']
        })
        solver_id_counter += 1

with ThreadPoolExecutor() as executor:
    futures = [
        executor.submit(
            lambda sid, m, c: {
                "solver_id": sid,
                "model": m,
                "response": c.send_message(get_solver_prompt(question), SolverResponse)
            }, 
            solver['solver_id'],
            solver['model'], 
            solver['conversation']
        ) 
        for solver in solvers_assignments
    ]
    solver_answers = [future.result() for future in futures]

solver_conversations = {s['solver_id']: s['conversation'] for s in solvers_assignments}

with ThreadPoolExecutor() as executor:
    futures = [
        executor.submit(
            get_feedback,
            ans['solver_id'], 
            solver_conversations[ans['solver_id']],
            solver_answers
        )
        for ans in solver_answers
    ]
    peer_feedbacks = [future.result() for future in futures]

with ThreadPoolExecutor() as executor:
    futures = [
        executor.submit(
            get_refinement,
            ans['solver_id'], 
            solver_conversations[ans['solver_id']],
            peer_feedbacks
        )
        for ans in solver_answers
    ]
    refined_results = [future.result() for future in futures]

print("\nRefined Solutions:")
for res in refined_results:
    print(f"Solver: {res['solver_id']}")
    rr = res['refined_response']
    print(f"Confidence: {rr.confidence}")
    print(f"Refined Answer: {rr.refined_answer}")
    print(f"Refined Solution: {rr.refined_solution}")
    print("Changes Made:")
    for change in rr.changes_made:
        print(f"  - [{change.accepted}] Critique: {change.critique} -> Response: {change.response}")
    print("-" * 40)



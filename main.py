from conversation import *
from message import *
from schemas import *
from utils import *
from concurrent.futures import ThreadPoolExecutor

question = "In one of Indonesia’s schools, a cat helps the teacher maintain discipline in the lower grades. Question: What will the cat do if the students start making noise?"

models = [
    ('gpt-4o', CustomConversation('OpenAI', 'gpt-4o')),
    ('gpt-5.2-2025-12-11', CustomConversation('OpenAI', 'gpt-5.2-2025-12-11')),
    ('gemini-3-pro-preview', CustomConversation('Gemini', 'gemini-3-pro-preview')),
    ('gemini-2.0-flash-lite-001', CustomConversation('Gemini', 'gemini-2.0-flash-lite-001'))
]

# 1. Role Selection
print("Selecting Roles...")
model_confidences = run_parallel_task(
    lambda item: get_role_decision(item[0], item[1]), 
    models
)

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

print(f"\nJudge: {judge_assignment['model']}")
print(f"Solvers: {[s['solver_id'] + ' (' + s['model'] + ')' for s in solvers_assignments]}")

# 2. Initial Solutions
solver_answers = run_parallel_task(
    lambda s: get_initial_solution(s['solver_id'], s['model'], s['conversation'], question),
    solvers_assignments
)

print("\n--- Initial Answers ---")
for ans in solver_answers:
    print(f"{ans['solver_id']} ({ans['model']}): {ans['response'].answer}")

# 3. Peer Feedback Phase
print("\nGenerating Peer Feedbacks...")

solver_conversations = {s['solver_id']: s['conversation'] for s in solvers_assignments}

peer_feedbacks = run_parallel_task(
    lambda ans: get_feedback(ans['solver_id'], solver_conversations[ans['solver_id']], solver_answers),
    solver_answers
)

print("\n--- Peer Feedback Summaries ---")
for pf in peer_feedbacks:
    for feedback in pf['feedbacks'].feedbacks:
        print(f"{pf['reviewer_id']} -> {feedback.solution_id} | Assessment: {feedback.overall_assessment}")

# 4. Refinement Phase
print("\nRefining Solutions...")

refined_results = run_parallel_task(
    lambda ans: get_refinement(ans['solver_id'], solver_conversations[ans['solver_id']], peer_feedbacks),
    solver_answers
)

print("\n--- Refined Solutions ---")
for res in refined_results:
    rr = res['refined_response']
    print(f"{res['solver_id']} | Confidence: {rr.confidence} | Answer: {rr.refined_answer}")
    print(f"Accepted Changes: {sum(1 for c in rr.changes_made if c.accepted)}/{len(rr.changes_made)}")


judge_convo = judge_assignment['conversation']

# 5. Judge Decision Phase
print("\nJudge is deciding...")
prompt = get_judge_prompt(question, solver_answers, peer_feedbacks, refined_results)
final_verdict = judge_convo.send_message(prompt, FinalDecision)

print("\n" + "="*40)
print("FINAL VERDICT")
print("="*40)
print(f"Winner: {final_verdict.winner}")
print(f"Winning Answer: {final_verdict.winning_answer}")
print(f"Confidence: {final_verdict.confidence}")
print(f"Reasoning: {final_verdict.reasoning}")
print("="*40)



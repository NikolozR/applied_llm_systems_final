from agents import Agent, Solver, Judge
from utils import distribute_roles, run_parallel_task
from typing import List, Tuple, Optional, Any, Dict
from conversation import CustomConversation

def assign_roles(models: List[Tuple[str, CustomConversation]]) -> Tuple[Optional[Judge], List[Solver]]:
    print("Selecting Roles...")
    agents = [Agent(m[0], m[1]) for m in models]
    model_confidences = run_parallel_task(
        lambda a: a.get_role_preferences(), 
        agents
    )

    assignments = distribute_roles(model_confidences, models)

    judge = None
    solvers = []

    solver_id_counter = 1
    for model_name, data in assignments.items():
        if data['role'] == 'Judge':
            judge = Judge(model_name, data['conversation'])
        else:
            solvers.append(Solver(model_name, data['conversation'], f'solver_{solver_id_counter}'))
            solver_id_counter += 1

    if judge:
        print(f"\nJudge: {judge.model_name}")
    else:
        print("\nNo Judge assigned!")

    print(f"Solvers: {[s.solver_id + ' (' + s.model_name + ')' for s in solvers]}")
    
    return judge, solvers

def run_collaborative_solving(question: str, judge: Optional[Judge], solvers: List[Solver]) -> Optional[Any]:
    print(f"\nProcessing Question: {question}")
    
    # 2. Initial Solutions
    solver_answers = run_parallel_task(
        lambda s: s.initial_solve(question),
        solvers
    )

    print("\n--- Initial Answers ---")
    for ans in solver_answers:
        print(f"{ans['solver_id']} ({ans['model']}): {ans['response'].answer}")

    # 3. Peer Feedback Phase
    print("\nGenerating Peer Feedbacks...")

    peer_feedbacks = run_parallel_task(
        lambda s: s.peer_review(solver_answers),
        solvers
    )

    print("\n--- Peer Feedback Summaries ---")
    for pf in peer_feedbacks:
        for feedback in pf['feedbacks'].feedbacks:
            print(f"{pf['reviewer_id']} -> {feedback.solution_id} | Assessment: {feedback.overall_assessment}")

    # 4. Refinement Phase
    print("\nRefining Solutions...")

    refined_results = run_parallel_task(
        lambda s: s.refine_solution(peer_feedbacks),
        solvers
    )

    print("\n--- Refined Solutions ---")
    for res in refined_results:
        rr = res['refined_response']
        print(f"{res['solver_id']} | Confidence: {rr.confidence} | Answer: {rr.refined_answer}")
        print(f"Accepted Changes: {sum(1 for c in rr.changes_made if c.accepted)}/{len(rr.changes_made)}")

    # 5. Judge Decision Phase
    # 5. Judge Decision Phase
    final_verdict = None
    if judge:
        print("\nJudge is deciding...")
        final_verdict = judge.decide(question, solver_answers, peer_feedbacks, refined_results)

        print("\n" + "="*40)
        print("FINAL VERDICT")
        print("="*40)
        print(f"Winner: {final_verdict.winner}")
        print(f"Winning Answer: {final_verdict.winning_answer}")
        print(f"Confidence: {final_verdict.confidence}")
        print(f"Reasoning: {final_verdict.reasoning}")
        print("="*40)
    else:
        print("\nJudge decision skipped due to missing judge.")

    # Prepare detailed output
    return {
        "initial_solutions": [
            {
                "solver_id": ans['solver_id'],
                "model": ans['model'],
                "response": ans['response'].model_dump()
            } for ans in solver_answers
        ],
        "peer_feedbacks": [
            {
                "reviewer_id": pf['reviewer_id'],
                "feedbacks": pf['feedbacks'].model_dump()
            } for pf in peer_feedbacks
        ],
        "refined_solutions": [
            {
                "solver_id": res['solver_id'],
                "refined_response": res['refined_response'].model_dump()
            } for res in refined_results
        ],
        "final_verdict": final_verdict.model_dump() if final_verdict else None
    }

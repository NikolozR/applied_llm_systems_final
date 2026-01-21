from message import *
from schemas import *
from concurrent.futures import ThreadPoolExecutor


def distribute_roles(model_confidences, models):
    best_judge_model = None
    max_judge_score = -1.0

    for model_data in model_confidences:
        judge_score = next(
            (item['score'] for item in model_data['confidences'] if item['role'] == 'Judge'), 
            0.0
        )
        if judge_score > max_judge_score:
            max_judge_score = judge_score
            best_judge_model = model_data

    final_assignments = {}
    
    # Create lookup map for models
    # models structure is assumed to be list of tuples: [(name, conversation), ...]
    model_lookup = {name: convo for name, convo in models}
    
    for model_data in model_confidences:
        model_name = model_data['model']
        role = 'Judge' if model_data == best_judge_model else 'Solver'
        
        final_assignments[model_name] = {
            'role': role,
            'conversation': model_lookup.get(model_name)
        }
            
    return final_assignments

def get_role_decision(model_name, convo):
    response = convo.send_message(ROLE_SELECTION_PROMPT, RolePreference)
    return {
        "model": model_name,
        "confidences": [entry.model_dump() for entry in response.confidence_by_role]
    }

def get_feedback(reviewer_id, reviewer_convo, all_answers):
    others_answers = [ans for ans in all_answers if ans['solver_id'] != reviewer_id]
    prompt = get_feedback_prompt(others_answers)
    feedback_response = reviewer_convo.send_message(prompt, PeerFeedbackList)
    return {
        "reviewer_id": reviewer_id,
        "feedbacks": feedback_response
    }

def get_refinement(solver_id, solver_convo, all_feedbacks):
    # Filter feedbacks for this solver
    # In peer_feedbacks, each entry is from a reviewer. We need to find feedbacks TARGETING this solver_id.
    relevant_feedbacks = []
    for pf in all_feedbacks:
        # pf is {reviewer_id, feedbacks: PeerFeedbackList}
        # feedbacks.feedbacks is list[PeerFeedback]
        # PeerFeedback has solution_id which should match solver_id
        for fb in pf['feedbacks'].feedbacks:
            if fb.solution_id == solver_id:
                relevant_feedbacks.append({
                    'reviewer_id': pf['reviewer_id'],
                    'feedbacks': fb # actually PeerFeedback object
                })
    
    prompt = get_refinement_prompt(relevant_feedbacks)
    refinement_response = solver_convo.send_message(prompt, RefinedSolution)
    
    return {
        "solver_id": solver_id,
        "refined_response": refinement_response
    }

def get_initial_solution(solver_id, model_name, convo, question):
    response = convo.send_message(get_solver_prompt(question), SolverResponse)
    return {
        "solver_id": solver_id,
        "model": model_name,
        "response": response
    }

def run_parallel_task(task_func, items, *args):    
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(task_func, item, *args) for item in items]
        results = [future.result() for future in futures]
    return results
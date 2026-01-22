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


def run_parallel_task(task_func, items, *args):    
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(task_func, item, *args) for item in items]
        results = [future.result() for future in futures]
    return results
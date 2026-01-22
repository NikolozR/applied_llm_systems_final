from message import *
from schemas import *
from concurrent.futures import ThreadPoolExecutor
import json
from conversation import CustomConversation

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
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(task_func, item, *args) for item in items]
        results = [future.result() for future in futures]
    return results

def run_final_evaluation(results_path: str, output_path: str):
    print("\n\n>>> STARTING FINAL EVALUATION...")
    
    with open(results_path, "r") as f:
        results = json.load(f)
        
    # Use a reliable model for evaluation
    evaluator_model = CustomConversation('Gemini', 'gemini-2.5-flash')
    
    evaluation_summary = []
    
    for item in results:
        q_num = item.get('id', 'unknown')
        question = item['question']
        correct = item['correct_answer']
        
        # Extract winning answer from process output
        if item.get('process_output') and item['process_output'].get('final_verdict'):
            winning_ans = item['process_output']['final_verdict']['winning_answer']
        else:
            winning_ans = "NO ANSWER PROVIDED"

        eval_prompt = get_evaluation_prompt(q_num, question, correct, winning_ans)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                eval_result = evaluator_model.send_message(eval_prompt, EvaluationResult)
                evaluation_summary.append(eval_result.model_dump())
                print(f"Question {q_num}: {'CORRECT' if eval_result.is_correct else 'INCORRECT'}")
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Error evaluating question {q_num}: {e}")
                    evaluation_summary.append({
                        "question_number": q_num if isinstance(q_num, int) else 0,
                        "is_correct": False,
                    })
                else:
                    print(f"Retry {attempt+1}/{max_retries} for question {q_num}...")
                    import time
                    time.sleep(2)

    with open(output_path, "w") as f:
        json.dump(evaluation_summary, f, indent=2)
    
    print(f"\nFinal evaluation complete. Saved to {output_path}")
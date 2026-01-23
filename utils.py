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
    import time
    
    with open(results_path, "r") as f:
        results = json.load(f)
        
    # Use a reliable model for evaluation
    evaluator_model = CustomConversation('Gemini', 'gemini-2.5-flash')
    
    evaluation_summary = []
    
    for item in results:
        q_num = item.get('id', 'unknown')
        question = item['question']
        correct = item['correct_answer']
        
        # Helper function for evaluating a single answer
        def evaluate_answer(answer_text, label):
            eval_prompt = get_evaluation_prompt(q_num, question, correct, answer_text)
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    result = evaluator_model.send_message(eval_prompt, EvaluationResult)
                    return result
                except Exception as e:
                    if attempt == max_retries - 1:
                        print(f"Error evaluating {label} for Q{q_num}: {e}")
                        return None
                    else:
                        time.sleep(2)
            return None

        # 1. Evaluate Winning Answer
        if item.get('process_output') and item['process_output'].get('final_verdict'):
            winning_ans = item['process_output']['final_verdict']['winning_answer']
        else:
            winning_ans = "NO ANSWER PROVIDED"

        winner_result = evaluate_answer(winning_ans, "Final Winner")
        
        # Base entry (compatible with existing schema)
        entry = {
            "question_number": q_num if isinstance(q_num, int) else 0,
            "is_correct": winner_result.is_correct if winner_result else False,
            "winning_answer": winning_ans,
            "solver_details": []
        }
        
        print(f"Question {q_num} (Winner): {'CORRECT' if entry['is_correct'] else 'INCORRECT'}")

        # 2. Evaluate Individual Solvers
        process_out = item.get('process_output', {})
        
        # Evaluate Initial Answers
        for sol in process_out.get('initial_solutions', []):
            s_id = sol.get('solver_id', 'unknown')
            ans = sol.get('response', {}).get('answer', '')
            res = evaluate_answer(ans, f"Solver {s_id} Initial")
            is_correct = res.is_correct if res else False
            
            entry['solver_details'].append({
                "solver_id": s_id,
                "type": "initial",
                "answer": ans,
                "is_correct": is_correct
            })
            print(f"  Solver {s_id} (Initial): {'CORRECT' if is_correct else 'INCORRECT'}")

        # Evaluate Refined Answers
        for sol in process_out.get('refined_solutions', []):
            s_id = sol.get('solver_id', 'unknown')
            ans = sol.get('refined_response', {}).get('refined_answer', '')
            res = evaluate_answer(ans, f"Solver {s_id} Refined")
            is_correct = res.is_correct if res else False
            
            entry['solver_details'].append({
                "solver_id": s_id,
                "type": "refined",
                "answer": ans,
                "is_correct": is_correct
            })
            print(f"  Solver {s_id} (Refined): {'CORRECT' if is_correct else 'INCORRECT'}")

        evaluation_summary.append(entry)

    with open(output_path, "w") as f:
        json.dump(evaluation_summary, f, indent=2)
    
    print(f"\nFinal evaluation complete. Saved to {output_path}")
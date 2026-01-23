import json
from collections import Counter


def analyze_performance(results_path="results.json", evaluation_path="final_evaluation.json"):
    with open(results_path, "r") as f:
        results = json.load(f)
    
    with open(evaluation_path, "r") as f:
        evaluations = json.load(f)
        
    eval_map = {item['question_number']: item for item in evaluations}
    
    total_questions = len(results) or 1
    system_correct_count = 0
    
    solvers_improved_count = 0 # Initial=False -> Refined=True
    solvers_regressed_count = 0 # Initial=True -> Refined=False
    total_solver_pairs = 0
    
    for item in results:
        q_id = item.get('id', 0)
        eval_entry = eval_map.get(q_id)
        
        if not eval_entry:
            continue
            
        # 1. System Accuracy (Judge's Final Verdict)
        if eval_entry.get('is_correct', False):
            system_correct_count += 1
            
        # 2. Analyze Solver Details (Initial vs Refined)
        solver_details = eval_entry.get('solver_details', [])
        
        initial_sols = {s['solver_id']: s for s in solver_details if s['type'] == 'initial'}
        refined_sols = {s['solver_id']: s for s in solver_details if s['type'] == 'refined'}
        
        # Initial Stats
        for s_id, s_data in initial_sols.items():
            total_initial_solvers += 1
            if s_data['is_correct']:
                total_initial_correct += 1
            
            # Check Improvement for this solver
            if s_id in refined_sols:
                total_solver_pairs += 1
                refined_data = refined_sols[s_id]
                
                initial_correct = s_data['is_correct']
                refined_correct = refined_data['is_correct']
                
                if not initial_correct and refined_correct:
                    solvers_improved_count += 1
                elif initial_correct and not refined_correct:
                    solvers_regressed_count += 1
                
        # Refined Stats
        for s_data in refined_sols.values():
            total_refined_solvers += 1
            if s_data['is_correct']:
                total_refined_correct += 1
                
        # 3. Voting Baseline (Majority of Initial Answers)
        if initial_sols:
            initial_answer_texts = [s['answer'].strip().lower() for s in initial_sols.values()]
            # Find most common answer
            vote_counts = Counter(initial_answer_texts)
            if not vote_counts:
                majority_ans = ""
            else:
                majority_ans = vote_counts.most_common(1)[0][0]
            
            # Check if majority answer is correct
            is_majority_correct = False
            for s in initial_sols.values():
                if s['answer'].strip().lower() == majority_ans and s['is_correct']:
                    is_majority_correct = True
                    break
            
            if is_majority_correct:
                voting_correct += 1
                
            # 4. Consensus (Based on Initial Answers)
            if len(set(initial_answer_texts)) == 1:
                consensus_count += 1
            else:
                disagreement_cases += 1
                # If disagreement existed (in initial phase), but the Judge (System) got it right
                if eval_entry.get('is_correct', False):
                    judge_correct_disagreement += 1

    # Calculate Percentages
    system_accuracy = (system_correct_count / total_questions) * 100
    avg_initial_accuracy = (total_initial_correct / total_initial_solvers * 100) if total_initial_solvers else 0
    avg_refined_accuracy = (total_refined_correct / total_refined_solvers * 100) if total_refined_solvers else 0
    voting_accuracy = (voting_correct / total_questions) * 100
    consensus_rate = (consensus_count / total_questions) * 100
    judge_efficacy = (judge_correct_disagreement / disagreement_cases * 100) if disagreement_cases else 0
    
    # Improvement Rate: % of solvers that improved (Incorrect -> Correct)
    improvement_rate = (solvers_improved_count / total_solver_pairs * 100) if total_solver_pairs else 0

    print("\n" + "="*40)
    print("PHASE 3: EVALUATION AND ANALYSIS")
    print("="*40)
    
    print("\n[System-Level Performance]")
    print(f"Overall Accuracy:   {system_correct_count}/{total_questions} ({system_accuracy:.1f}%)")
    print(f"Improvement Rate:   {improvement_rate:.1f}% (Solvers Incorrect->Correct)")
    print(f"Consensus Rate:     {consensus_count}/{total_questions} ({consensus_rate:.1f}%)")
    print(f"Judge Efficacy:     {judge_correct_disagreement}/{disagreement_cases} ({judge_efficacy:.1f}%) [in disagreement]")
    
    print("\n[Comparison to Baselines]")
    print(f"System (Debate):    {system_accuracy:.1f}%")
    print(f"Simple Voting:      {voting_accuracy:.1f}%")
    print(f"Avg Initial (Solo): {avg_initial_accuracy:.1f}%")
    print(f"Avg Refined (Solo): {avg_refined_accuracy:.1f}%")
    
    visualize_metrics({
        "System": system_accuracy,
        "Voting": voting_accuracy,
        "Initial": avg_initial_accuracy,
        "Refined": avg_refined_accuracy,
        "Improvement": improvement_rate
    })
    
    # Save chart
    plot_metrics({
        "System": system_accuracy,
        "Voting": voting_accuracy,
        "Initial": avg_initial_accuracy,
        "Refined": avg_refined_accuracy,
        "Improvement": improvement_rate
    })

def plot_metrics(metrics):
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        
        colors = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f']
        
        bars = plt.bar(metrics.keys(), metrics.values(), color=colors[:len(metrics)])
        
        plt.title('Evaluation Metrics', fontsize=16)
        plt.ylabel('Percentage (%)', fontsize=12)
        plt.ylim(0, 100)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=10)
            
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('evaluation_metrics.png', dpi=300)
        print("Chart saved to evaluation_metrics.png")
        
    except ImportError:
        print("Matplotlib not installed. Skipping image generation.")
    except Exception as e:
        print(f"Error generating chart: {e}")

def visualize_metrics(metrics):
    print("\n" + "="*40)
    print("VISUALIZATION (ASCII)")
    print("="*40)
    
    max_label_len = max(len(k) for k in metrics.keys())
    bar_width = 30
    
    for label, score in metrics.items():
        filled_len = int(score / 100 * bar_width)
        bar = "█" * filled_len + "░" * (bar_width - filled_len)
        print(f"{label.ljust(max_label_len)} | {bar} | {score:.1f}%")
    print("="*40)

if __name__ == "__main__":
    analyze_performance()

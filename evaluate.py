import json
from utils import run_final_evaluation
from collections import Counter


def analyze_performance(results_path="results.json", evaluation_path="final_evaluation.json"):
    with open(results_path, "r") as f:
        results = json.load(f)
    
    with open(evaluation_path, "r") as f:
        evaluations = json.load(f)
        
    # Map evaluations by question number
    eval_map = {item['question_number']: item for item in evaluations}
    
    total_questions = len(results) or 1
    correct_count = sum(1 for e in evaluations if e['is_correct'])
    
    consensus_count = 0
    total_changes_accepted = 0
    total_solvers = 0
    
    judge_correct_disagreement = 0
    disagreement_cases = 0
    
    baseline_correct = 0 # Proxy: Solver 1's initial answer
    voting_correct = 0 # Simple majority of refined answers
    
    for item in results:
        q_id = item.get('id', 0)
        is_final_correct = eval_map.get(q_id, {}).get('is_correct', False)
        
        process_output = item['process_output']
        if not process_output: continue
        
        refined_sols = process_output.get('refined_solutions', [])
        initial_sols = process_output.get('initial_solutions', [])
        
        # 1. Improvement Rate Stats
        for sol in refined_sols:
            changes = sol['refined_response'].get('changes_made', [])
            try:
                accepted = sum(1 for c in changes if c.get('accepted', False))
            except AttributeError:
                # Handle case where changes might be strings or malformed
                accepted = 0
            total_changes_accepted += accepted
            total_solvers += 1
            
        # 2. Consensus Rate
        final_answers = [s['refined_response']['refined_answer'].strip().lower() for s in refined_sols]
        if len(set(final_answers)) == 1:
            consensus_count += 1
        else:
            disagreement_cases += 1
            # If there was disagreement, and the final verdict (Judge) was correct
            if is_final_correct:
                judge_correct_disagreement += 1
                
        # 3. Baselines
        voting_answer = Counter(final_answers).most_common(1)[0][0]
        winning_answer = process_output.get('final_verdict', {}).get('winning_answer', "").strip().lower()
        
        # Helper to check correctness against ground truth (string or list)
        def check_match(ans, correct):
            if isinstance(correct, list):
                return any(ans == c.strip().lower() for c in correct)
            return ans == correct.strip().lower()

        if voting_answer == winning_answer:
            if is_final_correct: voting_correct += 1
        else:
            if check_match(voting_answer, item['correct_answer']):
                voting_correct += 1
                
        # Single LLM Baseline: Solver 1 Initial
        if initial_sols:
            s1_ans = initial_sols[0]['response']['answer'].strip().lower()
            if check_match(s1_ans, item['correct_answer']):
                baseline_correct += 1
                
    
    avg_changes = total_changes_accepted / total_solvers if total_solvers else 0
    consensus_rate = consensus_count / total_questions * 100
    judge_efficacy = (judge_correct_disagreement / disagreement_cases * 100) if disagreement_cases else 0
    system_accuracy = correct_count / total_questions * 100
    voting_accuracy = voting_correct / total_questions * 100
    baseline_accuracy = baseline_correct / total_questions * 100

    print("\n" + "="*40)
    print("PHASE 3: EVALUATION AND ANALYSIS")
    print("="*40)
    
    print("\n[System-Level Performance]")
    print(f"Overall Accuracy:   {correct_count}/{total_questions} ({system_accuracy:.1f}%)")
    print(f"Improvement Rate:   {avg_changes:.2f} changes/solver")
    print(f"Consensus Rate:     {consensus_count}/{total_questions} ({consensus_rate:.1f}%)")
    print(f"Judge Efficacy:     {judge_correct_disagreement}/{disagreement_cases} ({judge_efficacy:.1f}%) [in disagreement]")
    
    print("\n[Comparison to Baselines (Approximate)]")
    print(f"System (Debate):    {system_accuracy:.1f}%")
    print(f"Simple Voting:      {voting_accuracy:.1f}%")
    print(f"Single LLM:         {baseline_accuracy:.1f}%")
    
    visualize_metrics({
        "System": system_accuracy,
        "Voting": voting_accuracy,
        "Single": baseline_accuracy,
        "Consensus": consensus_rate,
        "Judge Eff.": judge_efficacy
    })
    
    # Save chart
    plot_metrics({
        "System": system_accuracy,
        "Voting": voting_accuracy,
        "Single": baseline_accuracy,
        "Consensus": consensus_rate,
        "Judge Efficacy": judge_efficacy
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

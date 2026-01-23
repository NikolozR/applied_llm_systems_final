import json
from conversation import CustomConversation
from collaboration import run_collaborative_solving, assign_roles
from utils import run_final_evaluation

if __name__ == "__main__":
    with open("questions.json", "r") as f:
        questions_data = json.load(f)

    models = [
        ('gpt-4o', CustomConversation('OpenAI', 'gpt-4o')),
        ('gpt-5-mini-2025-08-07', CustomConversation('OpenAI', 'gpt-5-mini-2025-08-07')),
        ('gemini-2.5-flash', CustomConversation('Gemini', 'gemini-2.5-flash')),
        ('gemini-2.0-flash-lite-001', CustomConversation('Gemini', 'gemini-2.0-flash-lite-001'))
    ]

    judge, solvers = assign_roles(models)

    results = []
    processed_ids = set()
    try:
        with open("results.json", "r") as f:
            results = json.load(f)
            processed_ids = {item['id'] for item in results}
            print(f"Resuming: Found {len(results)} completed questions.")
    except (FileNotFoundError, json.JSONDecodeError):
        print("Starting fresh (no existing results).")
    
    # Process Questions
    for i, item in enumerate(questions_data, 1):
        if i in processed_ids:
            print(f"Skipping Question {i} (already processed)")
            continue

        question_text = item['question']
        correct_answer = item['answer']
        
        print(f"\n\n>>> STARTING NEW QUESTION: {question_text[:50]}...")
        process_output = run_collaborative_solving(question_text, judge, solvers)
        
        results.append({
            "id": i,
            "question": question_text,
            "correct_answer": correct_answer,
            "process_output": process_output
        })
        
        # Save incrementally
        with open("results.json", "w") as f:
            json.dump(results, f, indent=2)
            
    print("\nAll processing complete. Results saved to results.json")

    # Final Evaluation Phase
    run_final_evaluation("results.json", "final_evaluation.json")

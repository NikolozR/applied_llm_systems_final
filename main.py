import json
from conversation import CustomConversation
from collaboration import run_collaborative_solving, assign_roles

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
    for item in questions_data:
        question_text = item['question']
        correct_answer = item['answer']
        
        print(f"\n\n>>> STARTING NEW QUESTION: {question_text[:50]}...")
        verdict = run_collaborative_solving(question_text, judge, solvers)
        
        results.append({
            "question": question_text,
            "correct_answer": correct_answer,
            "verdict": verdict.model_dump() if verdict else None
        })
        
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nAll processing complete. Results saved to results.json")

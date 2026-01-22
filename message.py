"""
Prompt message for role selection in a multi-agent system.
This prompt asks the model to choose between Solver and Judge roles
for the intellectual game "What? Where? When?"
"""

ROLE_SELECTION_PROMPT = """You are participating in a multi-agent system designed to solve questions from the intellectual game "What? Where? When?" - a game that requires knowledge, logical thinking, and the ability to connect logical dots to arrive at creative answers.

There are two roles available:

1. **Solver**: This role is responsible for analyzing the question, applying logical reasoning, connecting clues, and proposing an answer. Solvers need strong analytical skills, broad knowledge, creative thinking, and the ability to make non-obvious connections.

2. **Judge**: This role is responsible for evaluating proposed answers by different models, checking their validity against the question requirements, and determining if the solution is correct. As a Judge, you will receive 3 different thinking processes and answers from 3 different models, and you must assess each one to determine which is the best. Judges need critical thinking, attention to detail, and the ability to objectively assess reasoning quality.

Based on your capabilities and strengths, which role would you prefer to take? Consider:
- Your ability to think creatively and make unexpected connections
- Your knowledge base across various domains
- Your logical reasoning and analytical skills
- Your ability to evaluate and critique solutions objectively

Please respond with your role preferences in order, confidence scores for each role variant (0-1 scale), and detailed reasoning explaining why you chose these preferences based on the nature of "What? Where? When?" questions."""

def get_solver_prompt(question):
    return f"""{question}

You have been selected as a **Solver** for the intellectual game 'What? Where? When?'. Your task is to solve the following question using your broad knowledge, logical reasoning, and ability to make non-obvious connections.

**Your Goal:**
1. **Analyze the Question:** Break down the text, identify key keywords, metaphors, and hidden clues. Pay attention to specific phrasing.
2. **Generate Hypotheses:** Brainstorm potential answers. Connect disparate facts.
3. **Refine & Select:** Perform a logical consistency check. Discard weak theories. Select the most precise and elegant answer that fits all the facts.
4. **Explain Your Solution:** Provide a clear, step-by-step explanation of your reasoning path. Why is this answer the only correct one? What specific steps led you to it?

**Constraint Checklist & Confidence Score:**
1. Check constraints! "What? Where? When?" questions often ask for a specific number of words or a short phrase.
2. Ensure your final answer is **concise**. NO preamble. Just the answer.
3. Provide a confidence score (0.0 - 1.0).
"""


def get_feedback_prompt(others_solutions):
    solutions_text = "\n\n".join([f"Solution ID: {sol['solver_id']}\nAnswer: {sol['response'].answer}\nExplanation: {sol['response'].explanation}" for sol in others_solutions])
    
    return f"""You are now acting as a peer reviewer. Review the following solutions provided by other solvers for the question you just solved.
    
{solutions_text}

Provide detailed feedback for EACH solution using the structured format. 
- Identify logical errors, weak arguments, or missing steps.
- Highlight strengths.
- Offer constructive suggestions for improvement.
- Be critical but fair."""

def get_refinement_prompt(feedbacks):
    feedback_text = "\n\n".join([f"Feedback from Solver {fb['reviewer_id']}:\nAssessment: {fb['feedbacks'].overall_assessment}\nCritique: {fb['feedbacks'].evaluation.errors}\nSuggestions: {fb['feedbacks'].evaluation.suggested_changes}" for fb in feedbacks])
    
    return f"""You have received feedback from other solvers on your initial solution:

{feedback_text}

Analyze this feedback carefully.
1. Evaluate each critique point: Is it valid? Did you miss something?
2. If the critique is valid, refine your solution.
3. If you disagree, explain why.
4. Provide a final, refined solution and answer.

Respond using the structured format."""

def get_judge_prompt(question, original_answers, peer_feedbacks, refined_solutions):
    # Format inputs for the prompt
    
    # Original Answers
    orig_text = "\n".join([f"Solver {ans['solver_id']} ({ans['model']}): {ans['response'].answer}\nExplanation: {ans['response'].explanation}\n" for ans in original_answers])
    
    # Peer Feedbacks
    feedback_text = ""
    for pf in peer_feedbacks:
        feedback_text += f"\nReviewer {pf['reviewer_id']} feedback:\n"
        for fb in pf['feedbacks'].feedbacks:
            feedback_text += f"- To {fb.solution_id}: Assessment: {fb.overall_assessment}, Critique: {fb.evaluation.errors}\n"
            
    # Refined Answers
    refined_text = "\n".join([f"Solver {res['solver_id']}:\nRefined Answer: {res['refined_response'].refined_answer}\nExplanation: {res['refined_response'].refined_solution}\nConfidence: {res['refined_response'].confidence}\nChanges: {[c.response for c in res['refined_response'].changes_made]}\n" for res in refined_solutions])

    return f"""You are the Judge in this intellectual competition.
    
**The Question:**
{question}

---
**Phase 1: Original Solutions**
{orig_text}

---
**Phase 2: Peer Feedback**
Solvers reviewed each other's work:
{feedback_text}

---
**Phase 3: Refined Solutions**
After considering feedback, solvers improved their answers:
{refined_text}

---
**Your Task:**
1. Evaluate the final refined solutions.
2. Consider how well they addressed valid critiques.
3. Select the single best solution.
4. Explain your reasoning and provide a confidence score.
5. Provide the exact text of the winning answer.

Choose the WINNER."""

def get_evaluation_prompt(q_num, question, correct, winning_ans):
    return f"""You are an objective evaluator for a QA system.
        
Question ({q_num}): {question}
Correct Answer: {correct}

System's Answer: {winning_ans}

Task: Determine if the System's Answer is semantically correct based on the Correct Answer. 
Ignore minor phrasing differences, capitalization, or punctuation. Focus on the core meaning.
Return your decision as a JSON object with:
- question_number by integer: {q_num}
- is_correct: true/false
"""

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

**Constraint:**
- Ensure your explanation explicitly details the logical steps taken to arrive at the final answer."""

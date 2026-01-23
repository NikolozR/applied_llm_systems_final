# Multi-LLM Collaborative Debate System

This project implements a multi-agent debate system where three LLMs collaboratively solve challenging problems ("What? Where? When?" questions). The system uses a structured workflow involving independent solution generation, peer review, refinement, and a final judgment phase.

## System Architecture

The system consists of four main roles filled by different LLMs (e.g., GPT-4o, GPT-5-mini, Gemini Pro/Flash):

1.  **Solvers (3 Agents)**: 
    *   Generate independent initial solutions.
    *   Review peers' solutions to identify errors and suggest improvements.
    *   Refine their own solutions based on received feedback.
2.  **Judge (1 Agent)**:
    *   Evaluates the final refined solutions.
    *   Selects the best answer based on logic, adherence to constraints, and improved reasoning.

## Dynamic Role Assignment
For each question, the agents self-assess their suitability for the roles of "Solver" or "Judge" based on the specific content of the question. The system then deterministically assigns the most confident agent to be the Judge, while the others become Solvers.

## Requirements

*   Python 3.10+
*   API Keys for OpenAI and Google Gemini stored in a `.env` file.

### Setup

1.  Clone the repository.
2.  Install dependencies (assuming a `requirements.txt` or manual install):
    ```bash
    pip install openai google-genai python-dotenv matplotlib
    ```
3.  Create a `.env` file in the root directory:
    ```
    OPENAI_API_KEY=your_openai_key_here
    GEMINI_API_KEY=your_gemini_key_here
    ```

## Usage

### Running the Debate System

To start the solving process:

```bash
python main.py
```

**IMPORTANT:** The system is designed to be resumable. It checks the `results.json` file for questions that have already been processed and skips them. 

**If you want to re-run the entire dataset or specific questions, you must clear or delete the `results.json` file before running the script.**

### Evaluation

After the debate process is complete, the system automatically runs an evaluation phase. You can also run the evaluation manually:

```bash
python evaluate.py
```

This will:
1.  Analyze the performance of the system.
2.  Compare the System's final verdict against individual solver performance.
3.  Generate an `evaluation_metrics.png` chart visualizing the results.
4.  Save detailed evaluation data to `final_evaluation.json`.

## Metrics Explanation

The system monitors several key performance indicators to validate the effectiveness of the collaborative debate:

*   **System Accuracy**: The percentage of questions where the Final Judge selected the correct answer. This is the primary measure of the system's overall success.
*   **Simple Voting Accuracy**: A baseline metric that selects the most common answer from the initial independent solutions. This helps determine if the complex debate process yields better results than a simple majority vote.
*   **Avg Initial Accuracy**: The average correctness of the Solvers' first attempts, before any peer review or refinement.
*   **Avg Refined Accuracy**: The average correctness of the Solvers' answers *after* the peer review and refinement phase. A higher refined accuracy compared to initial accuracy indicates that the debate process helped agents improve their reasoning.
*   **Improvement Rate**: The percentage of instances where a Solver initially had an incorrect answer but corrected it after receiving peer feedback. This directly measures the value of the "Debate" mechanism.
*   **Consensus Rate**: The frequency with which all Solvers agreed on the same answer initially.
*   **Judge Efficacy**: Specifically measures the Judge's ability to pick the correct answer in cases where the Solvers disagreed. High efficacy suggests the Judge is correctly discerning superior reasoning.

## Files

*   `main.py`: Main entry point. Handles role assignment, the debate loop, and saving results.
*   `questions.json`: Dataset of 25 challenging problems.
*   `agents.py` & `collaboration.py`: definitions of agent behaviors and the interaction workflow.
*   `evaluate.py`: Script for calculating accuracy metrics and generating plots.
*   `utils.py`: Helper functions for role distribution, concurrent execution, and evaluation logic.
*   `results.json`: Output file containing the full trace of the debate for each question.

## Team Members

*   Nikoloz Rusishvili
*   Mariam Vanadze
*   Demetre Kanachadze
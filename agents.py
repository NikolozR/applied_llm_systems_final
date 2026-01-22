from typing import List, Dict, Any
from conversation import CustomConversation
from message import (
    get_solver_prompt,
    get_feedback_prompt,
    get_refinement_prompt,
    get_judge_prompt,
    ROLE_SELECTION_PROMPT
)
from schemas import (
    RolePreference,
    SolverResponse,
    PeerFeedbackList,
    RefinedSolution,
    FinalDecision
)

class Agent:
    def __init__(self, model_name: str, conversation: CustomConversation):
        self.model_name = model_name
        self.conversation = conversation

    def get_role_preferences(self) -> Dict[str, Any]:
        response = self.conversation.send_message(ROLE_SELECTION_PROMPT, RolePreference)
        return {
            "model": self.model_name,
            "confidences": [entry.model_dump() for entry in response.confidence_by_role]
        }

class Solver(Agent):
    def __init__(self, model_name: str, conversation: CustomConversation, solver_id: str):
        super().__init__(model_name, conversation)
        self.solver_id = solver_id

    def initial_solve(self, question: str) -> Dict[str, Any]:
        response = self.conversation.send_message(get_solver_prompt(question), SolverResponse)
        return {
            "solver_id": self.solver_id,
            "model": self.model_name,
            "response": response
        }

    def peer_review(self, all_answers: List[Dict[str, Any]]) -> Dict[str, Any]:
        other_answers = [ans for ans in all_answers if ans['solver_id'] != self.solver_id]
        prompt = get_feedback_prompt(other_answers)
        feedback_response = self.conversation.send_message(prompt, PeerFeedbackList)
        return {
            "reviewer_id": self.solver_id,
            "feedbacks": feedback_response
        }

    def refine_solution(self, all_feedbacks: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Filter feedbacks for this solver
        relevant_feedbacks = []
        for pf in all_feedbacks:
            for fb in pf['feedbacks'].feedbacks:
                if fb.solution_id == self.solver_id:
                    relevant_feedbacks.append({
                        'reviewer_id': pf['reviewer_id'],
                        'feedbacks': fb
                    })
        
        prompt = get_refinement_prompt(relevant_feedbacks)
        refinement_response = self.conversation.send_message(prompt, RefinedSolution)
        
        return {
            "solver_id": self.solver_id,
            "refined_response": refinement_response
        }

class Judge(Agent):
    def decide(self, question: str, original_answers: List[Dict[str, Any]], 
               peer_feedbacks: List[Dict[str, Any]], refined_solutions: List[Dict[str, Any]]) -> FinalDecision:
        prompt = get_judge_prompt(question, original_answers, peer_feedbacks, refined_solutions)
        final_verdict = self.conversation.send_message(prompt, FinalDecision)
        return final_verdict

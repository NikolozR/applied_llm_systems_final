from pydantic import BaseModel, Field
from typing import List

class RoleConfidenceEntry(BaseModel):
    role: str
    score: float

class RolePreference(BaseModel):
    role_preferences: List[str]
    confidence_by_role: List[RoleConfidenceEntry]
    reasoning: str

class SolverResponse(BaseModel):
    answer: str
    explanation: str

class ErrorDetail(BaseModel):
    location: str
    error_type: str
    description: str
    severity: str

class FeedbackEvaluation(BaseModel):
    strengths: List[str]
    weaknesses: List[str]
    errors: List[ErrorDetail]
    suggested_changes: List[str]

class PeerFeedback(BaseModel):
    solution_id: str
    evaluation: FeedbackEvaluation
    overall_assessment: str

class PeerFeedbackList(BaseModel):
    feedbacks: List[PeerFeedback]

class ChangeResponse(BaseModel):
    critique: str
    response: str
    accepted: bool

class RefinedSolution(BaseModel):
    changes_made: List[ChangeResponse] = Field(description="List of critiques addressed and changes made")
    refined_solution: str = Field(description="The full detailed explanation and reasoning process leading to the answer")
    refined_answer: str = Field(description="The short, concise final answer requested by the user, e.g., a few words or a number")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")

class FinalDecision(BaseModel):
    winner: str
    winning_answer: str
    confidence: float
    reasoning: str

class EvaluationResult(BaseModel):
    question_number: int
    is_correct: bool
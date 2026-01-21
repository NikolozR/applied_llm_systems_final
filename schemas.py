from pydantic import BaseModel
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
    changes_made: List[ChangeResponse]
    refined_solution: str
    refined_answer: str
    confidence: float
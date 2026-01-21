from pydantic import BaseModel
from typing import List

class RoleConfidenceEntry(BaseModel):
    role: str
    score: float

class RolePreference(BaseModel):
    role_preferences: List[str]
    confidence_by_role: List[RoleConfidenceEntry]
    reasoning: str
"""
Code Review Environment - Type Models
All data structures are Pydantic models for OpenEnv compliance
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime


# ============ Enums ============

class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class BugType(str, Enum):
    SECURITY = "security"
    LOGIC = "logic"
    PERFORMANCE = "performance"
    STYLE = "style"
    DOCUMENTATION = "documentation"
    BEST_PRACTICE = "best_practice"
    RACE_CONDITION = "race_condition"
    MEMORY_LEAK = "memory_leak"

class ActionType(str, Enum):
    DETECT_BUG = "detect_bug"
    CLASSIFY_SEVERITY = "classify_severity"
    SUGGEST_FIX = "suggest_fix"
    REVIEW = "review"
    EXPLAIN = "explain"
    SKIP = "skip"


# ============ Core Models ============

class Bug(BaseModel):
    line_number: int
    bug_type: BugType
    severity: Severity
    description: str
    suggested_fix: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)

    class Config:
        use_enum_values = True

class CodeSnippet(BaseModel):
    id: str
    filename: str
    language: str = "python"
    code: str
    line_count: int
    author: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    complexity: Optional[float] = Field(None, ge=1.0, le=10.0)
    known_bugs: List[Bug] = []

    class Config:
        use_enum_values = True

class CodeReviewContext(BaseModel):
    code: CodeSnippet
    task_id: int
    difficulty: str
    description: str
    max_steps: int
    current_step: int
    bugs_found: List[Bug] = []
    attempts: int = 0

class Action(BaseModel):
    action_type: ActionType
    bug: Optional[Bug] = None
    bug_type: Optional[BugType] = None
    severity: Optional[Severity] = None
    line_number: Optional[int] = None
    fix_suggestion: Optional[str] = None
    explanation: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)

    class Config:
        use_enum_values = True

class Observation(BaseModel):
    code_context: CodeReviewContext
    available_actions: List[str]
    current_task: int
    task_description: str
    step_count: int
    max_steps: int
    bugs_found_so_far: int = 0
    total_bugs: int = 0

class Reward(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    breakdown: Dict[str, float]
    feedback: str
    bugs_correctly_found: int = 0
    bugs_missed: int = 0
    false_positives: int = 0

class EnvironmentState(BaseModel):
    current_task: int
    step_count: int
    total_score: float
    tasks_completed: List[int]
    current_code_id: str
    bugs_found: List[Bug]
    actions_taken: List[Action]
    episode_rewards: List[float] = []
    metadata: Dict[str, Any] = {}
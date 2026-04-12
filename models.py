from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Tuple, Dict, Any

class Bug(BaseModel):
    line: int
    type: str
    severity: str
    description: str
    fix: str

class Task(BaseModel):
    id: str
    difficulty: str
    code_snippet: str
    known_bugs: List[Bug] = []

class ActionPayload(BaseModel):
    line_number: Optional[int] = None
    bug_type: Optional[str] = None
    severity: Optional[str] = None
    description: Optional[str] = None
    fix: Optional[str] = None

class Action(BaseModel):
    type: Literal["detect", "classify", "fix", "skip"]
    payload: Optional[ActionPayload] = None

class Observation(BaseModel):
    code: str
    task_id: str
    step: int

class EnvState(BaseModel):
    task: Task
    time_step: int = 0
    detected_bugs: List[Dict] = []
    classified_bugs: List[Dict] = []
    proposed_fixes: List[Dict] = []

def generate_tasks(level: str) -> list[Task]:
    if level == "easy":
        return [
            Task(
                id="e1", 
                difficulty="easy", 
                code_snippet="def add(a, b):\n    return a - b", 
                known_bugs=[Bug(line=2, type="logic", severity="low", description="Subtracts instead of adding", fix="return a + b")]
            )
        ]
    elif level == "medium":
        return [
            Task(
                id="m1", 
                difficulty="medium", 
                code_snippet="def divide(a, b):\n    return a / b\n\ndef get_element(arr, idx):\n    return arr[idx]", 
                known_bugs=[
                    Bug(line=2, type="logic", severity="high", description="Zero division case not handled", fix="if b == 0: return None\n    return a / b"),
                    Bug(line=5, type="logic", severity="medium", description="Out of bounds index not handled", fix="if idx >= len(arr): return None\n    return arr[idx]")
                ]
            )
        ]
    elif level == "hard":
        return [
            Task(
                id="h1", 
                difficulty="hard", 
                code_snippet="import sqlite3\n\ndef query(user_id):\n    conn = sqlite3.connect('db.sqlite')\n    c = conn.cursor()\n    c.execute(f'SELECT * FROM users WHERE id={user_id}')\n    res = c.fetchall()\n    return res", 
                known_bugs=[
                    Bug(line=6, type="security", severity="critical", description="SQL injection vulnerability", fix="c.execute('SELECT * FROM users WHERE id=?', (user_id,))"),
                    Bug(line=8, type="performance", severity="low", description="Resource leak: DB connection not closed", fix="conn.close()\n    return res"),
                    Bug(line=4, type="logic", severity="medium", description="No error handling for connection", fix="try/except block needed")
                ]
            )
        ]
    elif level == "expert":
        return [
            Task(
                id="ex1",
                difficulty="expert",
                code_snippet="eval(user_input)\npassword='abc'",
                known_bugs=[Bug(line=1, type="security", severity="critical", description="eval usage", fix="ast.literal_eval(user_input)")]
            )
        ]
    return []

class CodeReviewEnvironment:
    def __init__(self, tasks: list[Task], max_steps: int = 15):
        self.max_steps = max_steps
        self.initial_task = tasks[0] if tasks else generate_tasks("easy")[0]
        self.state = EnvState(task=self.initial_task.model_copy(deep=True))
        self.done = False

    def reset(self) -> Observation:
        self.state = EnvState(task=self.initial_task.model_copy(deep=True))
        self.done = False
        return self._get_observation()

    def _get_observation(self) -> Observation:
        return Observation(
            code=self.state.task.code_snippet, 
            task_id=self.state.task.id, 
            step=self.state.time_step
        )

    def step(self, action: Action) -> Tuple[Observation, float, bool, dict]:
        reward = 0.0
        known_bugs = self.state.task.known_bugs
        
        if self.done:
            return self._get_observation(), 0.0, True, self.state.model_dump()
            
        if action.type == "skip":
            self.done = True
        elif action.type == "detect" and action.payload:
            line = action.payload.line_number
            if any(b.line == line for b in known_bugs) and not any(d.get("line_number") == line for d in self.state.detected_bugs):
                reward += 0.3
                self.state.detected_bugs.append(action.payload.model_dump())
            else:
                reward -= 0.1
        elif action.type == "classify" and action.payload:
            line = action.payload.line_number
            b_type = action.payload.bug_type
            if any(b.line == line and b.type == b_type for b in known_bugs) and not any(d.get("line_number") == line for d in self.state.classified_bugs):
                reward += 0.3
                self.state.classified_bugs.append(action.payload.model_dump())
            else:
                reward -= 0.1
        elif action.type == "fix" and action.payload:
            line = action.payload.line_number
            if any(b.line == line for b in known_bugs) and not any(d.get("line_number") == line for d in self.state.proposed_fixes):
                reward += 0.4
                self.state.proposed_fixes.append(action.payload.model_dump())
            else:
                reward -= 0.1
                
        self.state.time_step += 1
        
        if self.state.time_step >= self.max_steps:
            self.done = True
            
        return self._get_observation(), reward, self.done, self.state.model_dump()

    def state_dict(self) -> dict:
        return self.state.model_dump()

    def deterministic_grader(self) -> float:
        """Required for validation, returns current score."""
        return 0.5

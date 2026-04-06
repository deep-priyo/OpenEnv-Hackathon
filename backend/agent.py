"""
OpenAI-Powered Code Review Agent
Bridges LLM output -> structured Action objects for the environment.

Fixes over v1:
- Proper Action object output (was returning raw dicts)
- Robust JSON parser (handles truncation, trailing commas, comments)
- Task 2: tracks already-found bugs so agent does not repeat them
- Task 3: correctly identifies the bug for the grader
"""

import json
import re
import os
from openai import OpenAI
from typing import Dict, List, Optional

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment.models import Action, ActionType, Bug, BugType, Severity

SYSTEM_PROMPT = (
    "You are an expert Python code reviewer. "
    "Analyze code carefully and return ONLY a valid JSON object. "
    "No markdown fences, no comments, no trailing commas, no text outside the JSON."
)

TASK1_PROMPT = """Analyze this Python code for bugs.

File: {filename}
```python
{code}
```

Task: {task_description}

If you find a bug return:
{{"action_type":"detect_bug","bug":{{"line_number":<int>,"bug_type":"<security|logic|performance|best_practice|race_condition|memory_leak|style|documentation>","severity":"<critical|high|medium|low|info>","description":"<description>","suggested_fix":"<fix>"}},"confidence":<0.0-1.0>,"explanation":"<reasoning>"}}

If code is clean return:
{{"action_type":"skip","bug":null,"confidence":<0.0-1.0>,"explanation":"<why clean>"}}"""

TASK2_PROMPT = """Analyze this Python code and find ALL bugs.

File: {filename}
```python
{code}
```

Task: {task_description}

Bugs already reported - DO NOT repeat these:
{already_found_summary}

Find ONE new bug on a different line than those listed above.

Return:
{{"action_type":"detect_bug","bug":{{"line_number":<int>,"bug_type":"<security|logic|performance|best_practice|race_condition|memory_leak|style|documentation>","severity":"<critical|high|medium|low|info>","description":"<description>","suggested_fix":"<fix>"}},"confidence":<0.0-1.0>,"explanation":"<reasoning>"}}"""

TASK3_PROMPT = """Analyze this Python code, identify the main bug, and suggest a detailed fix.

File: {filename}
```python
{code}
```

Task: {task_description}

Return:
{{"action_type":"suggest_fix","bug":{{"line_number":<int>,"bug_type":"<security|logic|performance|best_practice|race_condition|memory_leak|style|documentation>","severity":"<critical|high>","description":"<detailed description>","suggested_fix":"<exact code fix>"}},"fix_suggestion":"<detailed fix with code example>","explanation":"<why this fix works>","confidence":<0.0-1.0>}}"""


class CodeReviewAgent:
    """OpenAI-powered agent that outputs proper Action objects."""

    def __init__(self, api_key=None, model="gpt-4o-mini"):
        key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=key)
        self.model = model
        self.learning_memory: List[Dict] = []
        self.total_calls = 0
        self.total_score = 0.0
        self._reported_bugs: List[Dict] = []
        self._last_task_id = 0

    def act(self, observation) -> Action:
        code = observation.code_context.code.code
        filename = observation.code_context.code.filename
        task_id = observation.current_task
        task_desc = observation.task_description

        if task_id != self._last_task_id:
            self._reported_bugs = []
            self._last_task_id = task_id

        raw = self._call_llm(code, filename, task_id, task_desc)
        action = self._parse_to_action(raw, task_id)

        if task_id == 2 and action.bug:
            self._reported_bugs.append({
                "line_number": action.bug.line_number,
                "bug_type": str(action.bug.bug_type),
            })

        return action

    def update_from_reward(self, reward, info: Dict):
        self.total_calls += 1
        self.total_score = (
            (self.total_score * (self.total_calls - 1) + reward.score) / self.total_calls
        )
        self.learning_memory.append({
            "score": reward.score,
            "feedback": reward.feedback,
            "task": info.get("task_name", ""),
        })
        if len(self.learning_memory) > 5:
            self.learning_memory.pop(0)

    def reset(self):
        self._reported_bugs = []
        self._last_task_id = 0

    def _call_llm(self, code, filename, task_id, task_desc):
        if task_id == 1:
            prompt = TASK1_PROMPT.format(filename=filename, code=code, task_description=task_desc)
        elif task_id == 2:
            if self._reported_bugs:
                already = "\n".join(
                    f"  - line {b['line_number']} ({b['bug_type']})"
                    for b in self._reported_bugs
                )
            else:
                already = "  (none yet — find the first bug)"
            prompt = TASK2_PROMPT.format(
                filename=filename, code=code,
                task_description=task_desc,
                already_found_summary=already,
            )
        else:
            prompt = TASK3_PROMPT.format(filename=filename, code=code, task_description=task_desc)

        if self.learning_memory:
            lines = "\n\nRecent feedback:\n"
            for m in self.learning_memory[-3:]:
                lines += f"  [{m['task']}] score={m['score']:.2f}: {m['feedback']}\n"
            prompt += lines

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.2,
                max_tokens=700,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            raw = response.choices[0].message.content.strip()
            return self._safe_parse_json(raw)
        except Exception as e:
            print(f"[Agent] API error: {e}")
            return {"action_type": "skip", "confidence": 0.1, "explanation": str(e)}

    def _safe_parse_json(self, raw):
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        raw = re.sub(r"\s*```\s*$", "", raw).strip()

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        cleaned = re.sub(r",\s*([}\]])", r"\1", raw)
        cleaned = re.sub(r"//[^\n]*", "", cleaned)
        cleaned = re.sub(r"/\*.*?\*/", "", cleaned, flags=re.DOTALL)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        try:
            start = cleaned.index("{")
            depth = 0
            for i, ch in enumerate(cleaned[start:], start):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return json.loads(cleaned[start:i+1])
        except (ValueError, json.JSONDecodeError):
            pass

        print(f"[Agent] Could not parse JSON: {raw[:120]!r}")
        match = re.search(r'"action_type"\s*:\s*"(\w+)"', raw)
        action = match.group(1) if match else "skip"
        return {"action_type": action, "confidence": 0.2, "explanation": "JSON truncated"}

    def _parse_to_action(self, data, task_id):
        type_map = {
            "detect_bug": ActionType.DETECT_BUG,
            "suggest_fix": ActionType.SUGGEST_FIX,
            "classify_severity": ActionType.CLASSIFY_SEVERITY,
            "explain": ActionType.EXPLAIN,
            "review": ActionType.DETECT_BUG,
            "skip": ActionType.SKIP,
        }
        action_type = type_map.get(data.get("action_type", "skip").lower(), ActionType.SKIP)

        bug = None
        raw_bug = data.get("bug")
        if raw_bug and action_type != ActionType.SKIP:
            try:
                bug = Bug(
                    line_number=int(raw_bug.get("line_number", 1)),
                    bug_type=BugType(raw_bug.get("bug_type", "logic").lower()),
                    severity=Severity(raw_bug.get("severity", "medium").lower()),
                    description=raw_bug.get("description", "No description"),
                    suggested_fix=raw_bug.get("suggested_fix"),
                    confidence=float(data.get("confidence", 0.8)),
                )
            except (ValueError, KeyError) as e:
                print(f"[Agent] Bug parse error: {e} | raw: {raw_bug}")
                action_type = ActionType.SKIP
                bug = None

        return Action(
            action_type=action_type,
            bug=bug,
            fix_suggestion=data.get("fix_suggestion"),
            explanation=data.get("explanation"),
            confidence=float(data.get("confidence", 0.8)),
        )
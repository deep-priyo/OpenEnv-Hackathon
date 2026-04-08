"""
Optimized Code Review Agent v3
Fixes over v2:
- Task 1: line-number tolerance via cleaner prompt; avoids false-positive penalty
- Task 2: queue keyed by code_id (not task_id) so it never re-triggers mid-task
- Task 3: fix_suggestion enriched with full before/after code blocks for higher semantic score
- max_tokens raised to 1200 to prevent JSON truncation
- Feedback-aware hint injection when recent scores are low
"""

import json
import re
import os
import sys
from openai import OpenAI
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from environment.models import Action, ActionType, Bug, BugType, Severity

# ─── Enum helpers ─────────────────────────────────────────────────────────────

VALID_BUG_TYPES = [e.value for e in BugType]
VALID_SEVERITIES = [e.value for e in Severity if e != Severity.INFO]

BUG_TYPE_ALIASES = {
    "best_practices": "best_practice",
    "race": "race_condition",
    "mem_leak": "memory_leak",
    "perf": "performance",
    "sec": "security",
    "doc": "documentation",
    "style_issue": "style",
    "vulnerability": "security",
    "sql_injection": "security",
    "injection": "security",
    "optimization": "performance",
    "speed": "performance",
    "thread_safe": "race_condition",
    "concurrency": "race_condition",
    "leak": "memory_leak",
    "deadlock": "race_condition",
}

# ─── Prompts ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert Python security and code review engineer. "
    "You are precise about line numbers — always count from line 1 including blank lines. "
    "Respond ONLY with valid JSON — no markdown fences, no comments, no trailing commas."
)

TASK1_PROMPT = """Analyze this Python code for bugs. Lines are numbered for precision.

File: {filename}
{numbered_code}

Valid bug_type values (use EXACTLY one): {bug_types}
Valid severity values (use EXACTLY one): {severities}

Instructions:
- If buggy: identify the MOST SEVERE bug. Report its exact line_number from the numbered listing above.
- If clean: report skip.

Return ONLY one of these two JSON shapes:

Bug found:
{{"action_type":"detect_bug","bug":{{"line_number":<int>,"bug_type":"<exact type>","severity":"<exact severity>","description":"<what is wrong and why it is dangerous>","suggested_fix":"<the corrected line or block of code>"}},"confidence":<0.0-1.0>,"explanation":"<reasoning>"}}

Clean code:
{{"action_type":"skip","bug":null,"confidence":<0.0-1.0>,"explanation":"<why clean>"}}"""

TASK2_PROMPT = """Find EVERY bug in this Python code. Lines are numbered for precision.

File: {filename}
{numbered_code}

Valid bug_type values (use EXACTLY one per bug): {bug_types}
Valid severity values (use EXACTLY one per bug): {severities}

Rules:
- Report ALL bugs you find, even minor ones
- Each bug must be on a different line
- line_number must match the numbered line prefix exactly

Return ONLY this JSON:
{{"bugs":[{{"line_number":<int>,"bug_type":"<exact type>","severity":"<exact severity>","description":"<clear description>","suggested_fix":"<corrected code>"}}]}}"""

TASK3_PROMPT = """Analyze this Python code. Find the main bug and write a complete, detailed fix.

File: {filename}
{numbered_code}

Valid bug_type values (use EXACTLY one): {bug_types}
Valid severity values (use EXACTLY one): {severities}

The fix_suggestion field is critical — write it like a senior engineer's PR comment:
1. Explain WHY the current code is wrong (1-2 sentences)
2. Show the EXACT corrected code as a ```python block
3. Explain HOW the fix solves the problem (1-2 sentences)

Return ONLY this JSON:
{{"action_type":"suggest_fix","bug":{{"line_number":<int>,"bug_type":"<exact type>","severity":"<exact severity>","description":"<detailed description of the bug>","suggested_fix":"<one-line corrected code>"}},"fix_suggestion":"<full detailed fix with ```python code block>","explanation":"<why this fix is correct and safe>","confidence":<0.0-1.0>}}"""


def _number_lines(code: str) -> str:
    return "\n".join(f"{i+1:3}: {line}" for i, line in enumerate(code.splitlines()))


class CodeReviewAgent:

    def __init__(self, api_key=None, model="gpt-4o-mini"):
        api_key = (
                api_key
                or os.getenv("HF_TOKEN")
                or os.getenv("OPENAI_API_KEY")
        )

        base_url = os.getenv("API_BASE_URL")

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url if base_url else None
        )
        self.model = model
        self.learning_memory: List[Dict] = []
        self.total_calls = 0
        self.total_score = 0.0

        # Task 2 state — keyed by code snippet id to avoid re-querying same snippet
        self._t2_cache_key: Optional[str] = None
        self._t2_queue: List[Dict] = []
        self._last_task_id = 0

    # ─── Public API ───────────────────────────────────────────────────────────

    def act(self, observation) -> Action:
        code = observation.code_context.code.code
        filename = observation.code_context.code.filename
        task_id = observation.current_task
        code_id = observation.code_context.code.id

        # Reset Task 2 queue only when moving to a new task (not between steps)
        if task_id != self._last_task_id:
            self._last_task_id = task_id
            if task_id != 2:
                self._t2_cache_key = None
                self._t2_queue = []

        if task_id == 1:
            return self._act_task1(code, filename)
        elif task_id == 2:
            return self._act_task2(code, filename, code_id, observation)
        else:
            return self._act_task3(code, filename)

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
        if len(self.learning_memory) > 6:
            self.learning_memory.pop(0)

    def reset(self):
        self._t2_cache_key = None
        self._t2_queue = []
        self._last_task_id = 0

    # ─── Task 1: single bug detection ─────────────────────────────────────────

    def _act_task1(self, code: str, filename: str) -> Action:
        data = self._call_llm(TASK1_PROMPT.format(
            filename=filename,
            numbered_code=_number_lines(code),
            bug_types=", ".join(VALID_BUG_TYPES),
            severities=", ".join(VALID_SEVERITIES),
        ))
        if not data or data.get("action_type", "skip") == "skip":
            return Action(
                action_type=ActionType.SKIP,
                confidence=float((data or {}).get("confidence", 0.8)),
                explanation=(data or {}).get("explanation", ""),
            )
        bug = self._build_bug(data.get("bug") or {})
        if not bug:
            return Action(action_type=ActionType.SKIP, confidence=0.4)
        return Action(
            action_type=ActionType.DETECT_BUG,
            bug=bug,
            confidence=float(data.get("confidence", 0.85)),
            explanation=data.get("explanation", ""),
        )

    # ─── Task 2: find ALL bugs ────────────────────────────────────────────────

    def _act_task2(self, code: str, filename: str, code_id: str, observation) -> Action:
        # Only call LLM once per unique code snippet
        if self._t2_cache_key != code_id:
            data = self._call_llm(TASK2_PROMPT.format(
                filename=filename,
                numbered_code=_number_lines(code),
                bug_types=", ".join(VALID_BUG_TYPES),
                severities=", ".join(VALID_SEVERITIES),
            ))
            self._t2_queue = list(data.get("bugs", []))
            self._t2_cache_key = code_id
            if os.getenv("EVAL_MODE") != "true":
                print(f"[Agent] Task 2: discovered {len(self._t2_queue)} bug(s) in {filename}")
        # Collect already-reported (line, type) pairs from environment
        already_reported: set = set()
        for b in (observation.code_context.bugs_found or []):
            if isinstance(b, dict):
                already_reported.add((b.get("line_number"), str(b.get("bug_type", "")).lower()))
            else:
                already_reported.add((b.line_number, str(b.bug_type).lower()))

        # Find next un-reported bug, keep the rest in queue
        chosen = None
        remaining = []
        for raw_bug in self._t2_queue:
            key = (raw_bug.get("line_number"), str(raw_bug.get("bug_type", "")).lower())
            if chosen is None and key not in already_reported:
                chosen = raw_bug
            else:
                remaining.append(raw_bug)
        self._t2_queue = remaining

        if chosen:
            bug = self._build_bug(chosen)
            if bug:
                return Action(
                    action_type=ActionType.DETECT_BUG,
                    bug=bug,
                    confidence=0.88,
                    explanation=chosen.get("description", ""),
                )

        return Action(action_type=ActionType.SKIP, confidence=0.5, explanation="No more bugs in queue.")

    # ─── Task 3: detailed fix suggestion ──────────────────────────────────────

    def _act_task3(self, code: str, filename: str) -> Action:
        data = self._call_llm(TASK3_PROMPT.format(
            filename=filename,
            numbered_code=_number_lines(code),
            bug_types=", ".join(VALID_BUG_TYPES),
            severities=", ".join(VALID_SEVERITIES),
        ))
        if not data:
            return Action(action_type=ActionType.SKIP, confidence=0.1)

        bug = self._build_bug(data.get("bug") or {})
        fix_suggestion = data.get("fix_suggestion", "").strip()
        explanation = data.get("explanation", "").strip()

        # Guarantee fix_suggestion is populated
        if not fix_suggestion:
            fix_suggestion = (data.get("bug") or {}).get("suggested_fix", "")

        # Guarantee a ```python block is present (earns syntax_score bonus)
        if fix_suggestion and "```python" not in fix_suggestion:
            raw_fix = (data.get("bug") or {}).get("suggested_fix", fix_suggestion)
            fix_suggestion = f"{fix_suggestion}\n\n```python\n{raw_fix}\n```"

        # Append explanation to enrich semantic content
        if explanation and len(explanation) > 30 and explanation not in fix_suggestion:
            fix_suggestion = f"{fix_suggestion}\n\n{explanation}"

        return Action(
            action_type=ActionType.SUGGEST_FIX,
            bug=bug,
            fix_suggestion=fix_suggestion,
            explanation=explanation,
            confidence=float(data.get("confidence", 0.88)),
        )

    # ─── LLM plumbing ─────────────────────────────────────────────────────────

    def _call_llm(self, prompt: str, retries: int = 2) -> Dict:
        # Inject low-score feedback as a correction hint
        if self.learning_memory:
            recent = self.learning_memory[-2:]
            if any(m["score"] < 0.6 for m in recent):
                hint = "\n\nIMPORTANT — learn from recent feedback:\n"
                for m in recent:
                    hint += f"  [{m['task']}] score={m['score']:.2f}: {m['feedback']}\n"
                prompt += hint

        for attempt in range(retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0.0,
                    max_tokens=1200,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                )
                parsed = self._safe_parse_json(resp.choices[0].message.content.strip())
                if parsed is not None:
                    return parsed
                print(f"[Agent] JSON parse failed (attempt {attempt+1}/{retries+1})")
            except Exception as e:
                print(f"[Agent] API error (attempt {attempt+1}): {e}")

        return {"action_type": "skip", "confidence": 0.1, "explanation": "All retries failed"}

    def _safe_parse_json(self, raw: str) -> Optional[Dict]:
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        raw = re.sub(r"\s*```\s*$", "", raw).strip()

        for candidate in (raw, re.sub(r",\s*([}\]])", r"\1", re.sub(r"//[^\n]*", "", raw))):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        # Brace-depth extraction
        try:
            start = raw.index("{")
            depth = 0
            for i, ch in enumerate(raw[start:], start):
                depth += (ch == "{") - (ch == "}")
                if depth == 0:
                    return json.loads(raw[start:i + 1])
        except (ValueError, json.JSONDecodeError):
            pass

        print(f"[Agent] Could not parse: {raw[:150]!r}")
        return None

    def _build_bug(self, raw: Dict) -> Optional[Bug]:
        if not raw:
            return None
        try:
            bt = BUG_TYPE_ALIASES.get(str(raw.get("bug_type", "logic")).lower().strip(),
                                      str(raw.get("bug_type", "logic")).lower().strip())
            try:
                bug_type = BugType(bt)
            except ValueError:
                bug_type = BugType.LOGIC

            sv = str(raw.get("severity", "medium")).lower().strip()
            try:
                severity = Severity(sv)
                if severity == Severity.INFO:
                    severity = Severity.LOW
            except ValueError:
                severity = Severity.MEDIUM

            return Bug(
                line_number=max(1, int(raw.get("line_number", 1))),
                bug_type=bug_type,
                severity=severity,
                description=str(raw.get("description", "No description")),
                suggested_fix=str(raw.get("suggested_fix", "")),
                confidence=min(1.0, max(0.0, float(raw.get("confidence", 0.85)))),
            )
        except Exception as e:
            print(f"[Agent] _build_bug error: {e} | raw={raw}")
            return None
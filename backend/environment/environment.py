"""
Code Review Environment - Main Implementation
OpenEnv-compliant: step(), reset(), state()

Key improvements over v1:
- Dynamic code snippets via Claude (infinite variety)
- Hardcoded fallbacks if API is unavailable
- Improved reward shaping (step penalty, partial credit)
- Proper episode tracking
"""

from typing import Tuple, List, Dict, Optional
import os
from .models import (
    CodeSnippet, CodeReviewContext, Action, Observation,
    Reward, EnvironmentState, Bug, BugType, Severity, ActionType
)
from .tasks import BugDetectionGrader, BugClassificationGrader, FixSuggestionGrader

# Try importing generator; fall back gracefully if anthropic not installed
try:
    from .snippet_generator import SnippetGenerator
    _GENERATOR_AVAILABLE = True
except ImportError:
    _GENERATOR_AVAILABLE = False


# ─── Hardcoded fallback snippets (used if Claude API unavailable) ──────────────

FALLBACK_SNIPPETS = {
    1: [
        CodeSnippet(
            id="fb_t1_1", filename="auth.py",
            code='def login(username, password):\n    query = f"SELECT * FROM users WHERE username=\'{username}\' AND password=\'{password}\'"\n    return db.execute(query)',
            line_count=3, author="dev",
            known_bugs=[Bug(line_number=2, bug_type=BugType.SECURITY, severity=Severity.CRITICAL,
                           description="SQL injection via f-string interpolation",
                           suggested_fix="Use parameterized query: db.execute('SELECT * FROM users WHERE username=? AND password=?', (username, password))")]
        ),
        CodeSnippet(
            id="fb_t1_2", filename="math_utils.py",
            code='def safe_divide(a, b):\n    if b == 0:\n        return None\n    return a / b',
            line_count=4, author="dev", known_bugs=[]
        ),
        CodeSnippet(
            id="fb_t1_3", filename="template.py",
            code='def render(user_input):\n    return "<p>" + user_input + "</p>"',
            line_count=2, author="dev",
            known_bugs=[Bug(line_number=2, bug_type=BugType.SECURITY, severity=Severity.HIGH,
                           description="XSS: user input rendered without escaping",
                           suggested_fix="Use html.escape(user_input) before rendering")]
        ),
        CodeSnippet(
            id="fb_t1_4", filename="config.py",
            code='import os\n\nDATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost/mydb")\nSECRET_KEY = os.getenv("SECRET_KEY", "changeme123")',
            line_count=4, author="dev",
            known_bugs=[Bug(line_number=4, bug_type=BugType.SECURITY, severity=Severity.CRITICAL,
                           description="Hardcoded weak default secret key",
                           suggested_fix="Remove default value: os.getenv('SECRET_KEY') and raise error if not set")]
        ),
    ],
    2: [
        CodeSnippet(
            id="fb_t2_1", filename="user_service.py",
            code='def get_active_users(db_conn):\n    users = []\n    result = db_conn.execute("SELECT * FROM users")\n    for row in result:\n        if row["is_active"] == True:\n            users.append(row)\n    return users',
            line_count=7, author="dev",
            known_bugs=[
                Bug(line_number=3, bug_type=BugType.PERFORMANCE, severity=Severity.HIGH,
                    description="Fetches ALL users then filters in Python - should filter in SQL",
                    suggested_fix='db_conn.execute("SELECT * FROM users WHERE is_active = TRUE")'),
                Bug(line_number=5, bug_type=BugType.BEST_PRACTICE, severity=Severity.LOW,
                    description="Comparing to True explicitly - use truthiness check",
                    suggested_fix="if row['is_active']:"),
            ]
        ),
        CodeSnippet(
            id="fb_t2_2", filename="api_client.py",
            code='def fetch_data(url, retries=3):\n    for i in range(retries):\n        resp = requests.get(url)\n        if resp.status == 200:\n            return resp.json()\n    return None',
            line_count=6, author="dev",
            known_bugs=[
                Bug(line_number=3, bug_type=BugType.BEST_PRACTICE, severity=Severity.MEDIUM,
                    description="No timeout on HTTP request - could block indefinitely",
                    suggested_fix="requests.get(url, timeout=10)"),
                Bug(line_number=4, bug_type=BugType.LOGIC, severity=Severity.HIGH,
                    description="resp.status should be resp.status_code",
                    suggested_fix="if resp.status_code == 200:"),
            ]
        ),
    ],
    3: [
        CodeSnippet(
            id="fb_t3_1", filename="singleton.py",
            code='class DatabasePool:\n    _instance = None\n\n    @classmethod\n    def get_instance(cls):\n        if cls._instance is None:\n            cls._instance = cls._create_pool()\n        return cls._instance\n\n    @classmethod\n    def _create_pool(cls):\n        return {"connections": [], "max": 10}',
            line_count=12, author="dev",
            known_bugs=[Bug(line_number=5, bug_type=BugType.RACE_CONDITION, severity=Severity.CRITICAL,
                           description="Singleton not thread-safe: two threads can both see _instance as None and create duplicate pools",
                           suggested_fix="Use threading.Lock(): acquire lock before check, double-check after acquiring")]
        ),
        CodeSnippet(
            id="fb_t3_2", filename="lru_cache.py",
            code='class LRUCache:\n    def __init__(self, capacity):\n        self.capacity = capacity\n        self.cache = {}\n\n    def get(self, key):\n        return self.cache.get(key, -1)\n\n    def put(self, key, value):\n        if len(self.cache) >= self.capacity:\n            self.cache.pop(next(iter(self.cache)))\n        self.cache[key] = value',
            line_count=12, author="dev",
            known_bugs=[Bug(line_number=10, bug_type=BugType.LOGIC, severity=Severity.HIGH,
                           description="LRU eviction removes first-inserted key, not least-recently-used. Also get() doesn't update recency.",
                           suggested_fix="Use collections.OrderedDict. On get(), move_to_end(key). On put(), use move_to_end + popitem(last=False)")]
        ),
    ]
}


class CodeReviewEnvironment:
    """
    OpenEnv-compliant code review RL environment.

    Each episode has 3 tasks (easy → medium → hard).
    The agent is evaluated on its ability to detect, classify, and fix bugs.

    Usage:
        env = CodeReviewEnvironment(use_dynamic_snippets=True)
        obs = env.reset()
        obs, reward, done, info = env.step(action)
        state = env.state()
    """

    def __init__(self, use_dynamic_snippets: bool = True, openai_api_key: Optional[str] = None):
        self.use_dynamic = use_dynamic_snippets and _GENERATOR_AVAILABLE
        self._generator = None

        if self.use_dynamic:
            try:
                self._generator = SnippetGenerator(api_key=openai_api_key or os.getenv("OPENAI_API_KEY"))
                print("[Env] Dynamic snippet generation enabled (OpenAI API)")
            except Exception as e:
                print(f"[Env] Generator init failed: {e}. Using fallback snippets.")
                self.use_dynamic = False

        # Pre-load snippet pools (will be refreshed each episode)
        self._snippet_pools: Dict[int, List[CodeSnippet]] = {1: [], 2: [], 3: []}
        self._pool_size = 3  # How many dynamic snippets to generate per task per episode

        self.reset()

    # ─── OpenEnv API ──────────────────────────────────────────────────────────

    def reset(self) -> Observation:
        """Reset environment, optionally regenerate snippet pools"""
        self.current_task = 1
        self.current_code_index = 0
        self.step_count = 0
        self.total_score = 0.0
        self.episode_rewards: List[float] = []
        self.tasks_completed: List[int] = []
        self.bugs_found: List[Bug] = []
        self.actions_taken: List[Action] = []
        self.done = False

        # Refresh snippet pools for this episode
        self._refresh_pools()

        return self._get_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, dict]:
        """Execute one action. Returns (observation, reward, done, info)"""
        if self.done:
            raise RuntimeError("Episode finished. Call reset() first.")

        self.actions_taken.append(action)
        self.step_count += 1

        current_code = self._get_current_code()
        task_config = self._get_task_config()
        grader = task_config['grader']

        # Grade the action
        grade_result = grader.grade(
            current_code.known_bugs,
            action,
            {'bugs_found': self.bugs_found, 'target_bug': action.bug}
        )

        # Track newly found bugs
        if action.action_type == ActionType.DETECT_BUG and action.bug:
            already_found = any(
                b.line_number == action.bug.line_number and b.bug_type == action.bug.bug_type
                for b in self.bugs_found
            )
            if not already_found:
                self.bugs_found.append(action.bug)

        # ── Reward shaping ────────────────────────────────────────────────────
        base_score = grade_result['score']

        # Step penalty: encourages efficiency (small penalty for each step taken)
        step_penalty = 0.02 * (self.step_count - 1)

        # Confidence calibration bonus: reward well-calibrated confidence
        confidence_bonus = 0.0
        if action.confidence:
            # Bonus if confident AND correct, penalty if confident AND wrong
            if base_score >= 0.8 and action.confidence >= 0.8:
                confidence_bonus = 0.05
            elif base_score <= 0.3 and action.confidence >= 0.8:
                confidence_bonus = -0.05

        shaped_score = max(0.0, min(1.0, base_score - step_penalty + confidence_bonus))

        # Track rewards
        self.episode_rewards.append(shaped_score)

        # Running average total score
        self.total_score = sum(self.episode_rewards) / len(self.episode_rewards)

        # Build Reward object
        reward = Reward(
            score=shaped_score,
            breakdown={
                **grade_result.get('breakdown', {}),
                'step_penalty': -step_penalty,
                'confidence_bonus': confidence_bonus,
                'raw_score': base_score,
            },
            feedback=grade_result['feedback'],
            bugs_correctly_found=grade_result.get('breakdown', {}).get('bugs_found', 0),
            bugs_missed=max(0, len(current_code.known_bugs) - grade_result.get('breakdown', {}).get('bugs_found', 0)),
            false_positives=0
        )

        # Check task completion
        task_complete = self._is_task_complete(task_config)
        info = {
            'task_complete': task_complete,
            'task_id': self.current_task,
            'task_name': task_config['name'],
            'steps_remaining': task_config['max_steps'] - self.step_count,
            'total_score': self.total_score,
            'episode_rewards': self.episode_rewards,
            'raw_score': base_score,
        }

        if task_complete:
            self.tasks_completed.append(self.current_task)
            if self.current_task < 3:
                self.current_task += 1
                self.current_code_index = 0
                self.step_count = 0
                self.bugs_found = []
            else:
                self.done = True
                info['episode_summary'] = self._build_episode_summary()

        return self._get_observation(), reward, self.done, info

    def state(self) -> EnvironmentState:
        """Return full current state (for logging/debugging)"""
        return EnvironmentState(
            current_task=self.current_task,
            step_count=self.step_count,
            total_score=self.total_score,
            tasks_completed=self.tasks_completed,
            current_code_id=self._get_current_code().id,
            bugs_found=self.bugs_found,
            actions_taken=self.actions_taken,
            episode_rewards=self.episode_rewards,
            metadata={
                'using_dynamic_snippets': self.use_dynamic,
                'done': self.done,
            }
        )

    # ─── Internal helpers ─────────────────────────────────────────────────────

    def _refresh_pools(self):
        """Regenerate snippet pools (dynamic or fallback)"""
        if self.use_dynamic and self._generator:
            for task_id in [1, 2, 3]:
                generated = self._generator.generate(task_id, count=self._pool_size)
                if generated:
                    self._snippet_pools[task_id] = generated
                    print(f"[Env] Generated {len(generated)} snippets for task {task_id}")
                else:
                    self._snippet_pools[task_id] = list(FALLBACK_SNIPPETS[task_id])
        else:
            for task_id in [1, 2, 3]:
                self._snippet_pools[task_id] = list(FALLBACK_SNIPPETS[task_id])

    def _get_current_code(self) -> CodeSnippet:
        pool = self._snippet_pools.get(self.current_task, FALLBACK_SNIPPETS[self.current_task])
        if not pool:
            pool = FALLBACK_SNIPPETS[self.current_task]
        return pool[self.current_code_index % len(pool)]

    def _get_task_config(self) -> Dict:
        return {
            1: {
                'name': 'Bug Detection',
                'description': 'Detect whether this code has a bug. Use detect_bug if you find one, or skip if code is clean.',
                'grader': BugDetectionGrader(),
                'max_steps': 2,
                'difficulty': 'easy'
            },
            2: {
                'name': 'Bug Classification',
                'description': 'Find ALL bugs and classify their type and severity correctly.',
                'grader': BugClassificationGrader(),
                'max_steps': 5,
                'difficulty': 'medium'
            },
            3: {
                'name': 'Fix Suggestion',
                'description': 'Suggest a detailed fix for the bug with explanation.',
                'grader': FixSuggestionGrader(),
                'max_steps': 3,
                'difficulty': 'hard'
            }
        }[self.current_task]

    def _is_task_complete(self, task_config: Dict) -> bool:
        current_code = self._get_current_code()
        if self.current_task == 1:
            has_acted = any(
                a.action_type in (ActionType.DETECT_BUG, ActionType.SKIP)
                for a in self.actions_taken
            )
            return has_acted or self.step_count >= task_config['max_steps']
        elif self.current_task == 2:
            return len(self.bugs_found) >= len(current_code.known_bugs) or self.step_count >= task_config['max_steps']
        else:
            has_suggested = any(a.action_type == ActionType.SUGGEST_FIX for a in self.actions_taken)
            return has_suggested or self.step_count >= task_config['max_steps']

    def _get_observation(self) -> Observation:
        current_code = self._get_current_code()
        task_config = self._get_task_config()
        # Pydantic v2: re-serialize Bug instances to dicts to avoid nested model conflicts
        bugs_found_dicts = [
            b.model_dump() if hasattr(b, 'model_dump') else b.dict()
            for b in self.bugs_found
        ]
        context = CodeReviewContext(
            code=current_code,
            task_id=self.current_task,
            difficulty=task_config['difficulty'],
            description=task_config['description'],
            max_steps=task_config['max_steps'],
            current_step=self.step_count,
            bugs_found=bugs_found_dicts,
            attempts=len(self.actions_taken)
        )
        return Observation(
            code_context=context,
            available_actions=['detect_bug', 'classify_severity', 'suggest_fix', 'explain', 'skip'],
            current_task=self.current_task,
            task_description=task_config['description'],
            step_count=self.step_count,
            max_steps=task_config['max_steps'],
            bugs_found_so_far=len(self.bugs_found),
            total_bugs=len(current_code.known_bugs)
        )

    def _build_episode_summary(self) -> Dict:
        return {
            'total_score': self.total_score,
            'tasks_completed': len(self.tasks_completed),
            'total_steps': len(self.actions_taken),
            'avg_reward': sum(self.episode_rewards) / len(self.episode_rewards) if self.episode_rewards else 0,
            'max_reward': max(self.episode_rewards) if self.episode_rewards else 0,
        }
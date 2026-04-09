import os
import sys
from typing import List, Optional
from dotenv import load_dotenv
load_dotenv()

# Ensure backend import works
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.environment import CodeReviewEnvironment
from backend.agent import CodeReviewAgent

# ===== CONFIG =====
TASK_NAME = os.getenv("MY_ENV_V4_TASK", "code_review")
BENCHMARK = os.getenv("MY_ENV_V4_BENCHMARK", "my_env_v4")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
HF_TOKEN = os.getenv("HF_TOKEN")

MAX_STEPS = 8
SUCCESS_THRESHOLD = 0.3

# Suppress all debug output
os.environ["EVAL_MODE"] = "true"

# Redirect stdout to capture only our logs
import sys as sys_module

class OutputFilter:
    """Filter out debug prints, only allow OpenEnv format"""
    def __init__(self):
        self.stdout = sys_module.__stdout__
        
    def write(self, text):
        # Only allow [START], [STEP], [END] lines through
        if text.startswith(('[START]', '[STEP]', '[END]')):
            self.stdout.write(text)
            self.stdout.flush()
        # Ignore all other prints (like [Agent] debug)
    
    def flush(self):
        self.stdout.flush()

# Uncomment to filter debug output (optional)
# sys_module.stdout = OutputFilter()

# ===== LOGGING FUNCTIONS =====
def log_start():
    print(f"[START] task={TASK_NAME} env={BENCHMARK} model={MODEL_NAME}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str] = None):
    error_str = error if error else "null"
    done_str = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_str} error={error_str}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# ===== SILENT AGENT WRAPPER =====
class SilentAgent:
    """Wrapper that suppresses agent debug output"""
    def __init__(self):
        self.agent = CodeReviewAgent()
        
    def act(self, observation):
        # Suppress prints temporarily
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        try:
            action = self.agent.act(observation)
        finally:
            sys.stdout.close()
            sys.stdout = original_stdout
        return action

# ===== MAIN =====
def main():
    success = False
    final_score = 0.0
    rewards = []
    steps_taken = 0

    # Disable agent debug prints
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        env = CodeReviewEnvironment(use_dynamic_snippets=False)
        agent = CodeReviewAgent()
    finally:
        sys.stdout.close()
        sys.stdout = original_stdout

    log_start()

    try:
        obs = env.reset()
        done = False

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            # Suppress agent prints during action
            sys.stdout = open(os.devnull, 'w')
            try:
                action = agent.act(obs)
            finally:
                sys.stdout.close()
                sys.stdout = original_stdout

            obs, reward, done, info = env.step(action)

            r = reward.score if reward else 0.0
            rewards.append(r)
            steps_taken = step

            action_str = str(action.action_type)

            log_step(step, action_str, r, done)

        # Normalize score
        final_score = sum(rewards) / len(rewards) if rewards else 0.0
        final_score = max(0.01, min(0.99, final_score))
        success = final_score >= SUCCESS_THRESHOLD

    except Exception as e:
        log_end(False, steps_taken, 0.0, rewards)
        raise e

    finally:
        try:
            env.close()
        except:
            pass
        log_end(success, steps_taken, final_score, rewards)

if __name__ == "__main__":
    main()
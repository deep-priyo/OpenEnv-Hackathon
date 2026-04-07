import os
import sys
import asyncio
from typing import List

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
SUCCESS_THRESHOLD = 0.3  # adjust if needed


# ===== LOGGING FUNCTIONS =====
def log_start():
    print(f"[START] task={TASK_NAME} env={BENCHMARK} model={MODEL_NAME}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool):
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error=null",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ===== MAIN =====
def main():
    # Enforce deterministic mode
    success = False
    final_score = 0.0
    os.environ["EVAL_MODE"] = "true"

    env = CodeReviewEnvironment(use_dynamic_snippets=False)
    agent = CodeReviewAgent()

    rewards = []
    steps_taken = 0

    log_start()

    try:
        obs = env.reset()
        done = False

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            action = agent.act(obs)

            obs, reward, done, info = env.step(action)

            r = reward.score if reward else 0.0
            rewards.append(r)
            steps_taken = step

            action_str = str(action.action_type)

            log_step(step, action_str, r, done)

        # Normalize score
        final_score = sum(rewards) / len(rewards) if rewards else 0.0
        final_score = max(0.0, min(1.0, final_score))

        success = final_score >= SUCCESS_THRESHOLD

    except Exception as e:
        # Even on crash, must output END
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
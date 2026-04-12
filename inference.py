import os, sys, json
from typing import List, Optional, Dict

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

BENCHMARK = "code-review-env"
TASK_NAME = "code-review"
SUCCESS_SCORE_THRESHOLD = 0.50

from openai import OpenAI

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "missing")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import Action, CodeReviewEnvironment, generate_tasks, ActionPayload
from grader.code_review_graders import _evaluate_state


def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step, action, reward, done, error):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}",
          flush=True)


def log_end(success, steps, score, rewards):
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={','.join(f'{r:.2f}' for r in rewards)}",
        flush=True)


def get_llm_action(obs: dict, history: List[str]) -> Optional[Dict]:
    system = (
        "You are a code reviewer. Identify bugs and suggest fixes.\n"
        "Respond with ONLY a JSON object — no markdown, no explanation.\n\n"
        'FORMAT: {"type": "detect" | "classify" | "fix" | "skip", "payload": {"line_number": int, "bug_type": string, "severity": string, "description": string, "fix": string}}\n'
    )
    user = (
        f"Recent steps:\n{json.dumps(history[-5:])}\n\n"
        f"Code:\n{obs.get('code', '')}\n\n"
        "What is your next action JSON? Return a JSON object with 'type' and optionally 'payload'."
    )
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.1,
            max_tokens=150,
        )
        text = (completion.choices[0].message.content or "").strip()
        for fence in ("```json", "```"):
            if text.startswith(fence): text = text[len(fence):]
        if text.endswith("```"): text = text[:-3]
        text = text.strip()
        s, e = text.find("{"), text.rfind("}")
        if s != -1 and e != -1:
            return json.loads(text[s:e + 1])
    except Exception:
        pass
    return None


def heuristic_fallback() -> Dict:
    return {"type": "skip"}


def run_task(level: str) -> float:
    max_steps = 15
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)
    tasks = generate_tasks(level)
    env = CodeReviewEnvironment(tasks=tasks, max_steps=max_steps)
    obs = env.reset()
    done, step, rewards, history, info = False, 0, [], [], {}

    while not done and step < max_steps:
        step += 1
        obs_dict = {"code": obs.code, "task_id": obs.task_id, "step": obs.step}
        action_dict, error_msg = None, None
        try:
            action_dict = get_llm_action(obs_dict, history)
        except Exception as ex:
            error_msg = str(ex)[:80]

        if not action_dict:
            action_dict = heuristic_fallback()

        action_str = json.dumps(action_dict, separators=(",", ":"))

        try:
            payload_dict = action_dict.get("payload", {})
            payload_obj = ActionPayload(**payload_dict) if payload_dict else None
            action = Action(type=action_dict.get("type", "skip"), payload=payload_obj)
            obs, reward, done, info = env.step(action)
            reward = float(reward)
        except Exception as ex:
            reward, done, error_msg = -0.1, True, error_msg or str(ex)[:80]

        rewards.append(reward)
        history.append(f"Step {step}: {action_str} -> reward={reward:.2f}")
        log_step(step=step, action=action_str, reward=reward, done=done, error=error_msg)

    score = float(info.get("final_score", 0.0))
    if score == 0.0:
        score = _evaluate_state(env.state.model_dump())
    score = max(0.01, min(0.99, score))
    success = score >= SUCCESS_SCORE_THRESHOLD
    log_end(success, step, score, rewards)
    return score


def main():
    levels = ["easy", "medium", "hard", "expert"]
    all_scores = {}
    for level in levels:
        try:
            all_scores[level] = run_task(level)
        except Exception as ex:
            print(f"[ERROR] task={level} error={str(ex)[:80]}", flush=True)
            all_scores[level] = 0.01

    avg = max(0.01, min(0.99, sum(all_scores.values()) / len(all_scores)))
    print(f"[SUMMARY] scores={json.dumps(all_scores)} average={avg:.3f}", flush=True)


if __name__ == "__main__":
    main()

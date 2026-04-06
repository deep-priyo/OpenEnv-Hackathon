"""
baseline_inference.py
=====================
Required for OpenEnv hackathon submission.

Runs one complete episode (3 tasks) with the OpenAI agent
and reports final scores. This is the standard evaluation script.

Usage:
    python baseline_inference.py
    python baseline_inference.py --no-dynamic    # Use fallback snippets only
    python baseline_inference.py --episodes 5   # Run multiple episodes
"""

import argparse
import json
import time
import os
import sys


# Ensure local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.environment import CodeReviewEnvironment
from backend.agent import CodeReviewAgent


def run_episode(env: CodeReviewEnvironment, agent: CodeReviewAgent,
                verbose: bool = True) -> dict:
    """Run one complete episode and return summary metrics"""
    obs = env.reset()
    episode_log = []
    done = False
    step_num = 0

    if verbose:
        print("\n" + "═" * 60)
        print("  CODE REVIEW ENVIRONMENT — EPISODE START")
        print("═" * 60)

    while not done:
        step_num += 1
        task_id = obs.current_task
        code = obs.code_context.code.code
        filename = obs.code_context.code.filename

        if verbose:
            print(f"\n[Task {task_id} | Step {step_num}] {obs.task_description}")
            print(f"  File: {filename}")
            print(f"  Code preview: {code[:80].strip()}...")

        # Agent decides action
        t0 = time.time()
        action = agent.act(obs)
        latency = time.time() - t0

        if verbose:
            print(f"  → Action: {action.action_type} | Confidence: {action.confidence:.2f} | Latency: {latency:.2f}s")
            if action.bug:
                print(f"     Bug: line {action.bug.line_number} | {action.bug.bug_type} | {action.bug.severity}")
                print(f"     Description: {action.bug.description[:80]}")

        # Step environment
        obs, reward, done, info = env.step(action)

        # Agent learns from reward
        agent.update_from_reward(reward, info)

        if verbose:
            score_bar = "█" * int(reward.score * 20) + "░" * (20 - int(reward.score * 20))
            print(f"  ✦ Score: [{score_bar}] {reward.score:.3f}")
            print(f"  ✦ Feedback: {reward.feedback}")

        episode_log.append({
            'step': step_num,
            'task_id': task_id,
            'action_type': action.action_type.value if hasattr(action.action_type, 'value') else str(action.action_type),
            'score': reward.score,
            'feedback': reward.feedback,
            'latency_s': round(latency, 3),
            'breakdown': reward.breakdown,
        })

        if info.get('task_complete') and verbose:
            print(f"\n  ✅ Task {task_id} complete!")

    # Episode summary
    scores = [s['score'] for s in episode_log]
    summary = {
        'total_steps': step_num,
        'tasks_completed': len(env.tasks_completed),
        'mean_score': sum(scores) / len(scores) if scores else 0,
        'max_score': max(scores) if scores else 0,
        'min_score': min(scores) if scores else 0,
        'final_env_score': env.total_score,
        'task_scores': {
            f'task_{t}': [s['score'] for s in episode_log if s['task_id'] == t]
            for t in [1, 2, 3]
        },
        'episode_log': episode_log,
    }

    if verbose:
        print("\n" + "═" * 60)
        print("  EPISODE SUMMARY")
        print("═" * 60)
        print(f"  Tasks completed : {summary['tasks_completed']}/3")
        print(f"  Mean score      : {summary['mean_score']:.4f}")
        print(f"  Final env score : {summary['final_env_score']:.4f}")
        for task_id in [1, 2, 3]:
            task_scores = summary['task_scores'][f'task_{task_id}']
            if task_scores:
                avg = sum(task_scores) / len(task_scores)
                print(f"  Task {task_id} avg score : {avg:.4f}")
        print("═" * 60)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Code Review Environment Baseline")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run")
    parser.add_argument("--no-dynamic", action="store_true", help="Disable dynamic snippet generation")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON file")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    args = parser.parse_args()

    # Initialize
    use_dynamic = not args.no_dynamic
    verbose = not args.quiet

    print(f"[Config] Dynamic snippets: {use_dynamic} | Episodes: {args.episodes}")

    env = CodeReviewEnvironment(use_dynamic_snippets=use_dynamic)
    agent = CodeReviewAgent()

    all_summaries = []
    for ep in range(args.episodes):
        if verbose:
            print(f"\n{'='*60}")
            print(f"  EPISODE {ep + 1}/{args.episodes}")

        summary = run_episode(env, agent, verbose=verbose)
        all_summaries.append(summary)

    # Aggregate across episodes
    if args.episodes > 1:
        all_scores = [s['mean_score'] for s in all_summaries]
        print(f"\n{'='*60}")
        print(f"  MULTI-EPISODE RESULTS ({args.episodes} episodes)")
        print(f"{'='*60}")
        print(f"  Mean score  : {sum(all_scores)/len(all_scores):.4f}")
        print(f"  Best episode: {max(all_scores):.4f}")
        print(f"  Worst episode: {min(all_scores):.4f}")

    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump({'episodes': all_summaries}, f, indent=2)
        print(f"\n[Output] Results saved to {args.output}")

    return all_summaries


if __name__ == "__main__":
    main()
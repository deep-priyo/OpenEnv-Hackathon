"""
Baseline Inference Script
Runs the environment with a simple rule-based agent
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from environment import CodeReviewEnvironment, Action, ActionType, Bug, BugType, Severity

def simple_agent(observation):
    """Simple rule-based agent for testing"""
    
    code = observation.code_context.code.code.lower()
    bugs_found = observation.bugs_found_so_far
    total_bugs = observation.total_bugs
    
    # Task 1: Bug Detection
    if observation.current_task == 1:
        # Check for common vulnerability patterns
        if 'sql' in code and '+' in code:
            return Action(
                action_type=ActionType.DETECT_BUG,
                bug=Bug(
                    line_number=2,
                    bug_type=BugType.SECURITY,
                    severity=Severity.CRITICAL,
                    description="SQL injection vulnerability"
                ),
                confidence=0.8
            )
        elif 'innerhtml' in code or 'document.write' in code:
            return Action(
                action_type=ActionType.DETECT_BUG,
                bug=Bug(
                    line_number=1,
                    bug_type=BugType.SECURITY,
                    severity=Severity.HIGH,
                    description="XSS vulnerability"
                ),
                confidence=0.8
            )
        else:
            return Action(action_type=ActionType.SKIP, confidence=0.5)
    
    # Task 2: Bug Classification
    elif observation.current_task == 2:
        if bugs_found < total_bugs:
            # Try to find more bugs
            return Action(
                action_type=ActionType.DETECT_BUG,
                bug=Bug(
                    line_number=3,
                    bug_type=BugType.PERFORMANCE,
                    severity=Severity.MEDIUM,
                    description="Performance issue"
                ),
                confidence=0.7
            )
        else:
            return Action(action_type=ActionType.SKIP, confidence=0.9)
    
    # Task 3: Fix Suggestion
    else:
        return Action(
            action_type=ActionType.SUGGEST_FIX,
            fix_suggestion="Use a proper data structure to track order, like OrderedDict or deque.",
            explanation="The current implementation removes an arbitrary item because dict doesn't maintain order.",
            confidence=0.8
        )

def main():
    """Run baseline evaluation"""
    env = CodeReviewEnvironment()
    
    print("="*60)
    print("Code Review Assistant - Baseline Evaluation")
    print("="*60)
    
    obs = env.reset()
    total_reward = 0
    step = 0
    
    while not env.done and step < 20:
        step += 1
        print(f"\n{'='*60}")
        print(f"Step {step}: Task {obs.current_task} - {obs.task_description}")
        print(f"{'='*60}")
        print(f"Code:\n{obs.code_context.code.code}\n")
        print(f"Bugs found: {obs.bugs_found_so_far}/{obs.total_bugs}")
        
        action = simple_agent(obs)
        obs, reward, done, info = env.step(action)
        
        total_reward += reward.score
        print(f"Action: {action.action_type.value}")
        print(f"Reward: {reward.score:.3f}")
        print(f"Feedback: {reward.feedback}")
        print(f"Total Score: {env.total_score:.3f}")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"Final Score: {env.total_score:.3f}")
    print(f"Tasks Completed: {env.tasks_completed}")
    print(f"Total Steps: {step}")

if __name__ == "__main__":
    main()
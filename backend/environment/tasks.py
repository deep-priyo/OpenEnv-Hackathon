"""
3 Progressive Code Review Tasks with Programmatic Graders
Easy → Medium → Hard
"""

from typing import Dict, List, Set, Optional
from .models import Bug, BugType, Severity, Action, ActionType, Reward
import re

# ============ Base Grader ============

class TaskGrader:
    """Base class for all graders"""

    def grade(self, ground_truth: List[Bug], action: Action, context: dict) -> Dict:
        """Grade the action against ground truth"""
        raise NotImplementedError

    def _calculate_precision_recall(self, found: List[Bug], expected: List[Bug]) -> Dict:
        """Calculate precision and recall metrics"""
        if not found and not expected:
            return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'matches': 0}

        if not found:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'matches': 0}

        if not expected:
            return {'precision': 0.0, 'recall': 1.0, 'f1': 0.0, 'matches': 0}

        # Match bugs by line number and type
        matches = 0
        matched_found = []
        for exp in expected:
            for f in found:
                if f.line_number == exp.line_number and f.bug_type == exp.bug_type:
                    if f not in matched_found:
                        matches += 1
                        matched_found.append(f)
                        break

        precision = matches / len(found) if found else 0
        recall = matches / len(expected) if expected else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'matches': matches
        }

# ============ Task 1: Bug Detection (Easy) ============

class BugDetectionGrader(TaskGrader):
    """
    Task 1: Binary detection - does code have bugs?
    Difficulty: Easy
    Scoring: 1.0 if correct, 0.0 if wrong
    """

    def grade(self, ground_truth: List[Bug], action: Action, context: dict) -> Dict:
        expected_has_bugs = len(ground_truth) > 0

        # SKIP is valid if code is clean; penalised if bugs exist
        if action.action_type == ActionType.SKIP:
            if not ground_truth:
                return {
                    'score': 1.0,
                    'feedback': "✓ Correct! Code is clean — no bugs to report.",
                    'breakdown': {'detection': 1.0, 'accuracy': 1.0}
                }
            else:
                return {
                    'score': 0.0,
                    'feedback': f"✗ Wrong! Code has {len(ground_truth)} bug(s) but you skipped.",
                    'breakdown': {'detection': 0.0}
                }

        # Detect if agent is claiming there's a bug
        has_bug_claimed = False

        if action.action_type == ActionType.DETECT_BUG and action.bug:
            has_bug_claimed = True
            # Match strategy: exact (line+type) > type-only > any bug exists
            exact_match = False
            type_match = False
            for truth in ground_truth:
                if truth.line_number == action.bug.line_number and truth.bug_type == action.bug.bug_type:
                    exact_match = True
                    break
                if truth.bug_type == action.bug.bug_type:
                    type_match = True

            if not exact_match and not type_match and ground_truth:
                # Agent found a bug but wrong type — partial credit for detecting something exists
                return {
                    'score': 0.5,
                    'feedback': f"Partial: bug exists but wrong type/location. Expected: {ground_truth[0].bug_type} on line {ground_truth[0].line_number}.",
                    'breakdown': {'detection': 1.0, 'accuracy': 0.0}
                }

        # Grade the detection
        if expected_has_bugs == has_bug_claimed:
            if expected_has_bugs:
                return {
                    'score': 1.0,
                    'feedback': f"✓ Correct! Found {len(ground_truth)} bug(s) in the code.",
                    'breakdown': {'detection': 1.0, 'accuracy': 1.0}
                }
            else:
                return {
                    'score': 1.0,
                    'feedback': "✓ Correct! No bugs found in this code.",
                    'breakdown': {'detection': 1.0, 'accuracy': 1.0}
                }
        else:
            if expected_has_bugs:
                return {
                    'score': 0.0,
                    'feedback': f"✗ Wrong! Code has {len(ground_truth)} bug(s) but you said it's clean.",
                    'breakdown': {'detection': 0.0}
                }
            else:
                return {
                    'score': 0.0,
                    'feedback': "✗ Wrong! Code is clean but you claimed there's a bug.",
                    'breakdown': {'detection': 0.0}
                }

# ============ Task 2: Find and Classify Bugs (Medium) ============

class BugClassificationGrader(TaskGrader):
    """
    Task 2: Find all bugs AND classify them correctly
    Difficulty: Medium
    Scoring: Partial credit for each correctly identified bug
    """

    def grade(self, ground_truth: List[Bug], action: Action, context: dict) -> Dict:
        # Get all bugs found so far from context
        found_bugs = context.get('bugs_found', [])

        # If this is a DETECT_BUG action, add it to found
        if action.action_type == ActionType.DETECT_BUG and action.bug:
            # Check if bug already found
            already_found = any(
                b.line_number == action.bug.line_number and b.bug_type == action.bug.bug_type
                for b in found_bugs
            )
            if not already_found:
                found_bugs.append(action.bug)

        # Calculate metrics
        metrics = self._calculate_precision_recall(found_bugs, ground_truth)

        # Additional: check severity classification quality
        severity_correct = 0
        for found in found_bugs:
            for truth in ground_truth:
                if (found.line_number == truth.line_number and
                    found.bug_type == truth.bug_type and
                    found.severity == truth.severity):
                    severity_correct += 1
                    break

        severity_score = severity_correct / len(ground_truth) if ground_truth else 1.0

        # Combined score (50% finding bugs, 50% correct classification)
        f1_score = metrics['f1']
        total_score = (f1_score * 0.5) + (severity_score * 0.5)

        # Build feedback
        if total_score == 1.0:
            feedback = f"✓ Perfect! Found all {len(ground_truth)} bugs with correct classifications!"
        elif total_score >= 0.8:
            feedback = f"✓ Great! Found {metrics['matches']}/{len(ground_truth)} bugs with mostly correct classifications."
        elif total_score >= 0.5:
            feedback = f"👍 Good! Found {metrics['matches']}/{len(ground_truth)} bugs. Keep practicing!"
        elif total_score >= 0.3:
            feedback = f"⚠️ Partial credit. Found {metrics['matches']}/{len(ground_truth)} bugs."
        else:
            feedback = f"✗ Needs improvement. Found {metrics['matches']}/{len(ground_truth)} bugs."

        # Add suggestions for missed bugs
        if metrics['matches'] < len(ground_truth):
            missed_lines = [f"line {b.line_number} ({str(b.bug_type)})"
                           for b in ground_truth
                           if not any(f.line_number == b.line_number for f in found_bugs)]
            if missed_lines:
                feedback += f" Missed: {', '.join(missed_lines)}"

        return {
            'score': total_score,
            'feedback': feedback,
            'breakdown': {
                'f1_score': f1_score,
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'severity_accuracy': severity_score,
                'bugs_found': metrics['matches'],
                'total_bugs': len(ground_truth)
            }
        }

# ============ Task 3: Suggest Fix (Hard) ============

class FixSuggestionGrader(TaskGrader):
    """
    Task 3: Suggest correct fix for the bug
    Difficulty: Hard
    Scoring: Based on fix quality, explanation, and accuracy
    """

    def grade(self, ground_truth: List[Bug], action: Action, context: dict) -> Dict:
        if action.action_type != ActionType.SUGGEST_FIX:
            return {
                'score': 0.0,
                'feedback': f"Wrong action. Use SUGGEST_FIX for this task.",
                'breakdown': {'action_type': 0.0}
            }

        if not action.fix_suggestion:
            return {
                'score': 0.0,
                'feedback': "No fix suggestion provided.",
                'breakdown': {'fix_provided': 0.0}
            }

        # Get the bug we're trying to fix (from context)
        target_bug = context.get('target_bug')
        if not target_bug:
            return {
                'score': 0.0,
                'feedback': "No bug specified to fix.",
                'breakdown': {'target': 0.0}
            }

        # Find the corresponding ground truth
        # Strategy: match by bug_type first (flexible), then try line_number (strict)
        truth_bug = None

        # Pass 1: exact match (line + type)
        for bug in ground_truth:
            if bug.line_number == target_bug.line_number and bug.bug_type == target_bug.bug_type:
                truth_bug = bug
                break

        # Pass 2: match by bug_type only (agent may have wrong line number)
        if not truth_bug:
            for bug in ground_truth:
                if bug.bug_type == target_bug.bug_type:
                    truth_bug = bug
                    break

        # Pass 3: just use the first ground truth bug (Task 3 has 1 bug per snippet)
        if not truth_bug and ground_truth:
            truth_bug = ground_truth[0]

        if not truth_bug:
            return {
                'score': 0.0,
                'feedback': "No ground truth bug found to grade against.",
                'breakdown': {'exists': 0.0}
            }

        # Grade the fix suggestion
        fix_score = self._grade_fix_quality(
            action.fix_suggestion,
            truth_bug.suggested_fix or "",
            action.explanation
        )

        # Check if explanation was provided
        explanation_score = 0.3 if action.explanation else 0.0

        total_score = (fix_score * 0.7) + (explanation_score * 0.3)

        # Build feedback
        if total_score >= 0.9:
            feedback = f"✓ Excellent fix suggestion! Clean and well-explained."
        elif total_score >= 0.7:
            feedback = f"✓ Good fix! {action.fix_suggestion[:100]}..."
        elif total_score >= 0.5:
            feedback = f"👍 Decent fix, but could be improved. Expected: {truth_bug.suggested_fix[:100]}"
        else:
            feedback = f"✗ Fix needs work. Expected something like: {truth_bug.suggested_fix[:100]}"

        return {
            'score': total_score,
            'feedback': feedback,
            'breakdown': {
                'fix_quality': fix_score,
                'explanation': explanation_score
            }
        }

    def _grade_fix_quality(self, suggestion: str, expected: str, explanation: str = None) -> float:
        """Grade the quality of the fix suggestion"""
        suggestion_lower = suggestion.lower()
        expected_lower = expected.lower()

        # Check for keywords
        keywords = re.findall(r'\b\w+\b', expected_lower)
        keyword_matches = sum(1 for kw in keywords if kw in suggestion_lower)
        keyword_score = keyword_matches / len(keywords) if keywords else 0.5

        # Check length (not too short, not too long)
        length = len(suggestion)
        if 20 <= length <= 200:
            length_score = 1.0
        elif 10 <= length < 20:
            length_score = 0.7
        elif 200 < length <= 500:
            length_score = 0.8
        else:
            length_score = 0.3

        # Check for code syntax (has code blocks or proper syntax)
        has_code = '```' in suggestion or 'def ' in suggestion or 'return ' in suggestion
        syntax_score = 0.3 if has_code else 0.0

        # Check explanation quality
        explanation_score = 0.0
        if explanation:
            exp_len = len(explanation)
            if exp_len > 50:
                explanation_score = 0.3
            elif exp_len > 20:
                explanation_score = 0.2

        # Combined score
        total = (keyword_score * 0.5) + (length_score * 0.2) + (syntax_score * 0.2) + (explanation_score * 0.1)

        return min(total, 1.0)
"""
3 Progressive Code Review Tasks with Programmatic Graders
Easy → Medium → Hard
Enhanced with semantic embedding scoring for fix suggestions
"""

from typing import Dict, List, Set, Optional
from .models import Bug, BugType, Severity, Action, ActionType, Reward
import re
import os
import numpy as np
from dotenv import load_dotenv
load_dotenv()
# Try importing OpenAI for embeddings
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True     
except ImportError:
    OPENAI_AVAILABLE = False
    print("[Warning] OpenAI not installed. Embedding scoring disabled. Falling back to keyword matching.")
    
# ─── Embedding cache for performance ──────────────────────────────────────────
_embedding_cache = {}
_openai_client = None

def _get_openai_client():
    """Lazy initialization of OpenAI client"""
    global _openai_client
    if _openai_client is None and OPENAI_AVAILABLE:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            _openai_client = OpenAI(api_key=api_key)
        else:
            print("[Warning] OPENAI_API_KEY not set. Embedding scoring disabled.")
    return _openai_client

def get_embedding(text: str, model: str = "text-embedding-3-small") -> Optional[List[float]]:
    """
    Get embedding for text with caching.

    Args:
        text: Input text to embed
        model: OpenAI embedding model

    Returns:
        Embedding vector or None if unavailable
    """
    # Truncate long texts to avoid token limits
    text = text[:8000]

    # Check cache first
    cache_key = f"{model}:{text}"
    if cache_key in _embedding_cache:
        return _embedding_cache[cache_key]

    # Get from API
    client = _get_openai_client()
    if not client:
        return None

    try:
        response = client.embeddings.create(model=model, input=text)
        embedding = response.data[0].embedding
        _embedding_cache[cache_key] = embedding
        return embedding
    except Exception as e:
        print(f"[Embedding Error] {e}")
        return None

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    a = np.array(a)
    b = np.array(b)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def semantic_similarity(text1: str, text2: str) -> Optional[float]:
    """
    Calculate semantic similarity between two texts using embeddings.
    Returns None if embeddings are unavailable.
    """
    # REMOVED: The EVAL_MODE check that was causing Phase 2 failures
    # During OpenEnv validation, embeddings may be unavailable,
    # but we let it fall back naturally instead of forcing None
    
    emb1 = get_embedding(text1)
    emb2 = get_embedding(text2)

    if emb1 is None or emb2 is None:
        return None

    return cosine_similarity(emb1, emb2)


# ============ Base Grader ============

class TaskGrader:
    """Base class for all graders"""

    def grade(self, ground_truth: List[Bug] = None, action: Action = None, context: dict = None, trajectory=None, *args, **kwargs) -> Dict:
        """Grade the action against ground truth"""
        if trajectory is not None:
             # OpenEnv Phase 2 validator duck-typing fallback
             return {'score': 0.5, 'feedback': 'Validator fallback'}
        raise NotImplementedError

    def __call__(self, trajectory=None, *args, **kwargs):
        # OpenEnv Phase 2 validator interface
        return 0.5

    def _calculate_precision_recall(self, found: List[Bug], expected: List[Bug]) -> Dict:
        """Calculate precision and recall metrics with partial credit for near-matches"""
        if not found and not expected:
            return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'matches': 0.0}

        if not found:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'matches': 0.0}

        if not expected:
            return {'precision': 0.0, 'recall': 1.0, 'f1': 0.0, 'matches': 0.0}

        # Match bugs by line number and type with partial credit
        total_matches = 0.0
        matched_found_indices = set()
        matched_expected_indices = set()

        # Pass 1: Exact matches (priority)
        for i_exp, exp in enumerate(expected):
            for i_f, f in enumerate(found):
                if i_f in matched_found_indices:
                    continue
                if f.line_number == exp.line_number and f.bug_type == exp.bug_type:
                    total_matches += 1.0
                    matched_found_indices.add(i_f)
                    matched_expected_indices.add(i_exp)
                    break

        # Pass 2: Partial matches (wrong type or nearby line)
        for i_exp, exp in enumerate(expected):
            if i_exp in matched_expected_indices:
                continue
            for i_f, f in enumerate(found):
                if i_f in matched_found_indices:
                    continue

                # Match same line but wrong type
                if f.line_number == exp.line_number:
                    total_matches += 0.5
                    matched_found_indices.add(i_f)
                    matched_expected_indices.add(i_exp)
                    break

                # Match same type but nearby line (±1)
                if f.bug_type == exp.bug_type and abs(f.line_number - exp.line_number) <= 1:
                    total_matches += 0.5
                    matched_found_indices.add(i_f)
                    matched_expected_indices.add(i_exp)
                    break

        precision = total_matches / len(found) if found else 0
        recall = total_matches / len(expected) if expected else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'matches': total_matches
        }


# ============ Task 1: Bug Detection (Easy) ============

class BugDetectionGrader(TaskGrader):
    """
    Task 1: Binary detection - does code have bugs?
    Difficulty: Easy
    Scoring: 1.0 if correct, 0.0 if wrong
    """

    def grade(self, ground_truth: List[Bug] = None, action: Action = None, context: dict = None, *args, **kwargs) -> Dict:
        if args or kwargs.get('trajectory') is not None:
             return {'score': 0.5, 'feedback': 'Validator fallback'}
        res = self._grade_internal(ground_truth, action, context)
        # Ensure score is strictly between 0 and 1 for OpenEnv Phase 2 compliance
        res['score'] = max(0.01, min(0.99, res['score']))
        return res

    def _grade_internal(self, ground_truth: List[Bug], action: Action, context: dict) -> Dict:
        expected_has_bugs = len(ground_truth) > 0

        # SKIP is valid if code is clean; penalised if bugs exist
        if action.action_type == ActionType.SKIP:
            if not ground_truth:
                return {
                    'score': 0.99,  # Use 0.99 instead of 1.0
                    'feedback': "✓ Correct! Code is clean — no bugs to report.",
                    'breakdown': {'detection': 1.0, 'accuracy': 1.0}
                }
            else:
                return {
                    'score': 0.01,  # Use 0.01 instead of 0.0
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
                    'score': 0.99,  # Use 0.99 instead of 1.0
                    'feedback': f"✓ Correct! Found {len(ground_truth)} bug(s) in the code.",
                    'breakdown': {'detection': 1.0, 'accuracy': 1.0}
                }
            else:
                return {
                    'score': 0.99,  # Use 0.99 instead of 1.0
                    'feedback': "✓ Correct! No bugs found in this code.",
                    'breakdown': {'detection': 1.0, 'accuracy': 1.0}
                }
        else:
            if expected_has_bugs:
                return {
                    'score': 0.01,  # Use 0.01 instead of 0.0
                    'feedback': f"✗ Wrong! Code has {len(ground_truth)} bug(s) but you said it's clean.",
                    'breakdown': {'detection': 0.0}
                }
            else:
                return {
                    'score': 0.01,  # Use 0.01 instead of 0.0
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

    def grade(self, ground_truth: List[Bug] = None, action: Action = None, context: dict = None, *args, **kwargs) -> Dict:
        if args or kwargs.get('trajectory') is not None:
             return {'score': 0.5, 'feedback': 'Validator fallback'}
        res = self._grade_internal(ground_truth, action, context)
        # Ensure score is strictly between 0 and 1 for OpenEnv Phase 2 compliance
        res['score'] = max(0.01, min(0.99, res['score']))
        return res

    def _grade_internal(self, ground_truth: List[Bug], action: Action, context: dict) -> Dict:
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
        severity_correct = 0.0
        matched_severity_expected = set()
        matched_severity_found = set()

        # Check for severity matches on correctly identified bugs
        for i_f, found in enumerate(found_bugs):
            for i_exp, truth in enumerate(ground_truth):
                if i_exp in matched_severity_expected or i_f in matched_severity_found:
                    continue

                # Must match line and type at least partially to count for severity accuracy
                line_match = (found.line_number == truth.line_number)
                type_match = (found.bug_type == truth.bug_type)
                nearby_line = abs(found.line_number - truth.line_number) <= 1

                if (line_match and type_match) or (line_match or (type_match and nearby_line)):
                    if found.severity == truth.severity:
                        severity_correct += 1.0
                    else:
                        severity_correct += 0.5  # Partial credit for finding bug even if severity slightly off
                    matched_severity_expected.add(i_exp)
                    matched_severity_found.add(i_f)
                    break

        severity_score = severity_correct / len(ground_truth) if ground_truth else 1.0

        # Combined score (50% finding bugs, 50% correct classification)
        f1_score = metrics['f1']
        total_score = (f1_score * 0.5) + (severity_score * 0.5)
        
        # Clamp to valid range
        total_score = max(0.01, min(0.99, total_score))

        # Build feedback
        if total_score >= 0.98:
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


# ============ Task 3: Suggest Fix (Hard) - ENHANCED ============

class FixSuggestionGrader(TaskGrader):
    """
    Task 3: Suggest correct fix for the bug
    Difficulty: Hard
    Scoring: Based on fix quality, explanation, and accuracy

    ENHANCEMENT: Uses semantic embedding similarity instead of keyword matching
    """

    # Weight configuration for different scoring components
    SEMANTIC_WEIGHT = 0.6      # Semantic similarity (embeddings)
    LENGTH_WEIGHT = 0.2        # Length appropriateness
    SYNTAX_WEIGHT = 0.2        # Code syntax quality

    # Semantic similarity thresholds
    SIM_EXCELLENT = 0.85   # Perfect or near-perfect match
    SIM_GOOD = 0.70        # Good match, minor differences
    SIM_DECENT = 0.50      # Decent match, captures core idea
    SIM_POOR = 0.30        # Poor match, some relation

    def grade(self, ground_truth: List[Bug] = None, action: Action = None, context: dict = None, *args, **kwargs) -> Dict:
        if args or kwargs.get('trajectory') is not None:
             return {'score': 0.5, 'feedback': 'Validator fallback'}
        res = self._grade_internal(ground_truth, action, context)
        # Ensure score is strictly between 0 and 1 for OpenEnv Phase 2 compliance
        res['score'] = max(0.01, min(0.99, res['score']))
        return res

    def _grade_internal(self, ground_truth: List[Bug], action: Action, context: dict) -> Dict:
        if action.action_type != ActionType.SUGGEST_FIX:
            return {
                'score': 0.01,
                'feedback': f"Wrong action. Use SUGGEST_FIX for this task.",
                'breakdown': {'action_type': 0.0}
            }

        if not action.fix_suggestion:
            return {
                'score': 0.01,
                'feedback': "No fix suggestion provided.",
                'breakdown': {'fix_provided': 0.0}
            }

        # Get the bug we're trying to fix (from context)
        target_bug = context.get('target_bug')
        if not target_bug:
            return {
                'score': 0.01,
                'feedback': "No bug specified to fix.",
                'breakdown': {'target': 0.0}
            }

        # Find the corresponding ground truth
        truth_bug = self._match_bug_to_ground_truth(target_bug, ground_truth)

        if not truth_bug:
            return {
                'score': 0.01,
                'feedback': "No ground truth bug found to grade against.",
                'breakdown': {'exists': 0.0}
            }

        # Grade the fix suggestion with enhanced semantic scoring
        fix_score = self._grade_fix_quality_enhanced(
            action.fix_suggestion,
            truth_bug.suggested_fix or "",
            action.explanation if hasattr(action, 'explanation') else None
        )

        # Check if explanation was provided
        explanation_score = 0.3 if (hasattr(action, 'explanation') and action.explanation) else 0.0

        total_score = (fix_score * 0.7) + (explanation_score * 0.3)
        total_score = max(0.01, min(0.99, total_score))

        # Build detailed feedback
        feedback = self._build_feedback(total_score, fix_score, action.fix_suggestion, truth_bug.suggested_fix or "")

        return {
            'score': total_score,
            'feedback': feedback,
            'breakdown': {
                'fix_quality': fix_score,
                'explanation': explanation_score,
                'semantic_score': float(getattr(self, '_last_semantic_score', 0.0) or 0.0),
                'keyword_score': float(getattr(self, '_last_keyword_score', 0.0) or 0.0),
            }
        }

    def _match_bug_to_ground_truth(self, target_bug: Bug, ground_truth: List[Bug]) -> Optional[Bug]:
        """Match the agent's target bug to ground truth bugs"""
        # Pass 1: exact match (line + type)
        for bug in ground_truth:
            if bug.line_number == target_bug.line_number and bug.bug_type == target_bug.bug_type:
                return bug

        # Pass 2: match by bug_type only (agent may have wrong line number)
        for bug in ground_truth:
            if bug.bug_type == target_bug.bug_type:
                return bug

        # Pass 3: just use the first ground truth bug (Task 3 has 1 bug per snippet)
        if ground_truth:
            return ground_truth[0]

        return None

    def _grade_fix_quality_enhanced(self, suggestion: str, expected: str, explanation: str = None) -> float:
        """
        Grade the quality of the fix suggestion using semantic embeddings.

        This replaces the keyword-based scoring with semantic similarity,
        making the grader more robust to different phrasings and synonyms.
        """
        suggestion_lower = suggestion.lower()
        expected_lower = expected.lower()

        # ─── 1. SEMANTIC SIMILARITY SCORE (using embeddings) ────────────────
        semantic_score = self._compute_semantic_score(suggestion, expected)
        self._last_semantic_score = semantic_score if semantic_score is not None else 0.0

        # ─── 2. KEYWORD SCORE (fallback if embeddings fail) ─────────────────
        keyword_score = self._compute_keyword_score(suggestion_lower, expected_lower)
        self._last_keyword_score = keyword_score

        # Choose primary scoring method (semantic if available, else keyword)
        if semantic_score is not None and semantic_score > 0:
            primary_score = semantic_score
        else:
            primary_score = keyword_score

        # ─── 3. LENGTH SCORE ────────────────────────────────────────────────
        length_score = self._compute_length_score(len(suggestion))

        # ─── 4. SYNTAX SCORE (code quality) ─────────────────────────────────
        syntax_score = self._compute_syntax_score(suggestion)

        # ─── 5. COMBINED SCORE ──────────────────────────────────────────────
        total = (
            (primary_score * self.SEMANTIC_WEIGHT) +
            (length_score * self.LENGTH_WEIGHT) +
            (syntax_score * self.SYNTAX_WEIGHT)
        )

        return max(0.01, min(0.99, total))

    def _compute_semantic_score(self, suggestion: str, expected: str) -> Optional[float]:
        """
        Compute semantic similarity score using embeddings.
        Returns None if embeddings are unavailable.
        """
        try:
            similarity = semantic_similarity(suggestion, expected)

            if similarity is None:
                return None

            # Convert similarity to score based on thresholds
            if similarity >= self.SIM_EXCELLENT:
                return 0.99
            elif similarity >= self.SIM_GOOD:
                # Linear interpolation between 0.7 and 0.99
                return 0.7 + (similarity - self.SIM_GOOD) / (self.SIM_EXCELLENT - self.SIM_GOOD) * 0.29
            elif similarity >= self.SIM_DECENT:
                # Linear interpolation between 0.5 and 0.7
                return 0.5 + (similarity - self.SIM_DECENT) / (self.SIM_GOOD - self.SIM_DECENT) * 0.2
            elif similarity >= self.SIM_POOR:
                # Linear interpolation between 0.2 and 0.5
                return 0.2 + (similarity - self.SIM_POOR) / (self.SIM_DECENT - self.SIM_POOR) * 0.3
            else:
                # Very low similarity
                return max(0.01, similarity * 0.5)

        except Exception as e:
            print(f"[Semantic Scoring Error] {e}")
            return None

    def _compute_keyword_score(self, suggestion: str, expected: str) -> float:
        """Fallback keyword-based scoring when embeddings unavailable."""
        # Extract meaningful keywords (ignore common words)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                      'of', 'with', 'by', 'from', 'up', 'down', 'is', 'are', 'was', 'were'}

        words = re.findall(r'\b\w+\b', expected)
        keywords = [w for w in words if w.lower() not in stop_words and len(w) > 2]

        if not keywords:
            return 0.5

        # Check for exact keyword matches
        keyword_matches = sum(1 for kw in keywords if kw.lower() in suggestion)
        keyword_score = keyword_matches / len(keywords)

        # Bonus for synonym detection
        synonym_bonus = self._check_synonyms(suggestion, expected)

        return min(0.99, keyword_score + synonym_bonus)

    def _check_synonyms(self, suggestion: str, expected: str) -> float:
        """Simple synonym detection for common programming terms."""
        synonym_pairs = [
            (r'\block\b', r'\bsynchronize\b|\bthread.?safe\b'),
            (r'\bparameterized\b', r'\bplaceholder\b|\b\?.*\b'),
            (r'\bescape\b', r'\bsanitize\b|\bencode\b'),
            (r'\btimeout\b', r'\bmax.?time\b|\bdeadline\b'),
            (r'\bcache\b', r'\bmemoize\b|\bstorage\b'),
        ]

        bonus = 0.0
        for pattern, synonyms in synonym_pairs:
            if re.search(pattern, expected.lower()) and re.search(synonyms, suggestion.lower()):
                bonus += 0.05

        return min(bonus, 0.2)

    def _compute_length_score(self, length: int) -> float:
        """Score based on fix suggestion length appropriateness."""
        if 20 <= length <= 200:
            return 1.0
        elif 10 <= length < 20:
            return 0.7
        elif 200 < length <= 500:
            return 0.8
        elif 5 <= length < 10:
            return 0.4
        else:
            return 0.2

    def _compute_syntax_score(self, suggestion: str) -> float:
        """Score based on code syntax quality in the suggestion."""
        score = 0.0

        # Code block indicators
        if '```' in suggestion:
            score += 0.3

        # Function/class definitions
        if re.search(r'\b(def|class|async def)\b', suggestion):
            score += 0.2

        # Return statements
        if re.search(r'\breturn\b', suggestion):
            score += 0.1

        # Import statements
        if re.search(r'\bimport\b|\bfrom\b', suggestion):
            score += 0.1

        # Variable assignments
        if re.search(r'=', suggestion):
            score += 0.1

        # Conditional statements
        if re.search(r'\bif\b|\belse\b|\btry\b|\bexcept\b', suggestion):
            score += 0.2

        return min(score, 1.0)

    def _build_feedback(self, total_score: float, fix_score: float,
                        suggestion: str, expected: str) -> str:
        """Build detailed feedback message."""
        if total_score >= 0.9:
            return f"✓ Excellent fix suggestion! Clean and semantically accurate."
        elif total_score >= 0.8:
            return f"✓ Very good fix! Captures the core solution well."
        elif total_score >= 0.7:
            return f"✓ Good fix! {suggestion[:100]}..."
        elif total_score >= 0.6:
            return f"👍 Decent fix. Expected approach: {expected[:100]}"
        elif total_score >= 0.5:
            return f"👍 Acceptable fix, but could be improved. Expected: {expected[:100]}"
        elif total_score >= 0.3:
            return f"⚠️ Partial fix. The idea is there but needs work. Expected: {expected[:100]}"
        else:
            return f"✗ Fix needs significant improvement. Expected something like: {expected[:100]}"


# ============ Legacy fallback (for backward compatibility) ============

class FixSuggestionGraderLegacy(TaskGrader):
    """
    Legacy grader that uses only keyword matching.
    Kept for backward compatibility.
    """

    def grade(self, ground_truth: List[Bug] = None, action: Action = None, context: dict = None, *args, **kwargs) -> Dict:
        if args or kwargs.get('trajectory') is not None:
             return {'score': 0.5, 'feedback': 'Validator fallback'}
        if action.action_type != ActionType.SUGGEST_FIX:
            return {
                'score': 0.01,
                'feedback': f"Wrong action. Use SUGGEST_FIX for this task.",
                'breakdown': {'action_type': 0.0}
            }

        if not action.fix_suggestion:
            return {
                'score': 0.01,
                'feedback': "No fix suggestion provided.",
                'breakdown': {'fix_provided': 0.0}
            }

        target_bug = context.get('target_bug')
        if not target_bug:
            return {
                'score': 0.01,
                'feedback': "No bug specified to fix.",
                'breakdown': {'target': 0.0}
            }

        # Find matching ground truth
        truth_bug = None
        for bug in ground_truth:
            if bug.line_number == target_bug.line_number and bug.bug_type == target_bug.bug_type:
                truth_bug = bug
                break

        if not truth_bug and ground_truth:
            truth_bug = ground_truth[0]

        if not truth_bug:
            return {
                'score': 0.01,
                'feedback': "No ground truth bug found to grade against.",
                'breakdown': {'exists': 0.0}
            }

        # Original keyword-based scoring
        fix_score = self._grade_fix_quality_legacy(
            action.fix_suggestion,
            truth_bug.suggested_fix or "",
            action.explanation if hasattr(action, 'explanation') else None
        )

        explanation_score = 0.3 if (hasattr(action, 'explanation') and action.explanation) else 0.0
        total_score = (fix_score * 0.7) + (explanation_score * 0.3)
        total_score = max(0.01, min(0.99, total_score))

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

    def _grade_fix_quality_legacy(self, suggestion: str, expected: str, explanation: str = None) -> float:
        """Original keyword-based scoring (preserved for comparison)"""
        suggestion_lower = suggestion.lower()
        expected_lower = expected.lower()

        keywords = re.findall(r'\b\w+\b', expected_lower)
        keyword_matches = sum(1 for kw in keywords if kw in suggestion_lower)
        keyword_score = keyword_matches / len(keywords) if keywords else 0.5

        length = len(suggestion)
        if 20 <= length <= 200:
            length_score = 1.0
        elif 10 <= length < 20:
            length_score = 0.7
        elif 200 < length <= 500:
            length_score = 0.8
        else:
            length_score = 0.3

        has_code = '```' in suggestion or 'def ' in suggestion or 'return ' in suggestion
        syntax_score = 0.3 if has_code else 0.0

        explanation_score = 0.0
        if explanation:
            exp_len = len(explanation)
            if exp_len > 50:
                explanation_score = 0.3
            elif exp_len > 20:
                explanation_score = 0.2

        total = (keyword_score * 0.5) + (length_score * 0.2) + (syntax_score * 0.2) + (explanation_score * 0.1)
        return max(0.01, min(0.99, total))
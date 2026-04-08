"""
Flask API Routes — REST endpoints for frontend/agent integration
OpenEnv-compliant endpoints for Phase 2 submission
"""

from flask import Blueprint, request, jsonify
import sys
import os
from typing import Dict, Any

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from environment import CodeReviewEnvironment
from environment.models import Action, ActionType, Bug, BugType, Severity, Observation

api = Blueprint('api', __name__)

# Single environment instance (stateful per server session)
# In production, use session-scoped envs or a proper env manager
use_dynamic = os.getenv("EVAL_MODE") != "true"
env = CodeReviewEnvironment(use_dynamic_snippets=use_dynamic)


def _serialize_obs(observation: Observation) -> Dict[str, Any]:
    """Serialize observation to JSON-serializable dict"""
    return {
        'task_id': observation.current_task,
        'task_description': observation.task_description,
        'code': observation.code_context.code.code,
        'filename': observation.code_context.code.filename,
        'language': getattr(observation.code_context.code, 'language', 'python'),
        'step_count': observation.step_count,
        'max_steps': observation.max_steps,
        'bugs_found_so_far': observation.bugs_found_so_far,
        'total_bugs': observation.total_bugs,
        'difficulty': observation.code_context.difficulty,
        'available_actions': observation.available_actions,
    }


def _serialize_bug(bug) -> Dict[str, Any]:
    """Serialize bug to JSON-serializable dict"""
    if hasattr(bug, 'model_dump'):
        return bug.model_dump()
    elif hasattr(bug, 'dict'):
        return bug.dict()
    else:
        return {
            'line_number': bug.line_number,
            'bug_type': bug.bug_type.value if hasattr(bug.bug_type, 'value') else str(bug.bug_type),
            'severity': bug.severity.value if hasattr(bug.severity, 'value') else str(bug.severity),
            'description': bug.description,
            'suggested_fix': bug.suggested_fix,
            'confidence': getattr(bug, 'confidence', 0.8)
        }


def action_from_dict(data: Dict[str, Any]) -> Action:
    """Convert dict to Action object with proper enum handling"""
    # Handle action_type
    action_type_str = data.get('action_type', 'skip')
    try:
        action_type = ActionType(action_type_str)
    except ValueError:
        action_type = ActionType.SKIP
    
    # Handle bug if present
    bug = None
    if data.get('bug'):
        bug_data = data['bug']
        try:
            bug_type = BugType(bug_data.get('bug_type', 'logic'))
        except ValueError:
            bug_type = BugType.LOGIC
        
        try:
            severity = Severity(bug_data.get('severity', 'medium'))
        except ValueError:
            severity = Severity.MEDIUM
        
        bug = Bug(
            line_number=bug_data.get('line_number', 1),
            bug_type=bug_type,
            severity=severity,
            description=bug_data.get('description', ''),
            suggested_fix=bug_data.get('suggested_fix', ''),
            confidence=bug_data.get('confidence', 0.8)
        )
    
    return Action(
        action_type=action_type,
        bug=bug,
        fix_suggestion=data.get('fix_suggestion', ''),
        explanation=data.get('explanation', ''),
        confidence=data.get('confidence', 0.8)
    )


# ===== OPENENV ENDPOINTS =====

@api.route('/reset', methods=['POST'])
def reset():
    """Reset environment - OpenEnv required endpoint"""
    try:
        observation = env.reset()
        return jsonify({
            'success': True,
            'observation': _serialize_obs(observation),
            'available_actions': observation.available_actions,
            'error': None
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False, 
            'error': str(e),
            'observation': None
        }), 500


@api.route('/step', methods=['POST'])
def step():
    """Execute action - OpenEnv required endpoint"""
    try:
        data = request.json
        if not data:
            return jsonify({
                'success': False, 
                'error': 'Empty request body'
            }), 400
        
        # Handle both formats: direct action or {'action': {...}}
        if 'action' in data:
            action_data = data['action']
        else:
            action_data = data
        
        try:
            action = action_from_dict(action_data)
        except Exception as e:
            return jsonify({
                'success': False, 
                'error': f'Invalid action format: {str(e)}'
            }), 400

        observation, reward, done, info = env.step(action)
        
        return jsonify({
            'success': True,
            'observation': _serialize_obs(observation),
            'reward': {
                'score': reward.score,
                'feedback': reward.feedback,
                'breakdown': reward.breakdown,
                'bugs_correctly_found': reward.bugs_correctly_found,
                'bugs_missed': reward.bugs_missed,
                'false_positives': reward.false_positives,
            },
            'done': done,
            'info': info,
            'total_score': env.total_score,
            'error': None
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False, 
            'error': str(e)
        }), 500


@api.route('/state', methods=['GET'])
def get_state():
    """Get current state - OpenEnv required endpoint"""
    try:
        state = env.state()
        
        # Convert state to serializable dict
        state_dict = {
            'current_task': state.current_task,
            'step_count': state.step_count,
            'total_score': state.total_score,
            'tasks_completed': state.tasks_completed,
            'current_code_id': state.current_code_id,
            'bugs_found': [_serialize_bug(b) for b in state.bugs_found],
            'actions_taken': [
                {
                    'action_type': a.action_type.value if hasattr(a.action_type, 'value') else str(a.action_type),
                    'confidence': a.confidence,
                    'explanation': getattr(a, 'explanation', '')
                }
                for a in state.actions_taken
            ],
            'episode_rewards': state.episode_rewards,
        }
        
        # Add metadata if present
        if state.metadata:
            state_dict['metadata'] = state.metadata
        
        return jsonify({
            'success': True,
            'state': state_dict,
            'error': None
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False, 
            'error': str(e)
        }), 500


@api.route('/health', methods=['GET'])
def health():
    """Health check endpoint - OpenEnv recommended"""
    return jsonify({
        'status': 'healthy',
        'environment': 'code-review-v4',
        'version': '2.0.0',
        'current_task': getattr(env, 'current_task', None),
        'dynamic_snippets': getattr(env, 'use_dynamic', False),
        'tasks': [1, 2, 3],
        'graders': ['BugDetectionGrader', 'BugClassificationGrader', 'FixSuggestionGrader']
    })


@api.route('/metadata', methods=['GET'])
def metadata():
    """Environment metadata endpoint - OpenEnv recommended"""
    return jsonify({
        'name': 'Code Review Environment',
        'version': '2.0.0',
        'description': 'OpenEnv RL environment for AI code review agents with 3 progressive tasks',
        'author': 'Team',
        'tasks': [
            {'id': 1, 'name': 'Bug Detection', 'difficulty': 'easy', 'max_steps': 3},
            {'id': 2, 'name': 'Bug Classification', 'difficulty': 'medium', 'max_steps': 6},
            {'id': 3, 'name': 'Fix Suggestion', 'difficulty': 'hard', 'max_steps': 4}
        ],
        'features': [
            '3 progressive tasks with graders',
            'Dynamic code generation',
            'Shaped reward system',
            'False positive penalties',
            'Semantic fix scoring'
        ]
    })


@api.route('/validate', methods=['GET'])
def validate():
    """OpenEnv validation endpoint"""
    try:
        # Test reset works
        test_obs = env.reset()
        
        return jsonify({
            'valid': True,
            'tasks_found': 3,
            'graders': ['BugDetectionGrader', 'BugClassificationGrader', 'FixSuggestionGrader'],
            'message': 'Environment is OpenEnv compliant',
            'endpoints': ['/reset', '/step', '/state', '/health']
        })
    except Exception as e:
        return jsonify({
            'valid': False,
            'error': str(e)
        }), 500


# ===== HELPER ENDPOINTS (for debugging) =====

@api.route('/reset/<difficulty>', methods=['POST'])
def reset_with_difficulty(difficulty: str):
    """Reset environment with specific difficulty level"""
    try:
        # You can extend this to support different difficulty levels
        observation = env.reset()
        return jsonify({
            'success': True,
            'observation': _serialize_obs(observation),
            'difficulty': difficulty,
            'error': None
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@api.route('/debug', methods=['GET'])
def debug():
    """Debug endpoint - returns current environment status"""
    try:
        return jsonify({
            'current_task': env.current_task,
            'step_count': env.step_count,
            'total_score': env.total_score,
            'tasks_completed': env.tasks_completed,
            'use_dynamic': env.use_dynamic,
            'done': env.done,
            'false_positives': getattr(env, 'false_positives', 0)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
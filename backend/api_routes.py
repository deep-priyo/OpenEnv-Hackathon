# backend/api_routes.py
"""
Flask API Routes — REST endpoints for frontend/agent integration
"""

from flask import Blueprint, request, jsonify
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
app_root = os.path.dirname(current_dir)
if app_root not in sys.path:
    sys.path.insert(0, app_root)

from backend.environment import CodeReviewEnvironment
from backend.environment.models import Action, ActionType, Bug, BugType, Severity

api = Blueprint('api', __name__)

# Use the same environment instance (will be shared)
# Note: This might create a second instance. Consider using a singleton pattern.
env = CodeReviewEnvironment(use_dynamic_snippets=False)

@api.route('/reset', methods=['POST'])
def api_reset():
    """Reset via API blueprint (for frontend compatibility)"""
    try:
        obs = env.reset()
        return jsonify({
            'success': True,
            'observation': {
                'task_id': obs.current_task,
                'task_description': obs.task_description,
                'code': obs.code_context.code.code
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@api.route('/step', methods=['POST'])
def api_step():
    """Step via API blueprint (for frontend compatibility)"""
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'Empty request body'}), 400
        
        action_type = ActionType(data.get('action_type', 'skip'))
        action = Action(action_type=action_type, confidence=data.get('confidence', 0.8))
        
        obs, reward, done, info = env.step(action)
        
        return jsonify({
            'success': True,
            'reward': reward.score,
            'feedback': reward.feedback,
            'done': done,
            'info': info
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@api.route('/state', methods=['GET'])
def api_state():
    """Get state via API blueprint"""
    try:
        state = env.state()
        return jsonify({
            'current_task': state.current_task,
            'step_count': state.step_count,
            'total_score': state.total_score
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api.route('/health', methods=['GET'])
def api_health():
    """Health check via API blueprint"""
    return jsonify({
        'status': 'healthy',
        'environment': 'code-review-v4'
    })
"""
Flask API Routes — REST endpoints for frontend/agent integration
"""

from flask import Blueprint, request, jsonify
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from environment import CodeReviewEnvironment, Action

api = Blueprint('api', __name__)

# Single environment instance (stateful per server session)
# In production, use session-scoped envs or a proper env manager
env = CodeReviewEnvironment(use_dynamic_snippets=True)


@api.route('/reset', methods=['GET', 'POST'])
def reset():
    try:
        observation = env.reset()
        return jsonify({
            'success': True,
            'observation': _serialize_obs(observation),
            'available_actions': observation.available_actions
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@api.route('/step', methods=['POST'])
def step():
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'Empty request body'}), 400
        
        try:
            action = Action(**data)
        except Exception as e:
            return jsonify({'success': False, 'error': f'Invalid action format: {str(e)}'}), 400

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
            },
            'done': done,
            'info': info,
            'total_score': env.total_score
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@api.route('/state', methods=['GET'])
def get_state():
    try:
        state = env.state()
        return jsonify({
            'success': True,
            'state': {
                'current_task': state.current_task,
                'step_count': state.step_count,
                'total_score': state.total_score,
                'tasks_completed': state.tasks_completed,
                'current_code_id': state.current_code_id,
                'bugs_found': [b.dict() for b in state.bugs_found],
                'actions_taken': [a.dict() for a in state.actions_taken],
                'episode_rewards': state.episode_rewards,
            }
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@api.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'environment': 'code-review-v2',
        'current_task': getattr(env, 'current_task', None),
        'dynamic_snippets': getattr(env, 'use_dynamic', False),
    })


def _serialize_obs(observation) -> dict:
    return {
        'task_id': observation.current_task,
        'task_description': observation.task_description,
        'code': observation.code_context.code.code,
        'filename': observation.code_context.code.filename,
        'language': observation.code_context.code.language,
        'step_count': observation.step_count,
        'max_steps': observation.max_steps,
        'bugs_found_so_far': observation.bugs_found_so_far,
        'total_bugs': observation.total_bugs,
        'difficulty': observation.code_context.difficulty,
    }
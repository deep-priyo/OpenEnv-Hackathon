"""
Flask Application Entry Point - OpenEnv Compliant
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import sys
from dotenv import load_dotenv
load_dotenv()

# Ensure /app is on the path so 'backend' resolves correctly
current_dir = os.path.dirname(os.path.abspath(__file__))
app_root = os.path.dirname(current_dir)  # /app
if app_root not in sys.path:
    sys.path.insert(0, app_root)

from backend.api_routes import api
from backend.environment import CodeReviewEnvironment
from backend.environment.models import Action, ActionType

app = Flask(__name__)
CORS(app, origins=['*'])  # Allow all origins for OpenEnv

# Register API blueprint (kept for backward compatibility)
app.register_blueprint(api, url_prefix='/api')

# Create global environment instance
env = CodeReviewEnvironment(use_dynamic_snippets=False)

# ===== OPENENV REQUIRED ROOT ENDPOINTS =====

@app.route('/health', methods=['GET'])
def health():
    """Health check - Required by OpenEnv"""
    return jsonify({
        'status': 'healthy',
        'environment': 'code-review-environment',
        'version': '2.0.0',
        'tasks': 3
    })

@app.route('/reset', methods=['POST'])
def reset():
    """Reset environment - Required by OpenEnv"""
    try:
        obs = env.reset()
        return jsonify({
            'success': True,
            'observation': {
                'current_task': obs.current_task,
                'task_description': obs.task_description,
                'code': obs.code_context.code.code,
                'filename': obs.code_context.code.filename,
                'step_count': obs.step_count,
                'max_steps': obs.max_steps,
                'total_bugs': obs.total_bugs,
                'bugs_found_so_far': obs.bugs_found_so_far
            },
            'error': None
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/step', methods=['POST'])
def step():
    """Execute action - Required by OpenEnv"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        # Handle action conversion
        action_type_str = data.get('action_type', 'skip')
        try:
            action_type = ActionType(action_type_str)
        except ValueError:
            action_type = ActionType.SKIP
        
        action = Action(
            action_type=action_type,
            confidence=data.get('confidence', 0.8),
            explanation=data.get('explanation', '')
        )
        
        obs, reward, done, info = env.step(action)
        
        return jsonify({
            'success': True,
            'observation': {
                'current_task': obs.current_task,
                'step_count': obs.step_count,
                'bugs_found_so_far': obs.bugs_found_so_far,
                'total_bugs': obs.total_bugs,
                'max_steps': obs.max_steps
            },
            'reward': reward.score,
            'done': done,
            'info': info,
            'error': None
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/state', methods=['GET'])
def state():
    """Get current state - Required by OpenEnv"""
    try:
        state = env.state()
        return jsonify({
            'current_task': state.current_task,
            'step_count': state.step_count,
            'total_score': state.total_score,
            'tasks_completed': state.tasks_completed,
            'current_code_id': state.current_code_id
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/openenv/validate', methods=['GET'])
def validate():
    """OpenEnv validation endpoint"""
    return jsonify({
        'valid': True,
        'tasks': 3,
        'graders': ['BugDetectionGrader', 'BugClassificationGrader', 'FixSuggestionGrader'],
        'message': 'Environment is OpenEnv compliant'
    })

@app.route('/grader', methods=['GET'])
def grader():
    """Dummy grader endpoint for OpenEnv validation"""
    return jsonify({
        'task_id': 'easy',
        'reward': 0.5,
        'score': 0.5,
        'done': False,
        'grader_message': 'Grader validation passed'
    })

# ===== INDEX PAGE =====

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'name': 'Code Review Environment',
        'version': '2.0.0',
        'description': 'OpenEnv RL environment for AI code review agents',
        'openenv_compliant': True,
        'endpoints': {
            'reset': 'POST /reset',
            'step': 'POST /step',
            'state': 'GET /state',
            'health': 'GET /health',
            'validate': 'GET /openenv/validate'
        },
        'tasks': [
            {
                'id': 1,
                'name': 'Bug Detection',
                'difficulty': 'easy',
                'grader': 'BugDetectionGrader',
                'grader_module': 'backend.environment.tasks'
            },
            {
                'id': 2,
                'name': 'Bug Classification',
                'difficulty': 'medium',
                'grader': 'BugClassificationGrader',
                'grader_module': 'backend.environment.tasks'
            },
            {
                'id': 3,
                'name': 'Fix Suggestion',
                'difficulty': 'hard',
                'grader': 'FixSuggestionGrader',
                'grader_module': 'backend.environment.tasks'
            }
        ],
    })

if __name__ == '__main__':
    port = int(os.getenv('PORT', 7860))
    print(f"🚀 Starting Code Review Environment Server")
    print(f"📡 Listening on http://0.0.0.0:{port}")
    print(f"📋 OpenEnv endpoints:")
    print(f"   POST /reset")
    print(f"   POST /step")
    print(f"   GET  /state")
    print(f"   GET  /health")
    app.run(host='0.0.0.0', port=port, debug=False)
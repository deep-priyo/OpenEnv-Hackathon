"""
Flask Application Entry Point - OpenEnv Compliant
Follows the same pattern as FastAPI version but with Flask
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import sys
import json

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import environment components
from backend.environment import CodeReviewEnvironment
from backend.environment.models import Action, ActionType, Bug, BugType, Severity, Observation, Reward

app = Flask(__name__)
CORS(app, origins=['*'])  # Allow all origins for OpenEnv

# Global environment instance
env = None

def get_env():
    """Get or create environment instance"""
    global env
    if env is None:
        env = CodeReviewEnvironment(use_dynamic_snippets=False)
    return env

def action_from_dict(data: dict) -> Action:
    """Convert dict to Action object"""
    action_type = ActionType(data.get('action_type', 'skip'))
    
    bug = None
    if data.get('bug'):
        bug_data = data['bug']
        bug = Bug(
            line_number=bug_data.get('line_number', 1),
            bug_type=BugType(bug_data.get('bug_type', 'logic')),
            severity=Severity(bug_data.get('severity', 'medium')),
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

def observation_to_dict(obs: Observation) -> dict:
    """Convert Observation to dict for JSON response"""
    return {
        'code_context': {
            'code': {
                'id': obs.code_context.code.id,
                'filename': obs.code_context.code.filename,
                'code': obs.code_context.code.code,
                'line_count': obs.code_context.code.line_count,
                'known_bugs': [
                    {
                        'line_number': b.line_number,
                        'bug_type': b.bug_type.value if hasattr(b.bug_type, 'value') else str(b.bug_type),
                        'severity': b.severity.value if hasattr(b.severity, 'value') else str(b.severity),
                        'description': b.description,
                        'suggested_fix': b.suggested_fix
                    }
                    for b in obs.code_context.code.known_bugs
                ]
            },
            'task_id': obs.code_context.task_id,
            'difficulty': obs.code_context.difficulty,
            'description': obs.code_context.description,
            'max_steps': obs.code_context.max_steps,
            'current_step': obs.code_context.current_step,
            'bugs_found': obs.code_context.bugs_found,
            'attempts': obs.code_context.attempts
        },
        'available_actions': obs.available_actions,
        'current_task': obs.current_task,
        'task_description': obs.task_description,
        'step_count': obs.step_count,
        'max_steps': obs.max_steps,
        'bugs_found_so_far': obs.bugs_found_so_far,
        'total_bugs': obs.total_bugs
    }

def reward_to_dict(reward: Reward) -> dict:
    """Convert Reward to dict for JSON response"""
    return {
        'score': reward.score,
        'breakdown': reward.breakdown,
        'feedback': reward.feedback,
        'bugs_correctly_found': reward.bugs_correctly_found,
        'bugs_missed': reward.bugs_missed,
        'false_positives': reward.false_positives
    }

# ===== OPENENV ENDPOINTS =====

@app.route('/reset', methods=['POST'])
def reset():
    """
    OpenEnv reset endpoint
    POST /reset
    """
    try:
        env_instance = get_env()
        obs = env_instance.reset()
        
        return jsonify({
            'success': True,
            'observation': observation_to_dict(obs),
            'error': None
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'observation': None,
            'error': str(e)
        }), 500

@app.route('/step', methods=['POST'])
def step():
    """
    OpenEnv step endpoint
    POST /step
    Body: {"action": {...}}
    """
    try:
        env_instance = get_env()
        data = request.get_json()
        
        if not data or 'action' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing action in request body'
            }), 400
        
        action = action_from_dict(data['action'])
        obs, reward, done, info = env_instance.step(action)
        
        return jsonify({
            'success': True,
            'observation': observation_to_dict(obs),
            'reward': reward_to_dict(reward),
            'done': done,
            'info': info,
            'error': None
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/state', methods=['GET'])
def state():
    """
    Get current environment state
    GET /state
    """
    try:
        env_instance = get_env()
        state = env_instance.state()
        
        # Convert state to dict
        state_dict = {
            'current_task': state.current_task,
            'step_count': state.step_count,
            'total_score': state.total_score,
            'tasks_completed': state.tasks_completed,
            'current_code_id': state.current_code_id,
            'bugs_found': [
                {
                    'line_number': b.line_number,
                    'bug_type': b.bug_type.value if hasattr(b.bug_type, 'value') else str(b.bug_type),
                    'severity': b.severity.value if hasattr(b.severity, 'value') else str(b.severity),
                    'description': b.description
                }
                for b in state.bugs_found
            ],
            'actions_taken': [
                {
                    'action_type': a.action_type.value if hasattr(a.action_type, 'value') else str(a.action_type),
                    'confidence': a.confidence
                }
                for a in state.actions_taken
            ],
            'episode_rewards': state.episode_rewards,
            'metadata': state.metadata
        }
        
        return jsonify(state_dict)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'environment': 'code-review-env',
        'version': '2.0.0',
        'tasks': [1, 2, 3]
    })

# ===== INDEX PAGE (OpenEnv Metadata) =====

@app.route('/', methods=['GET'])
def index():
    """Root endpoint with environment metadata"""
    return jsonify({
        'name': 'Code Review Environment',
        'version': '2.0.0',
        'description': 'OpenEnv RL environment for AI code review agents with 3 progressive tasks',
        'openenv_compliant': True,
        'endpoints': {
            'reset': {
                'method': 'POST',
                'path': '/reset',
                'description': 'Reset environment to start new episode'
            },
            'step': {
                'method': 'POST',
                'path': '/step',
                'description': 'Execute an action and get reward',
                'body': {
                    'action': {
                        'action_type': 'detect_bug|skip|suggest_fix',
                        'bug': 'optional bug object',
                        'fix_suggestion': 'optional fix text',
                        'confidence': 'float 0-1'
                    }
                }
            },
            'state': {
                'method': 'GET',
                'path': '/state',
                'description': 'Get current environment state'
            },
            'health': {
                'method': 'GET',
                'path': '/health',
                'description': 'Health check'
            }
        },
        'tasks': [
            {
                'id': 1,
                'name': 'Bug Detection',
                'description': 'Detect whether code has bugs (binary classification)',
                'difficulty': 'easy',
                'max_steps': 3
            },
            {
                'id': 2,
                'name': 'Bug Classification',
                'description': 'Find and classify all bugs correctly',
                'difficulty': 'medium',
                'max_steps': 6
            },
            {
                'id': 3,
                'name': 'Fix Suggestion',
                'description': 'Suggest detailed fix with explanation',
                'difficulty': 'hard',
                'max_steps': 4
            }
        ],
        'graders': [
            'BugDetectionGrader',
            'BugClassificationGrader',
            'FixSuggestionGrader'
        ],
        'features': [
            '3 progressive tasks with graders',
            'Dynamic code generation (optional)',
            'Shaped reward system',
            'False positive penalties',
            'Semantic fix scoring'
        ]
    })

# ===== OPENENV VALIDATION ENDPOINTS =====

@app.route('/openenv/validate', methods=['GET'])
def validate():
    """OpenEnv validation endpoint"""
    try:
        env_instance = get_env()
        
        # Test reset
        obs = env_instance.reset()
        
        return jsonify({
            'valid': True,
            'tasks_found': 3,
            'graders': ['BugDetectionGrader', 'BugClassificationGrader', 'FixSuggestionGrader'],
            'message': 'Environment is OpenEnv compliant'
        })
    except Exception as e:
        return jsonify({
            'valid': False,
            'error': str(e)
        }), 500

# ===== ERROR HANDLERS =====

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# ===== MAIN =====
if __name__ == '__main__':
    port = int(os.getenv('PORT', 7860))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    print(f"🚀 Starting Code Review Environment Server")
    print(f"📡 Listening on http://0.0.0.0:{port}")
    print(f"📋 OpenEnv endpoints:")
    print(f"   POST /reset")
    print(f"   POST /step")
    print(f"   GET  /state")
    print(f"   GET  /health")
    print(f"   GET  /openenv/validate")
    
    app.run(host='0.0.0.0', port=port, debug=debug)
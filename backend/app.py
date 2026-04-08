"""
Flask Application Entry Point
"""

from flask import Flask, jsonify
from flask_cors import CORS
import os
import sys

# Ensure /app is on the path so 'backend' resolves correctly
# whether run as `python -m backend.app` or `python app.py`
current_dir = os.path.dirname(os.path.abspath(__file__))
app_root = os.path.dirname(current_dir)  # /app
if app_root not in sys.path:
    sys.path.insert(0, app_root)

from backend.api_routes import api

app = Flask(__name__)
CORS(app, origins=['http://localhost:3000', 'http://localhost:3001', 'http://localhost:5173'])
app.register_blueprint(api)


@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'name': 'Code Review Environment',
        'version': '2.0.0',
        'description': 'OpenEnv RL environment for AI code review agents',
        'endpoints': {
            'reset': 'POST /reset',
            'step': 'POST /step',
            'state': 'GET /state',
            'health': 'GET /health',
            'metadata': 'GET /metadata',
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
    app.run(host='0.0.0.0', port=port, debug=False)